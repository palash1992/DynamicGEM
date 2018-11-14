from __future__ import print_function
from __future__ import print_function

import keras.backend as K
from dynamicgem.dynamictriad.core.kerasext import keras_backend_patches
from keras import optimizers, constraints
import numpy as np
import math
import warnings
import sys
from dynamicgem.dynamictriad.core.algorithm.samplers.pos_neg_tri import Sampler
try:
    from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
except ImportError:
    from sklearn.cross_validation import cross_val_score, KFold, StratifiedKFold
from dynamicgem.dynamictriad.core import utils
from dynamicgem.dynamictriad.core import gconfig as gconf
from dynamicgem.dynamictriad.core.algorithm.embutils import TrainFlow, WithData, Validator

try:
    import dynamicgem.dynamictriad.core.algorithm.dynamic_triad_cimpl as cimpl
except ImportError:
    warnings.warn("dynamic_triad_cimpl.so not found, falling back to python implementation")
    cimpl = None


class Model(Sampler, TrainFlow, WithData, Validator):
    def __init__(self, ds, pretrain_size=10, embdim=16, beta=None,
                 lr=0.1, batchsize=None, sampling_args=None):
        if beta is None:
            beta = [0.1, 0.1]
        if sampling_args is None:
            sampling_args = {}

        self.__dataset = ds

        TrainFlow.__init__(self, embdim=embdim, beta=beta, trainmod=self.name, datasetmod=ds.name)
        Sampler.__init__(self, **sampling_args)

        self.pretrain_size = pretrain_size
        self.lr = lr
        self.batchsize = batchsize

        self.__pretrain = None
        self.__online = None

    @property
    def name(self):
        return "dynamic_triad"

    @property
    def dataset(self):
        return self.__dataset

    @property
    def pretrain(self):
        if self.__pretrain is not None:
            return self.__pretrain
        lf, pr, vs, cache = self.make_pretrain()
        self.__pretrain = {'lossfunc': lf, 'predfunc': pr, 'vars': vs, 'cache': cache}
        return self.__pretrain

    @property
    def online(self):
        if self.__online is not None:
            return self.__online
        lf, pr, vs, cache = self.make_online()
        self.__online = {'lossfunc': lf, 'predfunc': pr, 'vars': vs, 'cache': cache}
        return self.__online

    def verbose(self, inputs):
        comp = self.pretrain['cache']['debug']
        return comp(inputs)

    # the current implementation of x function is (j - i) + (k - i)
    def make_pretrain(self):
        embedding = K.variable(
            np.random.uniform(0, 1, (self.pretrain_size, self.dataset.nsize, self.flowargs['embdim'])))
        theta = K.variable(np.random.uniform(0, 1, (self.flowargs['embdim'] + 1, )))
        data = K.placeholder(ndim=2, dtype='int32')  # (batchsize, 5), [k, from_pos, to_pos, from_neg, to_neg]
        weight = K.placeholder(ndim=1, dtype='float32')  # (batchsize, )
        triag_int = K.placeholder(ndim=2, dtype='int32')  # (batchsize, 4), [k, from, to1, to2]
        triag_float = K.placeholder(ndim=2, dtype='float32')  # (batchsize, 3), [coef, w1, w2]
        pred_data = K.placeholder(ndim=2, dtype='int32')  # (batchsize, 2)  [timestep, nodeid]

        if K._BACKEND == 'theano':
            # (batchsize, nsize, d) => (batchsize, nsize)
            pred = embedding[pred_data[:, 0] - 1, pred_data[:, 1]][:, None, :] - embedding[pred_data[:, 0] - 1]
            pred = -K.sum(K.square(pred), axis=-1)  # the closer the more probable

            # (batchsize, d) => (batchsize, )
            dist_pos = embedding[data[:, 0], data[:, 1]] - embedding[data[:, 0], data[:, 2]]
            dist_pos = K.sum(dist_pos * dist_pos, axis=-1)
            dist_neg = embedding[data[:, 0], data[:, 3]] - embedding[data[:, 0], data[:, 4]]
            dist_neg = K.sum(dist_neg * dist_neg, axis=-1)
        else:
            pred_tm = K.slice(pred_data, [0, 0], [-1, 1]) - 1
            node_idx = K.concatenate((pred_tm, K.slice(data, [0, 1], [-1, 1])), axis=1)
            pred = K.expand_dims(K.gather_nd(embedding, node_idx), 1) - K.gather(embedding, K.squeeze(pred_tm, 1))
            pred = -K.sum(K.square(pred), axis=-1)

            tm = K.slice(data, [0, 0], [-1, 1])
            posedge1 = K.concatenate((tm, K.slice(data, [0, 1], [-1, 1])), axis=1)
            posedge2 = K.concatenate((tm, K.slice(data, [0, 2], [-1, 1])), axis=1)
            negedge1 = K.concatenate((tm, K.slice(data, [0, 3], [-1, 1])), axis=1)
            negedge2 = K.concatenate((tm, K.slice(data, [0, 4], [-1, 1])), axis=1)
            dist_pos = K.gather_nd(embedding, posedge1) - K.gather_nd(embedding, posedge2)
            dist_pos = K.sum(dist_pos * dist_pos, axis=-1)
            dist_neg = K.gather_nd(embedding, negedge1) - K.gather_nd(embedding, negedge2)
            dist_neg = K.sum(dist_neg * dist_neg, axis=-1)

        margin = 1
        lprox = K.maximum(dist_pos - dist_neg + margin, 0) * weight

        # (1, )
        lprox = K.mean(lprox)

        # lsmooth
        lsmooth = embedding[1:] - embedding[:-1]  # (k - 1, nsize, d)
        lsmooth = K.sum(K.square(lsmooth), axis=-1)  # (k - 1, nsize)
        lsmooth = K.mean(lsmooth)

        # ltriag
        if K._BACKEND == 'theano':
            e1 = embedding[triag_int[:, 0], triag_int[:, 1]] - embedding[triag_int[:, 0], triag_int[:, 2]]  # (batchsize_t, d)
            e2 = embedding[triag_int[:, 0], triag_int[:, 1]] - embedding[triag_int[:, 0], triag_int[:, 3]]
            x = e1 * triag_float[:, 1, None] + e2 * triag_float[:, 2, None]
            iprod = K.dot(x, K.expand_dims(theta[:-1], axis=1)) + theta[-1]  # (batchsize_d, )
            iprod = K.clip(iprod, -50, 50)  # for numerical stability
            logprob = K.log(1 + K.exp(-iprod))
            ltriag = K.mean(triag_float[:, 0] * iprod + logprob)
        else:
            tm = K.slice(triag_int, [0, 0], [-1, 1])
            nc = K.concatenate((tm, K.slice(triag_int, [0, 1], [-1, 1])), axis=1)
            n1 = K.concatenate((tm, K.slice(triag_int, [0, 2], [-1, 1])), axis=1)
            n2 = K.concatenate((tm, K.slice(triag_int, [0, 3], [-1, 1])), axis=1)
            
            e1 = K.gather_nd(embedding, nc) - K.gather_nd(embedding, n1)
            e2 = K.gather_nd(embedding, nc) - K.gather_nd(embedding, n2)

            w1 = K.slice(triag_float, [0, 1], [-1, 1])
            w2 = K.slice(triag_float, [0, 2], [-1, 1])

            x = e1 * w1 + e2 * w2
            iprod = K.dot(x, K.expand_dims(theta[:-1], axis=1)) + theta[-1]  # (batchsize_d, )
            # logprob = K.log(1 + K.exp(-iprod))
            logprob = -K.log_softmax(K.concatenate((iprod, K.zeros_like(iprod)), axis=1), axis=1)
            logprob = K.slice(logprob, [0, 0], [-1, 1])  # discard results for appended zero line
            logprob = K.clip(logprob, -50, 50)  # if the softmax if too small

            coef = K.slice(triag_float, [0, 0], [-1, 1])
            ltriag = K.mean(coef * iprod + logprob)

        loss = lprox + self.flowargs['beta'][0] * lsmooth + self.flowargs['beta'][1] * ltriag

        opt = optimizers.get({'class_name': 'Adagrad', 'config': {'lr': self.lr}})
        cstr = {embedding: constraints.get({'class_name': 'maxnorm', 'config': {'max_value': 1, 'axis': 2}}),
                theta: constraints.get({'class_name': 'unitnorm', 'config': {'axis': 0}})}
        upd = opt.get_updates([embedding, theta], cstr, loss)
        lf = K.function([data, weight, triag_int, triag_float], [loss], updates=upd)
        pf = K.function([pred_data], [pred])
        
        if gconf.debug:
            debug = K.function([data, weight, triag_int, triag_float],
                               [lprox, lsmooth * self.flowargs['beta'][0], ltriag * self.flowargs['beta'][1],
                               K.mean(triag_float[:, 0]) * self.flowargs['beta'][1],
                               K.mean(iprod) * self.flowargs['beta'][1],
                               K.mean(logprob) * self.flowargs['beta'][1]])
            return lf, pf, [embedding, theta], {'debug': debug}
        else:
            return lf, pf, [embedding, theta], {}

    def make_online(self):
        embedding = K.variable(np.random.uniform(0, 1, (self.dataset.nsize, self.flowargs['embdim'])))
        prevemb = K.placeholder(ndim=2, dtype='float32')  # (nsize, d)
        data = K.placeholder(ndim=2, dtype='int32')  # (batchsize, 5), [k, from_pos, to_pos, from_neg, to_neg]
        weight = K.placeholder(ndim=1, dtype='float32')  # (batchsize, )

        if K._BACKEND == 'theano':
            # (batchsize, d) => (batchsize, )
            # data[:, 0] should be always 0, so we simply ignore it
            # note, when you want to use it, that according to data generation procedure, the actual data[:, 0] is not 0
            dist_pos = embedding[data[:, 1]] - embedding[data[:, 2]]
            dist_pos = K.sum(dist_pos * dist_pos, axis=-1)
            dist_neg = embedding[data[:, 3]] - embedding[data[:, 4]]
            dist_neg = K.sum(dist_neg * dist_neg, axis=-1)
        else:
            dist_pos = K.gather(embedding, K.squeeze(K.slice(data, [0, 1], [-1, 1]), axis=1)) - \
                       K.gather(embedding, K.squeeze(K.slice(data, [0, 2], [-1, 1]), axis=1))
            dist_pos = K.sum(dist_pos * dist_pos, axis=-1)
            dist_neg = K.gather(embedding, K.squeeze(K.slice(data, [0, 3], [-1, 1]), axis=1)) - \
                       K.gather(embedding, K.squeeze(K.slice(data, [0, 4], [-1, 1]), axis=1))
            dist_neg = K.sum(dist_neg * dist_neg, axis=-1)

        # (batchsize, )
        margin = 1
        lprox = K.maximum(margin + dist_pos - dist_neg, 0) * weight

        # (1, )
        lprox = K.mean(lprox)

        # lsmooth
        lsmooth = embedding - prevemb  # (nsize, d)
        lsmooth = K.sum(K.square(lsmooth), axis=-1)  # (nsize)
        lsmooth = K.mean(lsmooth)

        loss = lprox + self.flowargs['beta'][0] * lsmooth

        opt = optimizers.get({'class_name': 'Adagrad', 'config': {'lr': self.lr}})
        cstr = {embedding: constraints.get({'class_name': 'maxnorm', 'config': {'max_value': 1, 'axis': 1}})}
        upd = opt.get_updates([embedding], cstr, loss)
        lf = K.function([data, weight, prevemb], [loss], updates=upd)

        return lf, None, [embedding], {}

    def save_model(self, copy=True):
        if self.cur_train_begin < self.init_train_begin + self.pretrain_size < self.cur_train_end:
            raise RuntimeError("current training process crosses the boarder of pretraining???")
        # load from keras resources
        if self.cur_train_end <= self.init_train_begin + self.pretrain_size:
            self._sequence[self.init_train_begin:self.init_train_begin + self.pretrain_size] = K.get_value(self.pretrain['vars'][0])
            self._tagged['theta'] = K.get_value(self.pretrain['vars'][1])
        else:
            self._sequence[self.cur_train_begin] = K.get_value(self.online['vars'][0])
        return super(Model, self).save_model(copy=copy)

    def restore_model(self, model, begin=None, end=None, copy=True):
        super(Model, self).restore_model(model, begin, end, copy=copy)
        if begin is None:
            begin = self.cur_train_begin
        if end is None:
            end = self.cur_train_end
        if self.cur_train_begin < self.init_train_begin + self.pretrain_size < self.cur_train_end:
            raise RuntimeError("current training process crosses the boarder of pretraining???")

        # store to keras resources
        if self.is_training:
            if end <= self.init_train_begin + self.pretrain_size:
                K.set_value(self.pretrain['vars'][0], self._sequence[self.init_train_begin:self.init_train_begin + self.pretrain_size])
                K.set_value(self.pretrain['vars'][1], self._tagged['theta'])
            else:
                K.set_value(self.online['vars'][0], self._sequence[begin])

    def pretrain_begin(self, begin, end):
        TrainFlow.clear(self)
        TrainFlow.start_training(self, begin, end)

        Sampler.pretrain_begin(self, begin, end)

        self._sequence.extend([None] * self.pretrain_size)

    def pretrain_begin_iteration(self):
        Sampler.pretrain_begin_iteration(self)

        if self.cur_train_end > self.init_train_begin + self.pretrain_size:  # online phase
            return

        # compute EM coefficients here
        neg1_int, neg1_float = self.__emcoef(self._neg[1])

        self._neg = self._neg[:1] + [neg1_int, neg1_float]

    def pretrain_end_iteration(self):
        Sampler.pretrain_end_iteration(self)
        self.save_model()

    def pretrain_end(self):
        Sampler.pretrain_end(self)
        TrainFlow.stop_training(self)

    def online_begin(self, begin, end):
        TrainFlow.start_training(self, begin, end)

        Sampler.online_begin(self, begin, end)
        initv = np.random.uniform(0, 1, (self.dataset.nsize, self.flowargs['embdim'])).astype('float32')
        K.set_value(self.online['vars'][0], initv)
        self._sequence.append(None)

    # ends the current online training
    # store online training results
    # we need to reset online training variables
    def online_end(self):
        Sampler.online_end(self)
        assert self.cur_train_end == self.cur_train_begin + 1, "{} {}".format(self.cur_train_end, self.cur_train_begin)
        self.save_model()
        TrainFlow.stop_training(self)

    def make_pretrain_input(self, batch):
        ret = Sampler.make_pretrain_input(self, batch)
        # (data, weight, triad)
        # because embedding variable starts from index 0
        for d in ret[0]:  # data
            d[0] -= self.init_train_begin
        for d in ret[2]:
            d[0] -= self.init_train_begin
        return ret

    # return a list whatever number of inputs required
    def make_online_input(self, batch):
        ret = Sampler.make_online_input(self, batch)

        for d in ret[0]:  # data
            d[0] -= self.init_train_begin
        
        ret.append(self._sequence[self.cur_train_begin - 1])
        return ret

    def __emcoef_cimpl(self, data):
        nodenames = list(self.__dataset.gtgraphs['any'].vp['name'])
        emb, theta = [K.get_value(v) for v in self.pretrain['vars']]
        mygraphs = list(self.__dataset.mygraphs)
        
        res = cimpl.emcoef(data, emb, theta, mygraphs, nodenames, self.__dataset.localstep)
        neg1_int = [r[0] for r in res]
        neg1_float = [r[1] for r in res]

        return neg1_int, neg1_float

    def __emcoef_pyimpl(self, data):
        slices = utils.group_by(self._neg[1], key=lambda x: x[0])
        for i in range(len(slices)):
            mapper = utils.ParMap(self.__emcoef_calculator_factory(slices[i][0][0]), self.__emcoef_monitor)
            slices[i] = mapper.run(slices[i], chunk=min(10000, mapper.default_chunk(len(slices[i]))))

        # TODO: need a shuffle?
        neg1_int = [r[0] for s in slices for r in s]
        neg1_float = [r[1] for s in slices for r in s]

        assert len(neg1_int) == len(neg1_float) and len(neg1_int) == len(data), \
            "{} {} {}".format(len(neg1_int), len(neg1_float), len(data))
         
        return neg1_int, neg1_float

    def __emcoef(self, data):
        if cimpl is not None:
            if gconf.debug:
                data = [r for s in utils.group_by(self._neg[1], key=lambda x: x[0]) for r in s]
            cimpl_res = self.__emcoef_cimpl(data)
            if gconf.debug:
                pyimpl_res = self.__emcoef_pyimpl(data)
                print("checking cimpl results according to pyimpl results")
                assert len(pyimpl_res[0]) == len(cimpl_res[0]), "{} {}".format(len(pyimpl_res[0]), len(cimpl_res[0]))
                for i in range(len(pyimpl_res[0])):
                    assert pyimpl_res[0][i] == cimpl_res[0][i] and [math.fabs(i - j) < 1e-3 for i, j in zip(pyimpl_res[1][i], cimpl_res[1][i])], \
                        "{} {} {} {} {} {}".format(i, pyimpl_res[0][i], pyimpl_res[1][i], cimpl_res[0][i], cimpl_res[1][i], data[i])
                print("OK")

            return cimpl_res 
        else:
            return self.__emcoef_pyimpl(data)

    @staticmethod
    def __emcoef_monitor(reportq):
        total_proccnt = {}
        while True:
            obj = reportq.get()
            if isinstance(obj, StopIteration):
                break
            pid, proccnt = obj
            total_proccnt[pid] = proccnt
            print("EM coefficients calculated for {} samples\r".format(sum(total_proccnt.values())), end='')
            sys.stdout.flush()
        print("EM coefficients calculated for {} samples".format(sum(total_proccnt.values())))

    def __emcoef_calculator_factory(self, timestep):
        # TODO: split data by year so that we need not share the whole emb and mygraph
        nodenames = list(self.__dataset.gtgraphs['any'].vp['name'])
        name2idx = {n: i for i, n in enumerate(nodenames)}
        emb, theta = [K.get_value(v) for v in self.pretrain['vars']]
        # localstep = self.__dataset.localstep

        # localize for current time step
        g = self.__dataset.mygraphs[timestep]
        emb = emb[timestep - self.__dataset.localstep]
        # mygraphs = list(self.__dataset.mygraphs)

        def emcoef_calc(procinfo, data, reportq):
            ret = []
            for didx, d in enumerate(data):
                if didx % 10000 == 0:
                    reportq.put([id(procinfo), didx])
                y, k, i, j, lb, wtv1, wtv2 = d
                # y0based = y - localstep
                # g = mygraphs[y0based]
                # w = g.edge_properties['weight']
                if lb == 0:
                    C = 1
                else:
                    def x(a, b, c):
                        w1, w2 = g.edge(nodenames[a], nodenames[c]), g.edge(nodenames[b], nodenames[c])
                        return (emb[c] - emb[a]) * w1 + (emb[c] - emb[b]) * w2

                    def P(a, b, c):
                        power = -(np.dot(theta[:-1], x(a, b, c)) + theta[-1])

                        if power > 100:
                            return 0
                        else:
                            return 1.0 / (1 + math.exp(power))

                    C0 = P(i, j, k)

                    inbr = set(list(g.out_neighbours(nodenames[i])))
                    jnbr = set(list(g.out_neighbours(nodenames[j])))
                    cmnbr = inbr.intersection(jnbr)
                    C1 = 1 - np.prod([1 - P(i, j, name2idx[v]) for v in cmnbr])

                    eps = 1e-6
                    C = 1 - C0 / (C1 + eps)

                    if not np.isfinite(C):
                        print(C0, C1, C, [1 - P(i, j, name2idx[v]) for v in cmnbr])
                        print(i, j, k)
                        print(g.exists(nodenames[i], nodenames[k]),
                              g.exists(nodenames[j], nodenames[k]))
                        print([name2idx[n] for n in inbr], [name2idx[n] for n in jnbr])
                        assert 0

                ret.append(([y, k, i, j], [C, wtv1, wtv2]))
            reportq.put([id(procinfo), len(data)])
            return ret

        return emcoef_calc
