from __future__ import print_function
disp_avlbl = True
import os
if os.name == 'posix' and 'DISPLAY' not in os.environ:
    disp_avlbl = False
    import matplotlib

    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import sys
sys.path.append('./')
sys.path.append(os.path.realpath(__file__))
from .static_graph_embedding import StaticGraphEmbedding
from dynamicgem.utils import graph_util, plot_util, dataprep_util
from dynamicgem.evaluation import visualize_embedding as viz
from .sdne_utils import *
from keras import backend as KBack
import tensorflow as tf
import argparse
from dynamicgem.graph_generation import dynamic_SBM_graph
import operator
import time
from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from dynamicgem.dynamictriad.core import *
from six.moves import cPickle
import importlib
from os.path import isfile
import dynamicgem.dynamictriad.core.dataset.dataset_utils as du
import dynamicgem.dynamictriad.core.algorithm.embutils as eu
from dynamicgem.evaluation import evaluate_link_prediction as lp
import pdb
from sklearn.linear_model import LogisticRegression
import random

try:
    from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
except ImportError:
    from sklearn.cross_validation import cross_val_score, KFold, StratifiedKFold
from sklearn import svm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


class dynamicTriad(StaticGraphEmbedding):
    """ Initialize the embedding class
        Args:
       t    : Type of data to test the code
       nm   : number of nodes to migrate
       iter : number of optimization iterations
       m    : argparse.SUPPRESS
       d    : input directory name
       b    : batchsize for training
       n    : number of time steps
       K    : number of embedding dimensions
       l    : size of of a time steps
       s    : interval between two time steps
       o    : output directory name
       rd   : result directory name
       lr   : initial learning rate
       beta-smooth : coefficients for smooth component
       beta-triad  : coefficients for triad component
       negdup      : neg/pos ratio during sampling"
       datasetmod  : help='module name for dataset loading
       dataname    : name for the current data file
       validation  : 'link_reconstruction'
       te   : 'type of test, (node_classify, node_predict, link_classify, link_predict, '
                             'changed_link_classify, changed_link_predict, all)')
       classifier  : lr, svm
       repeat      : number of times to repeat experiment
       sm   : samples for test data
    """

    def __init__(self, d, *hyper_dict, **kwargs):
        self._d = d
        hyper_params = {
            'method_name': 'Dynamic TRIAD',
            'modelfile': None,
            'weightfile': None,
            'savefilesuffix': None

        }
        hyper_params.update(kwargs)
        for key in hyper_params.keys():
            self.__setattr__('_%s' % key, hyper_params[key])
        for dictionary in hyper_dict:
            for key in dictionary:
                self.__setattr__('_%s' % key, dictionary[key])
        self.clf = self.__make_classifier()
        self._model = None
        # self._clname='lr'       

    def __make_classifier(self):
        class_weight = 'balanced'

        if self._clname == 'svm':
            return svm.SVC(kernel='linear', class_weight=class_weight)
        elif self._clname == 'lr':
            return LogisticRegression(class_weight=class_weight)
        else:
            raise NotImplementedError()

    def load_trainmod(self, modname):
        mod = importlib.import_module(modname)
        return getattr(mod, 'Model')

    def load_datamod(self, modname):
        mod = importlib.import_module(modname)
        return getattr(mod, 'Dataset')

    def load_or_update_cache(self, ds, cachefn):
        if cachefn is None:
            return
        cachefn += '.cache'
        if isfile(cachefn + '.args'):
            args = cPickle.load(open(cachefn + '.args', 'r'))
            try:
                ds.load_cache(args, lambda: cPickle.load(open(cachefn, 'r')))
                print("Data loaded from cache file {}".format(cachefn))
                return
            except (ValueError, EOFError) as e:
                print("Failed to load cache file {}: {}".format(cachefn, e.message))

        # update cache
        print("updating cache file for prefix {}".format(cachefn))
        ar, args = ds.cache()
        cPickle.dump(args, open(cachefn + '.args', 'w'))
        cPickle.dump(ar, open(cachefn, 'w'))
        print("cache file {} updated".format(cachefn))

    def export(self, vertices, data, outdir):

        outdir = outdir + '/' + self._datatype
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        outdir = outdir + '/dynTriad'
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        for i in range(len(data)):
            assert len(vertices) == len(data[i]), (len(vertices), len(data[i]))
            fn = "{}/{}.out".format(outdir, i)
            fh = open(fn, 'w')
            for j in range(len(vertices)):
                print("{} {}".format(vertices[j], ' '.join(["{:.3f}".format(d) for d in data[i][j]])), file=fh)
            fh.close()

    def load_embedding(self, fn, vs):
        data = open(fn, 'r').read().rstrip('\n').split('\n')
        emb = {}
        for line in data:
            fields = line.split()
            emb[fields[0]] = [float(e) for e in fields[1:]]
        # it is possible that the output order differs from :param vs: given different node_type,
        # so we have to reorder the embedding according to :param vs:
        emb = [emb[str(v)] for v in vs]

        return np.vstack(emb)

    def get_method_name(self):
        return self._method_name

    def get_method_summary(self):
        return '%s_%d' % (self._method_name, self._d)

    def learn_embedding(self):

        # TensorFlow wizardry
        config = tf.ConfigProto()

        # Don't pre-allocate memory; allocate as-needed
        config.gpu_options.allow_growth = True

        # Only allow a total of half the GPU memory to be allocated
        config.gpu_options.per_process_gpu_memory_fraction = 0.2

        # Create a session to pass the above configuration
        sess = tf.Session(config=config)

        # Create a tensorflow debugger wrapper
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess) 

        # Create a session with the above options specified.
        KBack.tensorflow_backend.set_session(sess)

        TrainModel = self.load_trainmod(self._trainmod)
        Dataset = self.load_datamod(self._datasetmod)

        ds = Dataset(self._datafile, self._starttime, self._nsteps, stepsize=self._stepsize,
                     stepstride=self._stepstride)
        #         self.load_or_update_cache(ds, self._cachefn)
        # dsargs = {'datafile': self._datafile, 'starttime': self._starttime, 'nsteps': self._nsteps,
        #           'stepsize': self._stepsize, 'stepstride': self._stepstride, 'datasetmod': self._datasetmod}
        tm = TrainModel(ds, pretrain_size=self._pretrain_size, embdim=self._embdim, beta=self._beta,
                        lr=self._lr, batchsize=self._batchsize, sampling_args=self._sampling_args)

        edgecnt = [g.num_edges() for g in ds.gtgraphs]
        k_edgecnt = sum(edgecnt[:self._pretrain_size])
        print("{} edges in pretraining graphs".format(k_edgecnt))

        if self._pretrain_size > 0:
            initstep = int(ds.time2step(self._starttime))
            tm.pretrain_begin(initstep, initstep + self._pretrain_size)

            print("generating validation set")
            validargs = tm.dataset.sample_test_data(self._validation, initstep, initstep + self._pretrain_size,
                                                    size=10000)
            # print(validargs)
            print("{} validation samples generated".format(len(validargs[0])))

            max_val, max_idx, maxmodel = -1, 0, None

            # for early stopping
            start_time = time.time()
            scores = []
            for i in range(self._niters):
                tm.pretrain_begin_iteration()

                epoch_loss = 0
                for batidx, bat in enumerate(tm.batches(self._batchsize)):
                    inputs = tm.make_pretrain_input(bat)
                    l = tm.pretrain['lossfunc'](inputs)
                    if isinstance(l, (list, tuple)):
                        l = l[0]
                    epoch_loss += l
                    print("\repoch {}: {:.0%} completed, cur loss: {:.3f}".format(i, float(batidx * self._batchsize)
                                                                                  / tm.sample_size(), l.flat[0]),
                          end='')
                    sys.stdout.flush()
                tm.pretrain_end_iteration()

                print(" training completed, total loss {}".format(epoch_loss), end='')

                # without validation, the model exists only after I iterations
                if self._validation != 'none':
                    val_score = tm.validate(self._validation, *validargs)

                    if val_score > max_val:
                        max_val = val_score
                        max_idx = i
                        maxmodel = tm.save_model()
                    print(", validation score {:.3f}".format(val_score))
                else:
                    max_idx, max_val = i, epoch_loss
                    # maxmodel is not saved here in order to save time
                    print("")

                # checkpoint disabled
                # if i % 5 == 0:
                #     lastmodel = tm.save_model()
                #     if args.validation == 'none':
                #         maxmodel = lastmodel
                #
                #     tm.restore_model(maxmodel)  # restore parameters while preserving other info
                #     cPickle.dump([tm.archive(), dsargs, lastmodel], open(self._outdir, 'w'))
                #     tm.restore_model(lastmodel)

                if self._validation != 'none':
                    scores.append(val_score)
                    if max_val > 0 and i - max_idx > 5:
                        break

            print("best validation score at itr {}: {}".format(max_idx, max_val))
            print("{} seconds elapsed for pretraining".format(time.time() - start_time))
            # lastmodel = tm.save_model()  # for debug
            print("saving output to {}".format(self._outdir))
            tm.restore_model(maxmodel)
            tm.pretrain_end()
            self.export(tm.dataset.mygraphs['any'].vertices(), tm.export(), self._outdir)

        # online training disabled
        startstep = int(tm.dataset.time2step(self._starttime))
        for y in range(startstep + self._pretrain_size, startstep + self._nsteps):
            raise NotImplementedError()

    def get_embedding(self):
        self._X = dataprep_util.getemb_dynTriad(self._outdir + '/' + self._testDataType + '/dynTriad', self._nsteps,
                                                self._embdim)
        return self._X

    def get_edge_weight(self, t, i, j):
        try:
            feat = np.fabs(self._X[t][i, :] - self._X[t][j, :])
            # val= 1/(1+np.mean(np.fabs(self._X[t][i,:]- self._X[t][j,:])))
            # val= 1/(1+np.linalg.norm(self._X[t][i,:]- self._X[t][j,:]))
            # print(val)
            # pdb.set_trace()
            # return self._model.predict_proba(np.reshape(feat,[1,-1]))[0][1]
            return self._model.predict(np.reshape(feat, [1, -1]))[0]
            # return val
        except:
            pdb.set_trace()

    def get_reconstructed_adj(self, t, X=None, node_l=None):
        if X is not None:
            node_num = X.shape[0]
            # self._X = X
        else:
            node_num = self._node_num
        adj_mtx_r = np.zeros((node_num, node_num))
        for v_i in range(node_num):
            for v_j in range(node_num):
                if v_i == v_j:
                    continue
                adj_mtx_r[v_i, v_j] = self.get_edge_weight(t, v_i, v_j)
        return adj_mtx_r

    def sample_link_reconstruction(self, g, sample_nodes=None, negdup=1):
        pos = []
        # assert not g.is_directed()
        # for g in graphs:
        for e in g.edges():
            if int(e[0]) > int(e[1]):
                # check symmetric
                names = list(g.nodes())
                assert g.edges(e[0], e[1]), "{}: {} {}".format(names[e[0]],
                                                               names[e[1]])
                continue
            pos.append([int(e[0]), int(e[1])])
        pos = np.vstack(pos).astype('int32')

        neg = []
        vsize = len(g.nodes())
        nodenames = list(g.nodes())
        for i in range(negdup):
            for p in pos:
                src, tgt = p
                # g = self.mygraphs[tm + intv]
                assert g.out_degree(nodenames[src]) < vsize - 1 or g.out_degree(nodenames[tgt]) < vsize - 1, \
                    "We do not expect any node to connect to all other nodes"

                while True:
                    if random.randint(0, 1) == 0:  # replace source
                        # cur_range = negrange[tm][tgt]
                        # new_src = cur_range[random.randint(0, len(cur_range) - 1)]
                        new_src = random.randint(0, vsize - 1)
                        if not g.has_edge(nodenames[new_src], nodenames[tgt]):
                            neg.append([new_src, tgt])
                            break
                    else:  # replace target
                        # cur_range = negrange[tm][src]
                        # new_tgt = cur_range[random.randint(0, len(cur_range) - 1)]
                        new_tgt = random.randint(0, vsize - 1)
                        if not g.has_edge(nodenames[src], nodenames[new_tgt]):
                            neg.append([src, new_tgt])
                            break
        neg = np.vstack(neg).astype('int32')

        lbs = np.concatenate((np.ones(len(pos)), -np.ones(len(neg))))
        return np.concatenate((pos, neg), axis=0), lbs

    class ResultPresenter(object):
        def __init__(self):
            self.f1, self.prec, self.rec, self.acc = [], [], [], []

        def add_result(self, res):
            self.prec.extend(res[0])
            self.rec.extend(res[1])
            self.f1.extend(res[2])
            self.acc.extend(res[3])

        def show_result(self):
            print("precision mean: {} std: {}".format(np.mean(self.prec), np.std(self.prec)))
            print("recall mean: {} std: {}".format(np.mean(self.rec), np.std(self.rec)))
            print("f1 mean: {} std: {}".format(np.mean(self.f1), np.std(self.f1)))
            print("accuracy mean: {} std: {}".format(np.mean(self.acc), np.std(self.acc)))

    def __classify(self, feat, lbs):
        sm = None

        poscnt, negcnt = np.sum(lbs == 1), np.sum(lbs == -1)
        print("classifying with pos:neg = {}:{}".format(poscnt, negcnt))

        try:
            cv = StratifiedKFold(n_splits=5, shuffle=True)
            parts = cv.split(feat, lbs)
        except TypeError:
            cv = StratifiedKFold(lbs, n_folds=5, shuffle=True)
            parts = cv

        f1, prec, rec, acc = [], [], [], []
        for tr, te in parts:
            if sm is not None:
                x, y = sm.fit_sample(feat[tr], lbs[tr])
                # x, y = feat[tr], lbs[tr]
            else:
                x, y = feat[tr], lbs[tr]
            model = self.clf.fit(x, y)
            p = model.predict(feat[te])
            # self._model=model
            # if self.debug:
            #     print("results:", p, lbs[te])
            # print(p,np.shape(p))
            f1.append(f1_score(lbs[te], p))
            prec.append(precision_score(lbs[te], p))
            rec.append(recall_score(lbs[te], p))
            acc.append(accuracy_score(lbs[te], p))
        # idx = np.random.permutation(len(lbs))
        # x,y = feat[idx], lbs[idx]
        # self._model=self.clf.fit(x, y)    
        return prec, rec, f1, acc

    def link_predict(self, g, t, intv=0, repeat=1):
        samp, lbs = self.sample_link_reconstruction(g, sample_nodes=None, negdup=1)
        # pdb.set_trace()
        # TODO: different feature generation method might be used here
        try:
            feat = np.fabs(self._X[t][samp[:, 0]] - self._X[t][samp[:, 1]])
        except:
            pdb.set_trace()
        print("feature shape {}".format(feat.shape))

        # rp = self.ResultPresenter()
        # for i in range(repeat):
        #     res = self.__classify(feat, lbs)
        #     rp.add_result(res)
        # rp.show_result()

        idx = np.random.permutation(len(lbs))
        x, y = feat[idx], lbs[idx]
        self._model = self.clf.fit(x, y)

    def predict_next_adj(self, t, node_l=None):
        if node_l is not None:
            return self.get_reconstructed_adj(t, node_l)
        else:
            return self.get_reconstructed_adj(t)

    def plotresults(self, dynamic_sbm_series):
        plt.figure()
        plt.clf()
        viz.plot_static_sbm_embedding(self._X[-4:], dynamic_sbm_series[-4:])


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Learns static node embeddings')
    parser.add_argument('-t', '--testDataType',
                        default='sbm_cd',
                        type=str,
                        help='Type of data to test the code')
    parser.add_argument('-nm', '--nodemigration',
                        default=10,
                        type=int,
                        help='number of nodes to migrate')
    parser.add_argument('-iter', '--niters',
                        type=int,
                        help="number of optimization iterations",
                        default=20)
    parser.add_argument('-m', '--starttime',
                        type=str,
                        help=argparse.SUPPRESS,
                        default=0)
    parser.add_argument('-d', '--datafile',
                        type=str,
                        help='input directory name')
    parser.add_argument('-b', '--batchsize',
                        type=int,
                        help="batchsize for training",
                        default=1000)
    parser.add_argument('-n', '--nsteps',
                        type=int,
                        help="number of time steps",
                        default=10)
    parser.add_argument('-K', '--embdim',
                        type=int,
                        help="number of embedding dimensions",
                        default=128)
    parser.add_argument('-l', '--stepsize',
                        type=int,
                        help="size of of a time steps",
                        default=1)
    parser.add_argument('-s', '--stepstride',
                        type=int,
                        help="interval between two time steps",
                        default=1)
    parser.add_argument('-o', '--outdir',
                        type=str,
                        default='./output',
                        help="output directory name")
    parser.add_argument('-rd', '--resultdir',
                        type=str,
                        default='./results_link_all',
                        help="result directory name")
    parser.add_argument('--lr',
                        type=float,
                        help="initial learning rate",
                        default=0.1)
    parser.add_argument('--beta-smooth',
                        type=float,
                        default=0.1,
                        help="coefficients for smooth component")
    parser.add_argument('--beta-triad',
                        type=float,
                        default=0.1,
                        help="coefficients for triad component")
    parser.add_argument('--negdup',
                        type=int,
                        help="neg/pos ratio during sampling",
                        default=1)
    parser.add_argument('--datasetmod',
                        type=str,
                        default='core.dataset.adjlist',
                        help='module name for dataset loading',
                        )
    parser.add_argument('--validation',
                        type=str,
                        default='link_reconstruction',
                        help=', '.join(list(sorted(set(du.TestSampler.tasks) & set(eu.Validator.tasks)))))
    parser.add_argument('-te', '--test',
                        type=str,
                        nargs='+',
                        default='link_predict',
                        help='type of test, (node_classify, node_predict, link_classify, link_predict, '
                             'changed_link_classify, changed_link_predict, all)')
    parser.add_argument('--classifier',
                        type=str,
                        default='lr',
                        help='lr, svm')
    parser.add_argument('--repeat',
                        type=int,
                        default=1,
                        help='number of times to repeat experiment')
    parser.add_argument('-sm', '--samples',
                        default=5000,
                        type=int,
                        help='samples for test data')
    args = parser.parse_args()
    args.embdir = args.outdir + '/dynTriad/' + args.testDataType
    args.cachefn = '/tmp/' + args.testDataType
    args.beta = [args.beta_smooth, args.beta_triad]
    # some fixed arguments in published code
    args.pretrain_size = args.nsteps
    args.trainmod = 'dynamictriad.core.algorithm.dynamic_triad'
    args.sampling_args = {}
    args.debug = False
    args.scale = 1

    if args.validation not in du.TestSampler.tasks:
        raise NotImplementedError("Validation task {} not supported in TestSampler".format(args.validation))
    if args.validation not in eu.Validator.tasks:
        raise NotImplementedError("Validation task {} not supported in Validator".format(args.validation))

    print("running with options: ", args.__dict__)

    epochs = args.niters
    length = args.nsteps

    if args.testDataType == 'sbm_cd':
        node_num = 1000
        community_num = 2
        node_change_num = args.nodemigration
        dynamic_sbm_series = dynamic_SBM_graph.get_community_diminish_series_v2(node_num,
                                                                                community_num,
                                                                                length,
                                                                                1,
                                                                                node_change_num)
        graphs = [g[0] for g in dynamic_sbm_series]

        datafile = dataprep_util.prep_input_dynTriad(graphs, length, args.testDataType)

        embedding = dynamicTriad(niters=args.niters,
                                 starttime=args.starttime,
                                 datafile=datafile,
                                 batchsize=args.batchsize,
                                 nsteps=args.nsteps,
                                 embdim=args.embdim,
                                 stepsize=args.stepsize,
                                 stepstride=args.stepstride,
                                 outdir=args.outdir,
                                 cachefn=args.cachefn,
                                 lr=args.lr,
                                 beta=args.beta,
                                 negdup=args.negdup,
                                 datasetmod=args.datasetmod,
                                 trainmod=args.trainmod,
                                 pretrain_size=args.pretrain_size,
                                 sampling_args=args.sampling_args,
                                 validation=args.validation,
                                 datatype=args.testDataType,
                                 scale=args.scale,
                                 classifier=args.classifier,
                                 debug=args.debug,
                                 test=args.test,
                                 repeat=args.repeat,
                                 resultdir=args.resultdir,
                                 testDataType=args.testDataType,
                                 clname='lr',
                                 node_num=node_num )

        embedding.learn_embedding()
        embedding.get_embedding()
        # embedding.plotresults(dynamic_sbm_series)

        outdir = args.resultdir
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        outdir = outdir + '/' + args.testDataType
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        outdir = outdir + '/' + 'dynTRIAD'
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        lp.expstaticLP_TRIAD(dynamic_sbm_series,
                             graphs,
                             embedding,
                             1,
                             outdir + '/',
                             'nm' + str(args.nodemigration) + '_l' + str(args.nsteps) + '_emb' + str(args.embdim),
                             )


    elif args.testDataType == 'academic':
        print("datatype:", args.testDataType)

        sample = args.samples
        if not os.path.exists('./test_data/academic/pickle'):
            os.mkdir('./test_data/academic/pickle')
            graphs, length = dataprep_util.get_graph_academic('./test_data/academic/adjlist')
            for i in range(length):
                nx.write_gpickle(graphs[i], './test_data/academic/pickle/' + str(i))
        else:
            length = len(os.listdir('./test_data/academic/pickle'))
            graphs = []
            for i in range(length):
                graphs.append(nx.read_gpickle('./test_data/academic/pickle/' + str(i)))

        G_cen = nx.degree_centrality(graphs[29])  # graph 29 in academia has highest number of edges
        G_cen = sorted(G_cen.items(), key=operator.itemgetter(1), reverse=True)
        node_l = []
        i = 0
        while i < sample:
            node_l.append(G_cen[i][0])
            i += 1
        # pdb.set_trace()
        # node_l = np.random.choice(range(graphs[29].number_of_nodes()), 5000, replace=False)
        # print(node_l)
        for i in range(length):
            graphs[i] = graph_util.sample_graph_nodes(graphs[i], node_l)
        # pdb.set_trace()
        graphs = graphs[-args.nsteps:]
        datafile = dataprep_util.prep_input_dynTriad(graphs, args.nsteps, args.testDataType)

        embedding = dynamicTriad(niters=args.niters,
                                 starttime=args.starttime,
                                 datafile=datafile,
                                 batchsize=args.batchsize,
                                 nsteps=args.nsteps,
                                 embdim=args.embdim,
                                 stepsize=args.stepsize,
                                 stepstride=args.stepstride,
                                 outdir=args.outdir,
                                 cachefn=args.cachefn,
                                 lr=args.lr,
                                 beta=args.beta,
                                 negdup=args.negdup,
                                 datasetmod=args.datasetmod,
                                 trainmod=args.trainmod,
                                 pretrain_size=args.pretrain_size,
                                 sampling_args=args.sampling_args,
                                 validation=args.validation,
                                 datatype=args.testDataType,
                                 scale=args.scale,
                                 classifier=args.classifier,
                                 debug=args.debug,
                                 test=args.test,
                                 repeat=args.repeat,
                                 resultdir=args.resultdir,
                                 testDataType=args.testDataType,
                                 clname='lr',
                                 node_num=sample

                                 )
        embedding.learn_embedding()
        embedding.get_embedding()

        outdir = args.resultdir
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        outdir = outdir + '/' + args.testDataType
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        outdir = outdir + '/dynTriad'
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        lp.expstaticLP_TRIAD(None,
                             graphs,
                             embedding,
                             1,
                             outdir + '/',
                             'l' + str(args.nsteps) + '_emb' + str(args.embdim) + '_samples' + str(sample),
                             n_sample_nodes=sample
                             )


    elif args.testDataType == 'hep':
        print("datatype:", args.testDataType)

        if not os.path.exists('./test_data/hep/pickle'):
            os.mkdir('./test_data/hep/pickle')
            files = [file for file in os.listdir('./test_data/hep/hep-th') if '.gpickle' in file]
            length = len(files)
            graphs = []
            for i in range(length):
                G = nx.read_gpickle('./test_data/hep/hep-th/month_' + str(i + 1) + '_graph.gpickle')

                graphs.append(G)
            total_nodes = graphs[-1].number_of_nodes()

            for i in range(length):
                for j in range(total_nodes):
                    if j not in graphs[i].nodes():
                        graphs[i].add_node(j)

            for i in range(length):
                nx.write_gpickle(graphs[i], './test_data/hep/pickle/' + str(i))
        else:
            length = len(os.listdir('./test_data/hep/pickle'))
            graphs = []
            for i in range(length):
                graphs.append(nx.read_gpickle('./test_data/hep/pickle/' + str(i)))

        # pdb.set_trace()            
        sample = args.samples
        G_cen = nx.degree_centrality(graphs[-1])  # graph 29 in academia has highest number of edges
        G_cen = sorted(G_cen.items(), key=operator.itemgetter(1), reverse=True)
        node_l = []
        i = 0
        while i < sample:
            node_l.append(G_cen[i][0])
            i += 1
        for i in range(length):
            graphs[i] = graph_util.sample_graph_nodes(graphs[i], node_l)

        graphs = graphs[-args.nsteps:]
        datafile = dataprep_util.prep_input_dynTriad(graphs, args.nsteps, args.testDataType)

        embedding = dynamicTriad(niters=args.niters,
                                 starttime=args.starttime,
                                 datafile=datafile,
                                 batchsize=args.batchsize,
                                 nsteps=args.nsteps,
                                 embdim=args.embdim,
                                 stepsize=args.stepsize,
                                 stepstride=args.stepstride,
                                 outdir=args.outdir,
                                 cachefn=args.cachefn,
                                 lr=args.lr,
                                 beta=args.beta,
                                 negdup=args.negdup,
                                 datasetmod=args.datasetmod,
                                 trainmod=args.trainmod,
                                 pretrain_size=args.pretrain_size,
                                 sampling_args=args.sampling_args,
                                 validation=args.validation,
                                 datatype=args.testDataType,
                                 scale=args.scale,
                                 classifier=args.classifier,
                                 debug=args.debug,
                                 test=args.test,
                                 repeat=args.repeat,
                                 resultdir=args.resultdir,
                                 testDataType=args.testDataType,
                                 clname='lr',
                                 node_num=sample

                                 )
        embedding.learn_embedding()
        embedding.get_embedding()

        outdir = args.resultdir
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        outdir = outdir + '/' + args.testDataType
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        outdir = outdir + '/dynTriad'
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        lp.expstaticLP_TRIAD(None,
                             graphs,
                             embedding,
                             1,
                             outdir + '/',
                             'l' + str(args.nsteps) + '_emb' + str(args.embdim) + '_samples' + str(sample),
                             n_sample_nodes=sample
                             )


    elif args.testDataType == 'AS':
        print("datatype:", args.testDataType)

        files = [file for file in os.listdir('./test_data/AS/as-733') if '.gpickle' in file]
        length = len(files)
        graphs = []

        for i in range(length):
            G = nx.read_gpickle('./test_data/AS/as-733/month_' + str(i + 1) + '_graph.gpickle')
            graphs.append(G)

        sample = args.samples
        G_cen = nx.degree_centrality(graphs[-1])  # graph 29 in academia has highest number of edges
        G_cen = sorted(G_cen.items(), key=operator.itemgetter(1), reverse=True)
        node_l = []
        i = 0
        while i < sample:
            node_l.append(G_cen[i][0])
            i += 1
        for i in range(length):
            graphs[i] = graph_util.sample_graph_nodes(graphs[i], node_l)

        graphs = graphs[-args.nsteps:]
        datafile = dataprep_util.prep_input_dynTriad(graphs, args.nsteps, args.testDataType)

        embedding = dynamicTriad(niters=args.niters,
                                 starttime=args.starttime,
                                 datafile=datafile,
                                 batchsize=args.batchsize,
                                 nsteps=args.nsteps,
                                 embdim=args.embdim,
                                 stepsize=args.stepsize,
                                 stepstride=args.stepstride,
                                 outdir=args.outdir,
                                 cachefn=args.cachefn,
                                 lr=args.lr,
                                 beta=args.beta,
                                 negdup=args.negdup,
                                 datasetmod=args.datasetmod,
                                 trainmod=args.trainmod,
                                 pretrain_size=args.pretrain_size,
                                 sampling_args=args.sampling_args,
                                 validation=args.validation,
                                 datatype=args.testDataType,
                                 scale=args.scale,
                                 classifier=args.classifier,
                                 debug=args.debug,
                                 test=args.test,
                                 repeat=args.repeat,
                                 resultdir=args.resultdir,
                                 testDataType=args.testDataType,
                                 clname='lr',
                                 node_num=sample

                                 )

        embedding.learn_embedding()
        embedding.get_embedding()

        outdir = args.resultdir
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        outdir = outdir + '/' + args.testDataType
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        outdir = outdir + '/dynTriad'
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        lp.expstaticLP_TRIAD(None,
                             graphs,
                             embedding,
                             1,
                             outdir + '/',
                             'l' + str(args.nsteps) + '_emb' + str(args.embdim) + '_samples' + str(sample),
                             n_sample_nodes=sample
                             )

    elif args.testDataType == 'enron':
        print("datatype:", args.testDataType)

        files = [file for file in os.listdir('./test_data/enron') if 'month' in file]
        length = len(files)
        graphsall = []

        for i in range(length):
            G = nx.read_gpickle('./test_data/enron/month_' + str(i + 1) + '_graph.gpickle')
            graphsall.append(G)

        sample = graphsall[0].number_of_nodes()
        graphs = graphsall[-args.nsteps:]
        datafile = dataprep_util.prep_input_dynTriad(graphs, args.nsteps, args.testDataType)
        # pdb.set_trace()

        embedding = dynamicTriad(niters=args.niters,
                                 starttime=args.starttime,
                                 datafile=datafile,
                                 batchsize=100,
                                 nsteps=args.nsteps,
                                 embdim=args.embdim,
                                 stepsize=args.stepsize,
                                 stepstride=args.stepstride,
                                 outdir=args.outdir,
                                 cachefn=args.cachefn,
                                 lr=args.lr,
                                 beta=args.beta,
                                 negdup=args.negdup,
                                 datasetmod=args.datasetmod,
                                 trainmod=args.trainmod,
                                 pretrain_size=args.pretrain_size,
                                 sampling_args=args.sampling_args,
                                 validation=args.validation,
                                 datatype=args.testDataType,
                                 scale=args.scale,
                                 classifier=args.classifier,
                                 debug=args.debug,
                                 test=args.test,
                                 repeat=args.repeat,
                                 resultdir=args.resultdir,
                                 testDataType=args.testDataType,
                                 clname='lr',
                                 node_num=sample

                                 )

        embedding.learn_embedding()
        embedding.get_embedding()

        outdir = args.resultdir
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        outdir = outdir + '/' + args.testDataType
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        outdir = outdir + '/dynTriad'
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        lp.expstaticLP_TRIAD(None,
                             graphs,
                             embedding,
                             1,
                             outdir + '/',
                             'l' + str(args.nsteps) + '_emb' + str(args.embdim) + '_samples' + str(sample),
                             n_sample_nodes=sample
                             )
