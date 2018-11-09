from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import random
from . import sampler
from core import gconfig as gconf
from core import utils
from core.algorithm.embutils import WithData


class Sampler(sampler.Sampler, WithData):
    def __init__(self, **kwargs):
        self._pos = None
        self._pos_range = [-1, 1]
        self._neg = None
        self._valid = None
        self._negdup = kwargs.get('negdup', 1)

        self.__enable_cache = kwargs.get('replace_cache', False)
        self._replace_cache = utils.OffsetList(self.dataset.gtgraphs.offset, len(self.dataset.gtgraphs), lambda x: None)

    def __make_pos(self, begin, end):
        data = []
        weight = []
        # TODO: remove this ugly fix
        nodenames = list(self.dataset.gtgraphs['any'].vp['name'])
        for i in range(begin, end):
            assert not self.dataset.gtgraphs[i].is_directed()
            for e in self.dataset.gtgraphs[i].edges():
                src, tgt = int(e.source()), int(e.target())
                if src > tgt:
                    src, tgt = tgt, src
                nsrc, ntgt = nodenames[src], nodenames[tgt]
                # for debug
                if gconf.debug:  # because .edge is slow
                    assert self.dataset.mygraphs[i].exists(nsrc, ntgt), \
                        "{}: {} {}".format(i, nsrc, ntgt)

                data.append([i, src, tgt])
                weight.append(self.dataset.mygraphs[i].edge(nsrc, ntgt))
        data = np.array(data, dtype='int32')
        weight = np.array(weight, dtype='float32')
        if len(data) == 0:
            raise RuntimeError("No positive sample is generated given an empty graph")
        assert len(data) == sum([g.num_edges() for g in self.dataset.gtgraphs[begin:end]]), \
            "{}, {}".format(len(data), sum([g.num_edges() for g in self.dataset.gtgraphs[begin:end]]))
        return [data, weight]

    def pretrain_begin(self, begin, end):
        self._pos_range = [begin, end]

        self._pos = self.__make_pos(begin, end)

    def pretrain_end(self):
        pass

    def __make_neg(self, posdata, negdup=1):
        negdata = []
        # TODO: this is an ugly fix, try to add indexing support in mygraph
        nodenames = list(self.dataset.gtgraphs['any'].vp['name'])

        for d in posdata:
            k, src, tgt = d
            negdata.append([])
            for i in range(negdup):
                if utils.crandint(2) == 0:  # replace source
                    if self.__enable_cache:
                        curcache = self._rep_cache(k)[tgt]
                        new_src = curcache[utils.crandint(len(curcache))]
                        negdata[-1].extend([new_src, tgt])
                    else:
                        # TODO: although it is almost impossible for a node to have all edges, check this in advance
                        #new_src = random.randint(0, self.dataset.gtgraphs[k].num_vertices() - 1)
                        new_src = utils.crandint(self.dataset.gtgraphs[k].num_vertices())
                        assert not self.dataset.gtgraphs[k].is_directed()
                        while self.dataset.mygraphs[k].exists(nodenames[new_src], nodenames[tgt]):
                            #new_src = random.randint(0, self.dataset.gtgraphs[k].num_vertices() - 1)
                            new_src = utils.crandint(self.dataset.gtgraphs[k].num_vertices())
                        negdata[-1].extend([new_src, tgt])
                else:  # replace target
                    if self.__enable_cache:
                        curcache = self._rep_cache(k)[src]
                        #new_tgt = curcache[random.randint(0, len(curcache) - 1)]
                        new_tgt = curcache[utils.crandint(len(curcache))]
                        negdata[-1].extend([src, new_tgt])
                    else:
                        #new_tgt = random.randint(0, self.dataset.gtgraphs[k].num_vertices() - 1)
                        new_tgt = utils.crandint(self.dataset.gtgraphs[k].num_vertices())
                        while self.dataset.mygraphs[k].exists(nodenames[src], nodenames[new_tgt]):
                            #new_tgt = random.randint(0, self.dataset.gtgraphs[k].num_vertices() - 1)
                            new_tgt = utils.crandint(self.dataset.gtgraphs[k].num_vertices())
                        negdata[-1].extend([src, new_tgt])
        negdata = np.array(negdata)
        assert negdata.shape == (len(posdata), 2 * negdup), "{}, {}".format(negdata.shape, (len(posdata), 2 * negdup))

        return negdata

    # in this implementation, negative samples rely wholy on positive samples,
    # as a result, begin and end params are ignored in make_neg
    def pretrain_begin_iteration(self):
        # TODO: modify __make_neg to return a list
        self._neg = [self.__make_neg(self._pos[0], negdup=self._negdup)]

    def pretrain_end_iteration(self):
        pass

    def online_begin(self, begin, end):
        assert begin == end - 1
        self._pos = self.__make_pos(begin, end)
        self._pos_range = [begin, begin]

    def online_end(self):
        pass

    def online_begin_iteration(self):
        self.shuffle_sample()
        self._neg = [self.__make_neg(self._pos[0], negdup=self._negdup)]

    def online_end_iteration(self):
        pass

    def make_pretrain_input(self, batch):
        pos, weight, neg = batch[:3]
        assert neg.shape[1] % 2 == 0
        dupneg = neg.shape[1] / 2
        data = []
        # TODO: by doing so, we always train samples from the same edge together, does this matter?
        for i in range(len(pos)):
            for j in range(0, len(neg[i]), 2):
                data.append(list(pos[i]) + [neg[i][j], neg[i][j + 1]])
        data = np.array(data)
        assert data.shape == (len(pos) * dupneg, 5)
        return [data, weight]

    def make_online_input(self, batch):
        # polymorphism is not expected here, since make_pretrain_input and make_online_input are conceptually different
        # we call Sampler.make_pretrain_input here simply to avoid copying and pasting
        return Sampler.make_pretrain_input(self, batch)

    # def make_valid_input(self):
    #     return [self._valid[0]]
    #
    # def make_valid_labels(self):
    #     return self._valid[1]

    def _make_rep_cache(self, k):
        self._replace_cache[k] = []
        g = self.dataset.gtgraphs[k]
        all_nodes = set(range(g.num_vertices()))
        for j in range(g.num_vertices()):
            self._replace_cache[k].append(list(all_nodes - set([int(v) for v in g.vertex(j).out_neighbours()])))

    def _rep_cache(self, k):
        if self._replace_cache[k] is None:
            self._make_rep_cache(k)
        return self._replace_cache[k]

    def shuffle_sample(self):
        self._pos[0], order, invorder = utils.shuffle_sample(self._pos[0], return_order=True)
        for i in range(1, len(self._pos)):
            self._pos[i] = utils.apply_order(self._pos[i], order)
        # in this model, negative samples are drawn after shuffling positive samples

    def batches(self, batchsize):
        for i, s in enumerate(self._pos + self._neg):
            assert len(s) == self.sample_size(), "{}-th: {} {}".format(i, len(s), self.sample_size())
        isamp = [utils.islice_sample(s, chunk=batchsize) for s in self._pos + self._neg]
        for s in list(zip(*isamp)):
            yield s

    def sample_size(self):
        return len(self._pos[0])
