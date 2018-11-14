# triagnle data format:
#   time step of the open triangle
#   open triangle ceter node
#   open triangle node 1
#   open triangle node 2
#   label/coefficient
#   weight 1
#   weight 2 

from __future__ import print_function
from __future__ import absolute_import

import sys
from . import pos_neg
from core.algorithm.embutils import WithData
import core.gconfig as gconf
from core import utils
from collections import defaultdict


class Sampler(pos_neg.Sampler, WithData):
    # almost the same as datagen_pos_neg.DataGen, except that triangular data are sampled with negative samples
    def __init__(self, **kwargs):
        super(Sampler, self).__init__(**kwargs)

        self.__enable_cache = kwargs.get('triangle_cache', False)
        self._triangular_cache = utils.OffsetList(self.dataset.localstep, self.dataset.nsteps, lambda x: None)
        # self._triangular_cache = [None for _ in range(len(self.dataset.gtgraphs))]
        # self._edge_cache = [None for _ in range(len(self.dataset.gtgraphs))]
        self.__nbr_cache = [{} for _ in range(self.dataset.nsteps)]
        self.__succ_trial, self.__all_trial = 10, 12

    # in this implementation, negative samples rely wholy on positive samples,
    # as a result, begin and end params are ignored in make_neg
    def pretrain_begin_iteration(self):
        super(Sampler, self).pretrain_begin_iteration()

        # sample triangular data
        filtered_pos = [p for p in self._pos[0] if p[0] + 1 < self._pos_range[1]]  # except the last time slice!

        if len(filtered_pos) <= 0:
            print("No possible triangular samples, given positive range {} to {}".
                  format(self._pos_range[0], self._pos_range[1]))
            triagdata = [None] * len(self._pos[0])  # in order to pass assertion in datagen_pos_neg.batches()
        else:
            if not self.__enable_cache:
                mapper = utils.ParMap(self.__uncached_sampler_factory(), self.__sample_uncached_monitor, njobs=gconf.njobs)
                triagdata = []
                sample_round = 0
                while len(triagdata) < len(self._pos[0]):
                    left_cnt = len(self._pos[0]) - len(triagdata)

                    # verboses
                    print("sample round {}, target #samples {}".format(sample_round, left_cnt))
                    sample_round += 1

                    # increase the probability of finish sampling in a single round
                    left_cnt = int(left_cnt * (float(self.__all_trial) / self.__succ_trial + 0.2))
                    if left_cnt < 100:
                        left_cnt = 100
                        mapper.njobs = 1
                    lb = max(0, utils.crandint(len(filtered_pos) - left_cnt))
                    ub = min(lb + left_cnt, len(filtered_pos))
                    newsamples = mapper.run(filtered_pos[lb:ub])
                    self.__all_trial += (ub - lb)
                    self.__succ_trial += len(newsamples)
                    triagdata.extend(newsamples)
                triagdata = triagdata[:len(self._pos[0])]
            else:
                raise NotImplementedError()

        self._neg.append(triagdata)  # neg, triangdata_int, triangdata_float

    def _triag_cache(self, k, knode, onode):
        # assert self._edge(k, knode, onode) is not None
        key = "{},{}".format(knode, onode)
        if self._triangular_cache[k] is None:
            self._make_triag_cache(k)
        try:
            return self._triangular_cache[k][key]
        except KeyError as e:
            print(self._triangular_cache[k])
            raise e

    def _make_triag_cache(self, t):
        print("making triag cache for {}".format(t))
        self._triangular_cache[t] = {}
        g = self.dataset.mygraphs[t]
        name2idx = {n: i for i, n in enumerate(self.dataset.gtgraphs[t].vp['name'])}
        # assume g is not a graphview here, because we rely on int(vertex)
        # adj = spect.adjacency(g).todense()
        # NOTE: all following computations are based on names instead of indices
        for vi in g.vertices():
            # nbr = list(vi.out_neighbours())
            nbr = list(g.out_neighbours(vi))
            for vj in nbr:
                i, j = name2idx[vi], name2idx[vj]
                key = "{},{}".format(i, j)
                curcache = self._triangular_cache[t][key] = []
                #curcache = []
                for vk in nbr:
                    k = name2idx[vk]
                    #assert adj[j, k] == adj[k, j]
                    if vk != vj and not g.exists(vj, vk):
                        curcache.append(k)
        print("end making triag cache for {}".format(t))

    # batch: (pos, weight, neg, triadint, triadfloat)
    def make_pretrain_input(self, batch):
        input = pos_neg.Sampler.make_pretrain_input(self, batch)
        return input + list(batch[3:])

    # for parallel uncached sampling
    @staticmethod
    def __sample_uncached_monitor(reportq):
        procinfo = {}
        proc_reent = defaultdict(lambda: 0)
        while True:
            obj = reportq.get()
            if isinstance(obj, StopIteration):
                break
            if obj[1] is None:  # a proc terminates
                procinfo["{}_{}".format(obj[0], proc_reent[obj[0]])] = procinfo[obj[0]]
                del procinfo[obj[0]]
                proc_reent[obj[0]] += 1
                continue

            procinfo[obj[0]] = obj[1:]
            
            total_proccnt = sum([v[0] for v in procinfo.values()])
            total_availcnt = sum([v[1] for v in procinfo.values()])
            total_trycnt = sum([v[2] for v in procinfo.values()])
            print("{} samples processed, {} succeeded with avg try cnt {}\r".
                  format(total_proccnt, total_availcnt, float(total_trycnt) / max(total_availcnt, 1)), end='')
            sys.stdout.flush()
        total_proccnt = sum([v[0] for v in procinfo.values()])
        total_availcnt = sum([v[1] for v in procinfo.values()])
        total_trycnt = sum([v[2] for v in procinfo.values()])
        print("{} samples processed, {} succeeded with avg try cnt {}".
              format(total_proccnt, total_availcnt, float(total_trycnt) / max(total_availcnt, 1)))

    def __uncached_sampler_factory(self):
        # this method is added to avoid sharing the whole self object between processes
        sample_one_uncached = self.__sample_one_uncached
        # dill seems not to work with cython objects, this workaround requires __sample_one_uncached to accept
        # localstep as an argument
        mygraphs = list(self.dataset.mygraphs)

        nodenames = list(self.dataset.gtgraphs['any'].vp['name'])
        # the order defined by vp should be the same as mygraphs.vertices(),
        # which is also the storing order of the embedding
        for v1, v2 in zip(nodenames, self.dataset.mygraphs['any'].vertices()):
            assert v1 == v2, (v1, v2, type(v1), type(v2))

        name2idx = {n: i for i, n in enumerate(nodenames)}

        localstep = self.dataset.localstep

        def __sample_uncached(process, data, reportq):
            total_trycnt = 0
            total_avail_cnt = 0
            ret = []
            for i, sample in enumerate(data):
                if i % 10000 == 0:
                    reportq.put([id(process), i, total_avail_cnt, total_trycnt])
                curres, trycnt = sample_one_uncached(sample, nodenames, name2idx, mygraphs, localstep)
                total_trycnt += trycnt
                if curres is not None:
                    total_avail_cnt += 1
                    ret.append(curres)
            reportq.put([id(process), len(data), total_avail_cnt, total_trycnt])
            reportq.put([id(process), None])  # signal for terminate
            return ret

        return __sample_uncached

    @staticmethod
    def __sample_one_uncached(data, nodenames, name2idx, mygraphs, localstep):
        k, src, tgt = [int(d) for d in data]  # convert from np types to int, to avoid problems in c extensions
        k=int(k)
        src=int(src)
        tgt=int(tgt)
        localstep=int(localstep)
        myg = mygraphs[k - localstep]
        mynextg = mygraphs[k + 1 - localstep]

        if utils.crandint(2) == 0:  # target as key point
            trycnt = 0
            # new_src = random.randint(0, self.dataset.graphs[k].num_vertices() - 1)
            nbr = myg.out_neighbours(nodenames[tgt])
            new_src = name2idx[nbr[utils.crandint(len(nbr))]]
            # while self._edge(k, tgt, new_src) is None or self._edge(k, src, new_src) is not None:
            while new_src == tgt or new_src == src or not myg.exists(nodenames[tgt], nodenames[new_src]) or \
                    myg.exists(nodenames[src], nodenames[new_src]):
                if trycnt >= 5:
                    break
                # new_src = random.randint(0, self.dataset.graphs[k].num_vertices() - 1)
                new_src = name2idx[nbr[utils.crandint(len(nbr))]]
                trycnt += 1
            if trycnt >= 5:
                # nbr = [int(v) for v in self.dataset.gtgraphs[k].vertex(tgt).out_neighbours()
                #       if int(v) != src and int(v) != tgt and not myg.exists(nodenames[int(v)], nodenames[src])]
                #       if int(v) != src and self._edge(k, v, src) is None]
                cand = [name2idx[n] for n in nbr]
                cand = [n for n in cand if n != src and n != tgt and
                        not myg.exists(nodenames[n], nodenames[src])]
                if len(cand) <= 0:
                    return None, trycnt
                # new_src = nbr[random.randint(0, len(nbr) - 1)]
                new_src = cand[utils.crandint(len(cand))]
            # triagdata.append([k, tgt, src, new_src, self._edge(k + 1, src, new_src) is not None,
            #                  w[self._edge(k, tgt, src)], w[self._edge(k, tgt, new_src)]])
            ret = [k, tgt, src, new_src, mynextg.exists(nodenames[src], nodenames[new_src]),
                   myg.edge(nodenames[tgt], nodenames[src]),
                   myg.edge(nodenames[tgt], nodenames[new_src])]
        else:  # src as key point
            trycnt = 0
            nbr = myg.out_neighbours(nodenames[src])
            # new_tgt = random.randint(0, self.dataset.graphs[k].num_vertices() - 1)
            new_tgt = name2idx[nbr[utils.crandint(len(nbr))]]
            # while self._edge(k, src, new_tgt) is None or self._edge(k, tgt, new_tgt) is not None:
            while new_tgt == src or new_tgt == tgt or not myg.exists(nodenames[src], nodenames[new_tgt]) or \
                    myg.exists(nodenames[tgt], nodenames[new_tgt]):
                if trycnt >= 5:
                    break
                # new_tgt = random.randint(0, self.dataset.graphs[k].num_vertices() - 1)
                new_tgt = name2idx[nbr[utils.crandint(len(nbr))]]
                trycnt += 1
            if trycnt >= 5:
                # nbr = [int(v) for v in self.dataset.gtgraphs[k].vertex(src).out_neighbours()
                #       if int(v) != tgt and int(v) != src and not myg.exists(nodenames[int(v)], nodenames[tgt])]
                #       if int(v) != tgt and self._edge(k, v, tgt) is None]
                cand = [name2idx[n] for n in nbr]
                cand = [n for n in cand if n != tgt and n != src and
                        not myg.exists(nodenames[n], nodenames[tgt])]
                if len(cand) <= 0:
                    return None, trycnt
                # new_tgt = nbr[random.randint(0, len(nbr) - 1)]
                new_tgt = cand[utils.crandint(len(cand))]
            # triagdata.append([k, src, tgt, new_tgt, self._edge(k + 1, tgt, new_tgt) is not None,
            #                  w[self._edge(k, src, tgt)], w[self._edge(k, src, new_tgt)]])
            ret = [k, src, tgt, new_tgt, mynextg.exists(nodenames[tgt], nodenames[new_tgt]),
                   myg.edge(nodenames[src], nodenames[tgt]),
                   myg.edge(nodenames[src], nodenames[new_tgt])]

        assert len(set(ret[1:4])) == 3 and ret[5] > 0 and ret[5] > 0, ret
        return ret, trycnt

    def __debug_and_count_triangles(self, nodenames):
        # for debug, count all possible triangles
        for i in range(self.dataset.localstep, self.dataset.localstep + self.dataset.nsteps - 1):
            # adji = spect.adjacency(self.dataset.graphs[i]).todense()
            # adji1 = spect.adjacency(self.dataset.graphs[i + 1]).todense()
            triagcnt = 0
            edgeset = set()
            # same stat as above two, except that the missing edge exists in next graph
            postriagcnt = 0
            posedgeset = set()
            for e in self.dataset.gtgraphs[i].edges():
                isrc, itgt = int(e.source()), int(e.target())
                for v in self._triag_cache(i, isrc, itgt):
                    # assert adji[itgt, v] == 0
                    triagcnt += 1
                    v1, v2 = min(itgt, v), max(itgt, v)
                    edgeset.add((v1, v2))
                    # if adji1[itgt, v] != 0:
                    if self.dataset.mygraphs[i + 1].exists(nodenames[v1], nodenames[v2]):
                        postriagcnt += 1
                        posedgeset.add((v1, v2))
                for v in self._triag_cache(i, itgt, isrc):
                    # assert adji[isrc, v] == 0
                    triagcnt += 1
                    v1, v2 = min(isrc, v), max(isrc, v)
                    edgeset.add((v1, v2))
                    # if adji1[isrc, v] != 0:
                    if self.dataset.mygraphs[i + 1].exists(nodenames[v1], nodenames[v2]):
                        postriagcnt += 1
                        posedgeset.add((v1, v2))
            assert triagcnt % 2 == 0 and postriagcnt % 2 == 0
            triagcnt /= 2
            postriagcnt /= 2
            print("for {}-th graph".format(i))
            print("{} edges forms {} open triangles".format(len(edgeset), triagcnt))
            print("{} edges appear in next graph, forming {} open triangles".format(len(posedgeset), postriagcnt))
