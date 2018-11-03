from __future__ import print_function
from __future__ import absolute_import

from dynamicgem.dynamictriad.core import utils, gconv
from dynamicgem.dynamictriad.core import gconfig as gconf
import random
import numpy as np
from collections import defaultdict
from six.moves import range
import os


class Timeline(utils.Archivable):
    def __init__(self, inittime, stepsize, stepstride):
        self.initunit = self._time2unit(inittime)
        self.stepsize = stepsize
        self.stepstride = stepstride

    def time2step(self, tm):
        tmunit = self._time2unit(tm) - self.initunit
        if tmunit < 0 or tmunit % self.stepstride != 0:
            raise RuntimeError("Invalid step time {}({}), with len={}, stride={}"
                               .format(tmunit, tm, self.stepsize, self.stepstride))
        return tmunit / self.stepstride

    def step2time(self, step):
        tmunit = step * self.stepstride
        return self._unit2time(tmunit + self.initunit)

    def _step2unit(self, step):
        return self._time2unit(self.step2time(step))

    def _unit2step(self, unit):
        return self.time2step(self._unit2time(unit))

    def _time2unit(self, step):
        raise NotImplementedError()

    def _unit2time(self, unit):
        raise NotImplementedError()

    # note that archive is used in combination with init args


# Archive provides tool to build Dataset from an archive
class Archive(utils.Archivable):
    def _archive_args(self):
        return []

    def archive(self, name=None):
        if name is None:
            prefix = 'Archive'
        else:
            prefix = '{}_Archive'.format(name)

        ar = super(Archive, self).archive(name)
        ar['{}_args'.format(prefix)] = self._archive_args()
        return ar

    @classmethod
    def from_archive(cls, ar, copy=False, name=None):
        if name is None:
            prefix = 'Archive'
        else:
            prefix = '{}_Archive'.format(name)

        obj = cls(*ar.get('{}_args'.format(prefix)))
        obj.load_archive(ar, copy=copy)
        return obj


# Cache provides tool to load cached data given an already built Dataset
class Cache(utils.Archivable):
    def _cache_args(self):
        return {}

    def is_compatible(self, args):
        return self._cache_args() == args
    
    # evaluate all lazy operations before archiving
    def _full_archive(self, name=None):
        return self.archive(name)

    def cache(self):
        return self._full_archive(), self._cache_args()

    def load_cache(self, args, datagen):
        if self.is_compatible(args):
            self.load_archive(datagen(), copy=False)
        else:
            raise ValueError("Incompatible args {} vs. {}".format(self._cache_args(), args))


class DynamicGraph(Timeline, utils.Archivable):
    def __init__(self, inittime, localtime, nsteps, stepsize, stepstride, stepoffset):
        # Timeline anchors the starting time while DynamicGraph bounds the ending
        Timeline.__init__(self, inittime, stepsize, stepstride)
        self.localstep = self.time2step(localtime) + stepoffset
        self.localunit = self._step2unit(self.localstep)
        if self._step2unit(self.localstep) < self._time2unit(inittime):
            raise RuntimeError("localstep smaller than initial step, with inittime={}, localtime={},"
                               " step_stride={}, step_offset={}"
                               .format(inittime, localtime, stepstride, stepoffset))
        self.nsteps = nsteps
        self.nunits = self._step2unit(self.localstep + self.nsteps - 1) - self.localunit + self.stepsize
        # we use mygraph as main data, with supportive gtgraphs available
        self._mygraphs = utils.OffsetList(self.localstep, self.nsteps, lambda step: self._load_graph(step))
        self._gtgraphs = utils.OffsetList(self.localstep, self.nsteps,
                                          lambda i: gconv.mygraph2graphtool(self._mygraphs[i], convert_to='undirected'))

    @property
    def mygraphs(self):
        return self._mygraphs

    @property
    def gtgraphs(self):
        return self._gtgraphs

    @property
    def nsize(self):
        return self.gtgraphs['any'].num_vertices()

    # override this to apply acceleration techniques
    def _load_graph(self, step):
        curunit = self._time2unit(self.step2time(step))
        graphs = []
        for u in range(curunit, curunit + self.stepsize):
            graphs.append(self._load_unit_graph(self._unit2time(u)))
        return self._merge_unit_graphs(utils.OffsetList(0, self.stepsize, graphs, copy=False), step)

    def _load_unit_graph(self, tm):
        raise NotImplementedError()

    def _merge_unit_graphs(self, graphs, curstep):
        raise NotImplementedError()

    # required by Archivable
    def archive(self, name=None):
        if name is None:
            prefix = 'DynamicGraph'
        else:
            prefix = '{}_DynamicGraph'.format(name)

        ar = super(DynamicGraph, self).archive(name)
        # note that conversion from/to graph_tool is even slower than reading from file
        ar['{}_gtgraphs'.format(prefix)] = self._gtgraphs.archive()
        ar['{}_mygraphs'.format(prefix)] = self._mygraphs.archive()
        return ar

    def load_archive(self, ar, copy=False, name=None):
        if name is None:
            prefix = 'DynamicGraph'
        else:
            prefix = '{}_DynamicGraph'.format(name)

        super(DynamicGraph, self).load_archive(ar, copy=copy, name=name)
        self._gtgraphs.load_archive(ar['{}_gtgraphs'.format(prefix)], copy=copy)
        self._mygraphs.load_archive(ar['{}_mygraphs'.format(prefix)], copy=copy)


class TestSampler(object):
    # protocols
    def vertex_labels(self):
        raise NotImplementedError("vertex_labels")

    def vertex_raw_labels(self):
        raise NotImplementedError("vertex_raw_labels")

    def vertex_static_labels(self):
        raise NotImplementedError("vertex_static_labels")
    
    # helpers
    @staticmethod
    def __stratified_sample_size(size, possize, negsize):
        posrat = float(possize) / (possize + negsize)
        newpossize = int(size * posrat)
        newnegsize = size - newpossize
        return newpossize, newnegsize
    
    # main function and implementation 
    tasks = 'link_reconstruction', 'link_prediction', 'node_classify', 'node_predict', \
            'changed_link_classify', 'changed_link_prediction', 'order_links', 'none'

    @property
    def __task_handler(self):
        ret = defaultdict(lambda: self.__unknown)
        ret.update({'link_reconstruction': self._sample_link_reconstruction,
                    'link_prediction': self._sample_link_prediction,
                    'node_classify': self._sample_node_classify,
                    'node_predict': self._sample_node_predict,
                    'changed_link_classify': self._sample_changed_link_classify,
                    'changed_link_prediction': self._sample_changed_link_prediction,
                    'order_links': self._sample_order_links,
                    'none': self.__none})
        return ret

    def _sample_order_links(self, begin, end, size=None, intv=0, name=""):
        gtgraphs = self.gtgraphs[begin + intv:end]

        def nodeitr():
            for i in range(len(gtgraphs)):
                for j in range(gtgraphs[i].num_vertices()):
                    if gtgraphs[i].vertex(j).out_degree() > 0:
                        yield (i + begin, j)

        if size is None:
            samples = list(nodeitr())
        else:
            samples = random.sample(utils.ilen(nodeitr(), gtgraphs[0].num_vertices() * len(gtgraphs)), size)

        lbs = []
        for v in samples:
            # v[0] - begin is correct given intv, because v[0] is 'i + begin', whose label shall be found in
            # 'i + begin + intv'-th graph, which is exactly gtgraphs[i + begin + intv - (begin + intv)], i.e.
            # gtgraphs[v[0] - begin]
            lb = np.zeros((gtgraphs[v[0] - begin].num_vertices(),), dtype='int8')
            lb[[int(n) for n in gtgraphs[v[0] - begin].vertex(v[1]).out_neighbours()]] = 1
            assert np.sum(lb) == gtgraphs[v[0] - begin].vertex(v[1]).out_degree()
            lbs.append(lb)
        assert len(lbs) == len(samples), "{} {}".format(len(lbs), len(samples))
        return [samples, lbs]

    def _sample_link_reconstruction(self, begin, end, size=None, negdup=1, intv=0, name=""):
        pos = []
        for i, g in enumerate(self.gtgraphs[begin + intv:end]):
            for e in g.edges():
                assert not g.is_directed()
                if gconf.debug and int(e.source()) > int(e.target()):
                    # check symmetric
                    names = g.vertex_properties['name']
                    assert g.edge(e.target(), e.source()), "{}: {} {}".format(i + begin, names[e.source()],
                                                                              names[e.target()])
                    assert g.edge_properties['weight'][e] == g.edge_properties['weight'][
                        g.edge(e.target(), e.source())]
                    continue
                pos.append([i + begin, int(e.source()), int(e.target())])
        pos = np.vstack(pos).astype('int32')

        neg = []
        vsize = self.gtgraphs['any'].num_vertices()
        nodenames = list(self.gtgraphs['any'].vp['name'])
        for i in range(negdup):
            for p in pos:
                tm, src, tgt = p
                g = self.mygraphs[tm + intv]
                assert g.out_degree(nodenames[src]) < vsize - 1 or g.out_degree(nodenames[tgt]) < vsize - 1, \
                    "We do not expect any node to connect to all other nodes"

                while True:
                    if random.randint(0, 1) == 0:  # replace source
                        # cur_range = negrange[tm][tgt]
                        # new_src = cur_range[random.randint(0, len(cur_range) - 1)]
                        new_src = random.randint(0, vsize - 1)
                        if not g.exists(nodenames[new_src], nodenames[tgt]):
                            neg.append([tm, new_src, tgt])
                            break
                    else:  # replace target
                        # cur_range = negrange[tm][src]
                        # new_tgt = cur_range[random.randint(0, len(cur_range) - 1)]
                        new_tgt = random.randint(0, vsize - 1)
                        if not g.exists(nodenames[src], nodenames[new_tgt]):
                            neg.append([tm, src, new_tgt])
                            break
        neg = np.vstack(neg).astype('int32')

        lbs = np.concatenate((np.ones(len(pos)), -np.ones(len(neg))))
        return np.concatenate((pos, neg), axis=0), lbs

    def _sample_link_prediction(self, begin, end, size=None, name=""):
        return self._sample_link_reconstruction(begin, end, size, intv=1)

    # intv is used for predition from previous time steps
    def _sample_node_classify(self, begin, end, size=None, intv=0, name=""):
        lbs = np.array(self.vertex_labels()[begin + intv:end], copy=False)

        possamp = np.transpose(np.vstack(np.nonzero(lbs == 1)))
        negsamp = np.transpose(np.vstack(np.nonzero(lbs == -1)))
    
        if size is not None:
            possize, negsize = self.__stratified_sample_size(size, len(possamp), len(negsamp))
            if possize < len(possamp):
                possamp = random.sample(possamp, possize)
            if negsize < len(negsamp):
                negsamp = random.sample(negsamp, negsize)
        
        if len(possamp) == 0:
            raise RuntimeError("Not enough positive samples for training")
        samples = np.concatenate((possamp, negsamp), axis=0)
        samples[:, 0] += begin  # from begin-based time to 0-based time
        lbs = np.concatenate((np.ones(len(possamp)), -np.ones(len(negsamp))), axis=0)
        return [samples, lbs]
    
    def _sample_node_predict(self, begin, end, size=None, name=""):
        return self._sample_node_classify(begin, end, size, intv=1)

    def _sample_changed_link_classify(self, begin, end, size=None, intv=0, name=""):
        if end - begin < 2:
            raise RuntimeError("there must be at least 2 graphs in 'changed' sample method")

        samp, lbs = [], []
        for i in range(begin, end - 1 - intv):
            prevg, curg = self.gtgraphs[i + intv:i + 2 + intv]

            def edge_set(g):
                ret = set()
                for e in g.edges():
                    s, t = int(e.source()), int(e.target())
                    if s > t:
                        s, t = t, s
                    ret.add((s, t))
                return ret

            cure = edge_set(curg)
            preve = edge_set(prevg)

            for s, t in cure - preve:
                # i + 1 because i enumerates all prev graphs
                samp.append([i + 1, s, t])
                lbs.append(1)
            for s, t in preve - cure:
                samp.append([i + 1, s, t])
                lbs.append(-1)

            if gconf.debug:
                # only check in debug mode because it is time consuming to call g.edge
                for i in range(len(samp)):
                    if lbs[i] == 1:
                        assert self.gtgraphs[samp[i][0]].edge(samp[i][1], samp[i][2]) is not None
                    else:
                        assert self.gtgraphs[samp[i][0]].edge(samp[i][1], samp[i][2]) is None

        samp = np.array(samp)
        lbs = np.array(lbs)

        if size is not None:
            sampidx = random.sample(range(len(samp)), size)
            samp = samp[sampidx]
            lbs = lbs[sampidx]

        return samp, lbs

    def _sample_changed_link_prediction(self, begin, end, size, name=""):
        return self._sample_changed_link_classify(begin, end, size, intv=1)

    def __none(self, begin, end, size=None):
        return [[], []]

    def __unknown(self, begin, end, size=None, name=""):
        raise NotImplementedError("Unknown sampling task {}".format(name))

    def sample_test_data(self, task, begin, end, size=None):
        return self.__task_handler[task](begin, end, size, name=task)


class DatasetBase(DynamicGraph, Archive, Cache, TestSampler):
    initarg_names = ['datafn', 'localtime', 'nsteps', 'stepsize', 'stepstride', 'offset', 'dataname']

    @property
    def inittime(self):
        raise NotImplementedError()

    def _archive_args(self):
        return super(DatasetBase, self)._archive_args() + self.initargs

    def _cache_args(self):
        cargs = super(DatasetBase, self)._cache_args()
        cargs.update({n: a for a, n in zip(self.initargs, self.initarg_names)})
        if cargs['dataname'] is not None:
            del cargs['datafn']  # use name instead of data file name
        else:
            del cargs['dataname']
            cargs['datafn'] = os.path.abspath(cargs['datafn'])  # use absolute path for cache args
        return cargs

    def __init__(self, datafn, localtime, nsteps, stepsize=5, stepstride=1, offset=0, dataname=None):
        DynamicGraph.__init__(self, self.inittime, localtime, nsteps, stepsize, stepstride, offset)
        self.initargs = [datafn, localtime, nsteps, stepsize, stepstride, offset, dataname]

