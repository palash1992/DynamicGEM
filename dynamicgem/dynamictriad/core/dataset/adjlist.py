from __future__ import print_function

from dynamicgem.dynamictriad.core.dataset.dataset_utils import DatasetBase
from .. import mygraph
from .. import mygraph_utils as mgutils


class Dataset(DatasetBase):
    @property
    def inittime(self):
        return 0 

    def __init__(self, datafn, localtime, nsteps, stepsize, stepstride, offset=0, dataname=None):
        self.datafn = datafn
        self.__datadir = datafn 

        DatasetBase.__init__(self, datafn, localtime, nsteps, stepsize, stepstride, offset, dataname)

        self.__vertices = None

    @property
    def name(self):
        return "adjlist"

    # required by Timeline
    def _time2unit(self, tm):
        return int(float(tm))

    def _unit2time(self, unit):
        return str(unit)

    def __check_vertices(self, vs):
        assert len(vs) == len(self.__vertices), (len(vs), len(self.__vertices))
        for i in range(len(vs)):
            assert vs[i] == self.__vertices[i], (i, vs[i], self.__vertices[i])

    # required by DyanmicGraph
    def _load_unit_graph(self, tm):
        tm = self._time2unit(tm)
        fn = "{}/{}".format(self.__datadir, tm)
        g = mgutils.load_adjlist(fn)
        if self.__vertices is None:
            self.__vertices = g.vertices()
        else:
            try:
                self.__check_vertices(g.vertices())  # ensure all graphs share a same set of vertices
            except AssertionError as e:
                raise RuntimeError("Vertices in graph file {} are not compatible with files already loaded: {}"
                                   .format(fn, e.message))
        return g

    def _merge_unit_graphs(self, graphs, curstep):
        curunit = self._time2unit(self.step2time(curstep))
        print("merging graph from year {} to {}".format(curunit, curunit + self.stepsize - 1))

        ret = mygraph.Graph(graphs[0].node_type(), graphs[0].weight_type())
        for g in graphs:
            ret.merge(g, free_other=False)

        return ret

    # required by Archivable(Archive and Cache)
    # def _full_archive(self, name=None):
    #     return self.archive(name)

    def archive(self, name=None):
        ar = super(Dataset, self).archive()
        return ar

    def load_archive(self, ar, copy=False, name=None):
        super(Dataset, self).load_archive(ar, copy=copy)


