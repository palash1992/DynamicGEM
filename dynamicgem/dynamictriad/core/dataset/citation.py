from __future__ import print_function

# import graphtool_utils as gtutils
import numpy as np
import re
from six.moves import cPickle, reduce
from collections import Counter
from dataset_utils import DatasetBase
import core.gconfig as gconf
from core import utils, mygraph


class Dataset(DatasetBase):
    @property
    def inittime(self):
        return self.__data['args']['minyear']

    def __init__(self, datafn, localyear=None, nsteps=None, stepsize=None, stepstride=None, offset=0, dataname=None):
        self.datafn = datafn
        self.__data = cPickle.load(open(self.datafn, 'r'))
        
        nonecnt = sum([int(v is None) for v in (localyear, nsteps, stepsize, stepstride)])
        if nonecnt == 4:
            # use information from the data
            localyear = self.__data['args']['minyear']
            nsteps = self.__data['args']['maxyear'] - self.__data['args']['minyear'] + 1
            stepsize = 1
            stepstride = 1
        elif nonecnt != 0:
            raise RuntimeError("You should not specify a part of dataset arguments")

        DatasetBase.__init__(self, datafn, localyear, nsteps, stepsize, stepstride, offset, dataname)

        self.__vertex_raw_labels_cache = None

    @property
    def name(self):
        return "citation"

    # required by Timeline
    def _time2unit(self, tm):
        return int(tm)

    def _unit2time(self, unit):
        return str(unit)

    # required by DyanmicGraph
    def _load_unit_graph(self, tm):
        year = self._time2unit(tm)
        # return gtutils.load_graph(self.__data['graphs'][year],
        #                           fmt='mygraph', convert_to='undirected')
        return self.__data['graphs'][year]

    def _merge_unit_graphs(self, graphs, curstep):
        curunit = self._time2unit(self.step2time(curstep))
        print("merging graph from year {} to {}".format(curunit, curunit + self.stepsize - 1))

        ret = mygraph.Graph(graphs[0].node_type(), graphs[0].weight_type())
        for g in graphs:
            ret.merge(g, free_other=False)

        return ret

    # required by Archivable(Archive and Cache)
    def _full_archive(self, name=None):
        self.__vertex_raw_labels()  # evaluate lazy operations
        return self.archive(name)

    def archive(self, name=None):
        if name is None:
            prefix = 'Dataset'
        else:
            prefix = '{}_Dataset'.format(name)

        ar = super(Dataset, self).archive()
        ar['{}_cache'.format(prefix)] = [self.__vertex_raw_labels_cache]
        return ar

    def load_archive(self, ar, copy=False, name=None):
        if name is None:
            prefix = 'Dataset'
        else:
            prefix = '{}_Dataset'.format(name)

        super(Dataset, self).load_archive(ar, copy=copy)
        self.__vertex_raw_labels_cache, = ar['{}_cache'.format(prefix)]
        if copy:
            self.__vertex_raw_labels_cache = self.__vertex_raw_labels_cache.copy()

    @property
    def manual_features(self):
        raise NotImplementedError()

    @property
    def data(self):
        return self.__data

    @staticmethod
    def __label_vertices(feats, featnames, confdata):
        labels = []
        for f in feats:
            cur = [0] * len(confdata)
            for idx in np.nonzero(f)[0]:
                curconfidx = None
                for k, v in confdata.items():
                    if re.match(v[1], featnames[idx]) or re.match(v[2], featnames[idx]):
                        if curconfidx is None:
                            curconfidx = v[0]
                        else:
                            print("[Warning]: {} satisfies both patterns {} and {}".format(featnames[idx], curconfidx,
                                                                                           v[0]))
                if curconfidx is not None:
                    cur[curconfidx] += f[idx]
            if np.max(cur) <= 0:
                labels.append(-1)
            else:
                labels.append(np.argmax(cur))
        print("label distribution: {}".format(Counter(labels)))
        return np.array(labels)

    def __vertex_raw_labels(self, return_name=False):
        raw_names = {-1: 'Unknown', 0: 'Architecture', 1: 'Computer Network', 2: 'Computer Security',
                     3: 'Data Mining', 4: 'Theory', 5: 'Graphics'}

        if self.__vertex_raw_labels_cache is not None:
            if return_name:
                return self.__vertex_raw_labels_cache, raw_names
            else:
                return self.__vertex_raw_labels_cache

        # These are conferences that are to merged
        confdata = [['ASPLOS|Architectural Support for Programming Languages and Operating Systems',
                     'FAST|Conference on File and Storage Technologies',
                     'HPCA|High-Performance Computer Architecture',
                     'ISCA|Symposium on Computer Architecture',
                     'MICRO|MICRO',
                     'USENIX ATC|USENIX Annul Technical Conference',
                     'PPoPP|Principles and Practice of Parallel Programming'],
                    ['MOBICOM|Mobile Computing and Networking Transactions on Networking',
                     'SIGCOMM|applications, technologies, architectures, and protocols for computer communication',
                     'INFOCOM|Computer Communications'],
                    ['CCS|Computer and Communications Security',
                     'NDSS|Network and Distributed System Security',
                     # 'CRYPTO|International Cryptology Conference',
                     # 'EUROCRYPT|European Cryptology Conference',
                     'S\&P|Symposium on Security and Privacy',
                     'USENIX Security|Usenix Security Symposium'],
                    ['SIGMOD|Conference on Management of Data',
                     'SIGKDD|Knowledge Discovery and Data Mining',
                     'SIGIR|Research on Development in Information Retrieval',
                     'VLDB|Very Large Data Bases',
                     'ICDE|Data Engineering'],
                    ['STOC|ACM Symposium on Theory of Computing',
                     'FOCS|Symposium on Foundations of Computer Science',
                     'LICS|Symposium on Logic in Computer Science',
                     'CAV|Computer Aided Verification'],
                    [  # 'ACM MM|Multimedia',
                        'SIGGRAPH|SIGGRAPH Annual Conference',
                        'IEEE VIS|Visualization Conference',
                        'VR|Virtual Reality'],
                    # ['AAAI|AAAI Conference on Artificial Intelligence',
                    #  'CVPR|Computer Vision and Pattern Recognition',
                    #  'ICCV|International Conference on Computer Vision',
                    #  'ICML|International Conference on Machine Learning',
                    #  'IJCAI|International Joint Conference on Artificial Intelligence',
                    #  'NIPS|Annual Conference on Neural Information Processing Systems',
                    #  'ACL|Annual Meeting of the Association for Computational Linguistics']
                    ]
        # confdata records the representing conferences for each field
        confdata = {n: i for i, arr in enumerate(confdata) for n in arr}
        for k in confdata.keys():
            sname, lname = k.split('|')
            confdata[k] = [confdata[k], re.compile(sname), re.compile(lname, re.I)]
        # conffeat is a list of matrices, each of them is a user-conference matrix for participation information
        conffeat = [self.__data['conf_feat'][y] for y in range(self.localunit, self.localunit + self.nunits)]

        # names of conferences in the user-conference matrix, in global indices (before filtering)
        conffeat_names = self.__data['conf_names']
        confmap = self.__data['confmap']
        # maps conference names from global indices to their original names
        conffeat_names = [confmap[c] for c in conffeat_names]

        # we use theory conferences because it is more independent
        rawlb = []
        for i in range(self.nsteps):
            startunit = self._time2unit(self.step2time(i + self.localstep))
            endunit = startunit + self.stepsize
            relstartunit, relendunit = startunit - self.localunit, endunit - self.localunit
            print("generating samples for years from {} to {}, i.e. featidx from {} to {}".
                  format(startunit, endunit - 1, relstartunit, relendunit - 1))
            curconffeat = reduce(lambda x, y: x + y, conffeat[relstartunit:relendunit],
                                 np.zeros(conffeat[relstartunit].shape, dtype=conffeat[relstartunit].dtype)).A
            rawlb.append(self.__label_vertices(curconffeat, conffeat_names, confdata))

            print("{}/{} positive samples at step {}".format(np.sum(rawlb[-1] == 1), len(rawlb[-1]), i + self.localstep))
        rawlb = np.vstack(rawlb)
        self.__vertex_raw_labels_cache = rawlb
        if return_name:
            return self.__vertex_raw_labels_cache, raw_names
        else:
            return self.__vertex_raw_labels_cache

    # unlike classification_samples, this method returns labels for all nodes from time begin to end
    # with pos labeled as 1, neg labeled as -1 and unknown labeled as 0
    def vertex_labels(self, target=4, return_name=False):
        rawlb, raw_names = self.__vertex_raw_labels(return_name=True)
        if target == 'raw':
            lb = rawlb
            label_names = raw_names
        else:
            lb = rawlb.copy()

            # TODO: make sure the order of lb (i.e. order of feat) is 0:nnodes
            def mapper(x):
                if x == target:
                    return 1
                elif x == -1:
                    return 0
                else:
                    return -1
            lb = np.vectorize(mapper)(lb)
            label_names = {-1: 'Others', 1: raw_names[target], 0: 'Unknown'}

        assert lb.shape == (len(self.gtgraphs), self.gtgraphs['any'].num_vertices()), \
            "{}, ({}, {})".format(lb.shape, len(self.gtgraphs), self.gtgraphs['any'].num_vertices())
        
        if return_name:
            return utils.OffsetList(self.localstep, len(lb), lb, copy=False), label_names
        else:
            return utils.OffsetList(self.localstep, len(lb), lb, copy=False)
