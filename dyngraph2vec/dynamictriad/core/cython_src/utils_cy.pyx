# distutils: language=c++

from __future__ import print_function
import numpy as np

from collections import defaultdict
import itertools
import sys
from copy import deepcopy
from libcpp cimport bool
from libcpp.map cimport map
from libc.stdlib cimport rand, srand, RAND_MAX
from libc.time cimport time as ctime


class KeyDefaultDict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret


def slice_sample(sample, chunk=None, nslice=None):
    cdef int ichunk
    cdef int curstart

    slices = []
    if chunk is None:
        ichunk = int(len(sample) / nslice)
    else:
        if nslice is not None:
            raise RuntimeError("chunk ({}) and slice ({}) should not be specified simultaneously".format(chunk, nslice))
        else:
            ichunk = int(chunk)

    curstart = 0
    while True:
        if curstart >= len(sample):
            break
        slices.append(sample[curstart:min(curstart + ichunk, len(sample))])
        curstart += ichunk

    return slices


def islice_sample(sample, chunk=None, nslice=None):
    if chunk is None:
        chunk = int(len(sample) / nslice)
    else:
        if nslice is not None:
            raise RuntimeError("chunk ({}) and slice ({}) should not be specified simultaneously".format(chunk, nslice))

    curstart = 0
    while True:
        if curstart >= len(sample):
            break
        yield sample[curstart:min(curstart + chunk, len(sample))]
        curstart += chunk


def shuffle_sample(sample, return_order=False):
    # type: (iterable) -> tuple
    order = np.random.permutation(np.arange(len(sample)))
    invorder = np.zeros((len(sample), ), dtype='int32')
    invorder[order] = np.arange(len(sample))
    
    if return_order:
        return apply_order(sample, order), order, invorder
    else:
        return apply_order(sample, order)


def apply_order(sample, order):
    return [sample[o] for o in order]


# archive protocol
cdef class Archivable(object):
    def archive(self, name=None, copy=True):
        return {}

    def load_archive(self, ar, copy=True, name=None):
        pass


cdef class OffsetList(Archivable):
    cdef public int offset
    cdef public int length
    cdef public int __iter
    cdef public bool __managed
    cdef public map[int, bool] __accessed
    cdef public object __factory
    cdef public object __items

    def __cinit__(self, offset, length, datasrc, copy=True, managed=None):
        self.offset = offset
        self.length = length
        if managed is None:
            self.__managed = False
        else:
            self.__managed = managed

        self.__iter = 0

    def __init__(self, offset, length, datasrc, copy=True, managed=None):
        if hasattr(datasrc, '__getitem__'):
            if not copy:
                self.__items = datasrc
                if managed is None:
                    self.__managed = False
            else:
                self.__items = deepcopy(datasrc)
                if managed is None:
                    self.__managed = True
            self.__factory = lambda x: self.__items[x - self.offset]
        else:
            self.__items = [None] * self.length
            self.__factory = datasrc
            if managed is None:
                self.__managed = True
        # self.__iter = 0
        # self.__accessed = {}

    def __len__(self):
        return self.length

    cdef __normalize_slice(self, slc):
        cdef int start, stop, step
        if slc.start is None:
            start = self.offset
        else:
            start = int(slc.start)
        if slc.stop is None:
            stop = self.offset + self.length
        else:
            stop = int(slc.stop)
        if slc.step is None:
            step = 1
        else:
            step = int(slc.step)
        start = self.__normalize_neg_index(start)
        stop = self.__normalize_neg_index(stop)
        return slice(start, stop, step)

    cdef int __normalize_neg_index(self, int idx):
        if idx < 0:
            return idx + self.offset + self.length
        else:
            return idx

    def __setitem__(self, key, item):
        if isinstance(key, slice):
            slc = self.__normalize_slice(key)
            rg = range(slc.start, slc.stop, slc.step)

            if len(rg) != len(item):
                raise ValueError("Trying to set {} items with {} value".format(len(rg), len(item)))
            
            if slc.start < self.offset or slc.stop > self.offset + self.length:
                raise KeyError("{} not in range [{}, {})"
                               .format(slc, self.offset, self.offset + self.length))
            for step, itm in zip(rg, item):
                self._store_item(step, itm)
        else:
            key = self.__normalize_neg_index(key)
            if key < self.offset or key >= self.offset + self.length:
                raise KeyError("{} not in range [{}, {})"
                               .format(key, self.offset, self.offset + self.length))
            self._store_item(key, item)

    def __getitem__(self, key):
        if key == 'any':
            return self._load_item(self.offset)
        
        if isinstance(key, slice):
            slc = self.__normalize_slice(key)
            rg = range(slc.start, slc.stop, slc.step)
            
            if slc.start < self.offset or slc.stop > self.offset + self.length:
                raise KeyError("{} not in range [{}, {})"
                               .format(slc, self.offset, self.offset + self.length))
            items = []
            for step in rg:
                items.append(self._load_item(step))
            return items
        else:
            key = self.__normalize_neg_index(int(key))
            if key < self.offset or key >= self.offset + self.length:
                raise KeyError("{} not in range [{}, {})"
                               .format(key, self.offset, self.offset + self.length))
            return self._load_item(int(key))
    
    def __array__(self):
        # np.array fails when len(self) == 1, I have no idea why this happens but have to specify array interface manually
        return np.asarray(list(self))

    def __iter__(self):
        self.__iter = self.offset
        return self

    # for py3 compatibility
    # def __next__(self):
    #    return self.next()

    def append(self, item):
        if not self.__managed:
            raise RuntimeError("Cannot append to unmanaged OffsetList")
        self.length += 1
        self.__items.append(item)

    def extend(self, lst):
        if not self.__managed:
            raise RuntimeError("Cannot extend an unmanaged OffsetList")
        self.length += len(lst)
        self.__items.extend(lst)

    def __next__(self):
        if self.__iter >= self.offset + self.length:
            raise StopIteration
        ret = self._load_item(self.__iter)
        self.__iter += 1
        return ret

    cdef _load_item(self, int step):
        # do some caching
        #if self.__accessed.get(step, None) is None:
        if self.__accessed.find(step) == self.__accessed.end():
            self.__items[step - self.offset] = self.__factory(step)
            self.__accessed[step] = True
        return self.__items[step - self.offset]

    def _store_item(self, step, itm):
        self.__items[step - self.offset] = itm
        self.__accessed[step] = True

    def archive(self, name=None, copy=True):
        if name is None:
            prefix = 'OffsetList'
        else:
            prefix = '{}_OffsetList'.format(name)

        ar = super(OffsetList, self).archive(name=name, copy=copy)
        ar['{}_offset'.format(prefix)] = self.offset
        ar['{}_length'.format(prefix)] = self.length
        ar['{}_data'.format(prefix)] = list(self)
        if copy:
            ar['{}_data'.format(prefix)] = deepcopy(ar['{}_data'.format(prefix)])
        return ar

    def load_archive(self, ar, copy=False, name=None):
        if name is None:
            prefix = 'OffsetList'
        else:
            prefix = '{}_OffsetList'.format(name)

        self.__init__(ar['{}_offset'.format(prefix)], ar['{}_length'.format(prefix)], ar['{}_data'.format(prefix)], copy=copy)


# TODO: optimize this
def group_by(data, key=lambda x: x):
    ret = []
    key2idx = {}
    for d in data:
        k = key(d)
        idx = key2idx.get(k, None)
        if idx is None:
            idx = key2idx[k] = len(key2idx)
            ret.append([])
        ret[idx].append(d)
    return ret


cpdef int crandint(int ub):
    if ub <= 0:
        return 0

    return rand() % ub


# __init__
seed = ctime(NULL)
#seed = 1497269812
srand(seed)
#open("/tmp/random.seed", "w").write(str(seed))

