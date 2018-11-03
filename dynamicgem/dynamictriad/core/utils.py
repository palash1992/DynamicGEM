import sys
from dynamicgem.dynamictriad.core import gconfig
from os import path

file_dir = path.dirname(path.abspath(__file__))
build_opts = ['build_ext', '--build-lib', file_dir, '--build-temp', file_dir + '/.cython_build']
_old_argv = sys.argv

try:
    if not gconfig.use_cython:
        raise ImportError("Cython disabled in config")
            
    from Cython.Build import cythonize
    from distutils.core import setup
    sys.argv = sys.argv[:1] + build_opts
    setup(name="utils_cy", ext_modules=cythonize(
        file_dir + "/cython_src/utils_cy.pyx",
        language="c++"
    ))
    sys.argv = _old_argv
    from utils_cy import *
    __impl__ = 'cython'
except ImportError as e:
    sys.argv = _old_argv
    print("Cython not avaiable, falling back to python implemented utils")
    print("Err msg: {}".format(e))
    from dynamicgem.dynamictriad.core.utils_py import *
    __impl__ = 'python'


# the utilities that have to be implemented in pure python
import multiprocessing as mp
import math
import itertools
import dill
import contextlib


def func_wrapper(args):
    func = dill.loads(args[0])
    return func(mp.current_process()._identity, *args[1:])


class ParMap(object):
    """
    work(process, list_of_samples, reportq): map function, maps a sequence of samples to the sequence of results
    monitor(queue): process monitor function, exit receiveing a StopIteration object
    """

    def __init__(self, work, monitor=None, njobs=mp.cpu_count(), maxtasksperchild=100):
        self.work_func = work
        self.monitor_func = monitor
        self.__njobs = njobs
        self.__mtpc = maxtasksperchild

        self.__pool = None

    def close(self):
        if self.__pool is not None:
            self.__pool.close()
            self.__pool.join()
        self.__pool = None

    def __del__(self):
        self.close()

    @property
    def njobs(self):
        return self.__njobs

    @njobs.setter
    def njobs(self, n):
        self.__njobs = n
        self.close()

    def default_chunk(self, dlen):
        return int(math.ceil(float(dlen) / self.njobs))

    def run(self, data, chunk=None, shuffle=False):
        if chunk is None:
            chunk = self.default_chunk(len(data))

        if shuffle:
            data, order, invorder = shuffle_sample(data, return_order=True)

        slices = slice_sample(data, chunk=chunk)
        res = self.run_slices(slices)

        if shuffle:
            res = apply_order(res, invorder)

        return res

    def run_slices(self, slices):
        mgr = mp.Manager()
        report_queue = mgr.Queue()
        if self.monitor_func is not None:
            monitor = mp.Process(target=self.monitor_func, args=(report_queue,))
            monitor.start()
        else:
            monitor = None

        if self.njobs == 1:
            res = []
            for slc in slices:
                res.append(self.work_func(None, slc, report_queue))
        else:
            dill_work_func = dill.dumps(self.work_func)
            with contextlib.closing(mp.Pool(self.njobs, maxtasksperchild=self.__mtpc)) as pool:
                res = pool.map(func_wrapper, [[dill_work_func, slc, report_queue] for slc in slices])
        res = list(itertools.chain.from_iterable(res))

        report_queue.put(StopIteration())
        if monitor is not None:
            monitor.join()

        return res
