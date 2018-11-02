import sys
import os

rootpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(rootpath)

from core import gconfig

print("Checking mygraph submodule ...")
try:
    import core.mygraph
except ImportError as e:
    raise RuntimeError("mygraph submodule not avaiable: {}".format(str(e)))
print("OK\n")

print("Checking c extension ...")
try:
    import core.algorithm.dynamic_triad_cimpl
except ImportError as e:
    raise RuntimeError("c extention not available: {}".format(str(e)))
print("OK\n")

print("Checking Cython modules ...")
from core import utils
if gconfig.use_cython and utils.__impl__ != 'cython':
    raise RuntimeError("Cython not available")
print("OK\n")


