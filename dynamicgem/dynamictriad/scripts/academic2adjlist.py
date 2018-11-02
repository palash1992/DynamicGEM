from __future__ import print_function

import sys
import os
from six.moves import cPickle
import regex as re

try:
    from core import mygraph_utils as mgutils  # required to load core.mygraph
except ImportError:
    mainpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print("Error while importing core modules, try adding {} to python path".format(mainpath))
    sys.path.append(mainpath)

helpmsg = "python academic2edgelist.py <from> <to>\n" \
          "this converts the academic toy data to edgelist format\n" \
          "arguments:\n" \
          "\t<from>[,<begin>:<end>]: academic data file\n" \
          "\t<to>: output directory"

if len(sys.argv) >= 2 and sys.argv[1] == '-h':
    print(helpmsg)
    exit(0)
elif len(sys.argv) != 3:
    print("exactly two arguments expected, use -h for help")
    exit(1)

mat = re.match(r"([^,]+):(\d+),(\d+)", sys.argv[1])
if mat:
    infn = mat.captures(1)[0]
    begin = int(mat.captures(2)[0])
    end = int(mat.captures(3)[0])
else:
    infn = sys.argv[1]
    begin = end = None

data = cPickle.load(open(infn, 'r'))

graphs = sorted(data['graphs'].items(), key=lambda x: int(x[0]))
outidx = 0
for y, g in graphs:
    if begin is not None and int(y) < begin:
        continue
    if end is not None and int(y) >= end:
        break
    fn = "{}/{}".format(sys.argv[2], outidx)
    print("saving graph for year {} to file {}".format(y, fn))
    mgutils.save_adjlist(g, fn)
    
    outidx += 1
