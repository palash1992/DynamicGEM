from __future__ import print_function

import sys
import os
from six.moves import cPickle

try:
    from core import mygraph_utils as mgutils  # required to load core.mygraph
except ImportError:
    mainpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print("Error while importing core modules, try adding {} to python path".format(mainpath))
    sys.path.append(mainpath)

helpmsg = "python academic2edgelist.py <from> <to>\n" \
          "this converts the academic toy data to edgelist format\n" \
          "arguments:\n" \
          "\t<from>: academic data file\n" \
          "\t<to>: output directory"

if len(sys.argv) >= 2 and sys.argv[1] == '-h':
    print(helpmsg)
    exit(0)
elif len(sys.argv) != 3:
    print("exactly two arguments expected, use -h for help")
    exit(1)

data = cPickle.load(open(sys.argv[1], 'r'))

graphs = sorted(data['graphs'].items(), key=lambda x: x[0])
for i, (y, g) in enumerate(graphs):
    fn = "{}/{}".format(sys.argv[2], i)
    print("saving graph for year {} to file {}".format(y, fn))
    mgutils.save_adjlist(g, fn)
