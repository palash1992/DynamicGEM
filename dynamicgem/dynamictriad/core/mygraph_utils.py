from __future__ import print_function
import itertools
import math
import os
import ctypes
# mygraph=ctypes.cdll.LoadLibrary(os.path.realpath('')+'/dynamicgem/dynamictriad/core/mygraph.so')
import dynamicgem.dynamictriad.core.mygraph as mygraph


def type2python(tp):
    if tp == 'string':
        return str
    elif tp in ['short', 'int', 'long', 'long long', 'int16_t', 'int32_t', 'int64_t']:
        return int
    elif tp in ['double', 'float']:
        return float
    else:
        raise TypeError("Unknown type {}".format(tp))


def python2type(tp):
    if tp == int:
        return 'int'  # TODO: long is better, however, mygraph supports only int curretnly
    elif tp == str:
        return 'string'
    elif tp == float:
        return 'float'
    else:
        raise TypeError("Unsupported python type {}".format(tp))


# translate some typical type aliases
def format_type(tp):
    if tp in ['short', 'int16', 'int16_t']:
        return 'int16'
    elif tp in ['int', 'int32', 'int32_t']:
        return 'int32'
    elif tp in ['long', 'long long', 'int64', 'int64_t']:
        return 'int64'
    elif tp in ['float', 'real']:
        return 'float32'
    elif tp in ['double']:
        return 'float64'
    else:
        raise ValueError("Unknown Type {}".format(tp))


# TODO: add undirected mode
def save_graph(g, fn, fmt='adjlist'):
    if fmt == 'adjlist':
        save_adjlist(g, fn)
    # elif fmt == 'edgelist':
    #     save_edgelist(g, fn, weight=weight)
    # elif fmt == 'TNE':
    #     save_TNE(g, fn, weight=weight)
    else:
        raise RuntimeError("Unknown graph format {}".format(fmt))


# graph is directed and weighted by default
def save_adjlist(g, fn):
    fh = open(fn, 'w')

    nodes = g.vertices()
    for i in range(len(nodes) - 1):
        assert nodes[i] < nodes[i + 1], (nodes[i], nodes[i + 1])

    for i in nodes:
        nbrs = g.get(i)  # [(nbr, w), ...]
        strnbr = ' '.join([str(e) for e in itertools.chain.from_iterable(nbrs)])
        print("{} {}".format(i, strnbr), file=fh)
    fh.close()


def load_adjlist(fn, node_type='string', weight_type='float'):
    """
    loads only undirected graph, if multiple instances of the same edge is detected,
    their weights are summed up
    :param fn:
    :param node_type:
    :param weight_type:
    :return:
    """
    py_node_type = type2python(node_type)
    py_weight_type = type2python(weight_type)

    edgeset = set()  # check if the graph is undirected
    g = mygraph.Graph(node_type, weight_type)
    for line in open(fn, 'r'):
        fields = line.split()

        n = py_node_type(fields[0])
        if not g.exists(n):
            g.add_vertex(n)

        for v, w in zip(fields[1::2], fields[2::2]):
            v = py_node_type(v)
            w = py_weight_type(w)

            if v == n:
                print("[warning] loopback edge ({}, {}) detected".format(v, n))
                continue

            if not g.exists(v):
                g.add_vertex(v)
            
            if g.exists(n, v):
                raise RuntimeError("Multiple edges ({}, {}) found in {}".format(n, v, fn))

            if g.exists(v, n):  # check if the graph is undirected
                assert math.fabs(w - g.edge(v, n)) < 1e-6, \
                    "Inconsistent edge weight on ({}, {}), the graph is not undirected?" \
                    .format(v, n)
                edgeset.remove((v, n))
            else:
                edgeset.add((n, v))

            g.inc_edge(n, v, w)
    if len(edgeset) > 0:
        raise RuntimeError("One-sided edges detected".format(edgeset))
    return g

