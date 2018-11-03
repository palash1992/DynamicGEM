from __future__ import print_function

import graph_tool as gt
import itertools
from collections import defaultdict
import numpy as np
import dynamicgem.dynamictriad.core.utils
try:
    from itertools import izip
except:
    izip = zip    

__gtutils_debug = True


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
        return 'long'
    elif tp == str:
        return 'string'
    elif tp == float:
        return 'double'
    else:
        raise TypeError("Unsupported python type {}".format(tp))


def graph_summary(g):
    return "nsize: {}, esize: {}, weight_type: {}, name_type: {}, directed: {}". \
        format(g.num_vertices(), g.num_edges(), g.ep['weight'].python_value_type(), g.vp['name'].python_value_type(), g.is_directed())


def load_edge_list(fn, directed=True, nodename=None, nametype='string', convert_to=None):
    # load labels
    edge_list = utils.open_datasrc(fn).read().rstrip('\n').split('\n')
    edge_list = [e.split() for e in edge_list]
    namepytype = type2python(nametype)
    nodename_data = set([namepytype(l[i]) for l in edge_list for i in (0, 1)])
    if nodename is not None:
        unknown_nodes = nodename_data - set(nodename)
        if len(unknown_nodes) > 0:
            raise RuntimeError("Unknown nodes {} appeared in edge list {}".format(unknown_nodes, fn))
    else:
        nodename = list(sorted(nodename_data))

    name2idx = {n: i for i, n in enumerate(nodename)}

    g = gt.Graph(directed=directed)
    g.add_vertex(len(nodename))

    name = g.new_vp(nametype)
    for i in range(g.num_vertices()):
        name[g.vertex(i)] = nodename[i]
    g.vertex_properties['name'] = name

    wcache = {}
    for i in range(len(edge_list)):
        edge_list[i][0] = name2idx[edge_list[i][0]]
        edge_list[i][1] = name2idx[edge_list[i][1]]
        if len(edge_list[i]) > 2:
            edge_list[i][2] = float(edge_list[i][2])
        else:
            edge_list[i].append(1.0)

        # integrity check
        f, t = edge_list[i][:2]
        if not directed:
            f, t = min(f, t), max(f, t)
        if (f, t) in wcache:
            raise RuntimeError("Duplicated edge {} to {} in {}".format(nodename[f], nodename[t], fn))
        wcache[(f, t)] = edge_list[i][2]

    weight = g.new_ep('float')
    g.add_edge_list(edge_list, eprops=[weight])

    return g


# note that mygraph format is always directed, it is converted into an undirected if convert_to=False
# TODO: move conversion code from directed to undirected into a separate method
def load_mygraph_core(vertices, vid2elist, directed=True, nametype=None, weighttype=None, convert_to=None, check=True):

    if directed is None:
        directed = True
    if not directed:
        raise RuntimeError("mygraph format is always directed at the moment, maybe you mean convert_to='undirected'?")
    if convert_to == 'directed':
        convert_to = True
    elif convert_to == 'undirected':
        convert_to = False
    else:
        convert_to = directed

    if nametype is None:
        namepytype = type(vertices[0])
        nametype = python2type(namepytype)
    else:
        namepytype = type2python(nametype)

    if weighttype is None:
        weightpytype = type(vid2elist[0])
        weighttype = python2type(weightpytype)
    else:
        weightpytype = type2python(weighttype)

    g = gt.Graph(directed=convert_to)  # because mygraph support only directed graphs
    g.add_vertex(len(vertices))
    # we do not rstrip('\n') here because there is a possible blank line at the end

    # TODO: ugly fix by imposing an order here, that is, we assume the vertices in mygraph are in increasing order,
    # TODO: and generate vertex order in graphtool accordingly, even though mygraph is actually unordered
    # TODO: a better solution is to add order in mygraph
    for i in range(len(vertices) - 1):
        assert namepytype(vertices[i]) < namepytype(vertices[i + 1]), "{} {}".format(namepytype(vertices[i]), namepytype(vertices[i + 1]))
    
    names = g.new_vertex_property(nametype)
    for i, k in enumerate(vertices):
        names[i] = namepytype(k)
    g.vertex_properties['name'] = names
    name2id = {v: i for i, v in enumerate(names)}

    weight = g.new_edge_property(weighttype)
    edge_cache = defaultdict(lambda: [0, 0])  # weight, flag: two bits for two edge directions
    for i, k in enumerate(vertices):
        recs = vid2elist[i]  # list of tuples
        k = namepytype(k)
        for rk, rv in recs:
            rv = weightpytype(rv)
            kid, rkid = name2id[k], name2id[namepytype(rk)]
            if kid == rkid:
                print("[debug] loopback edge detected: ({}, {})".format(k, rk))
            # check for conflict
            if convert_to:
                if check and (kid, rkid) in edge_cache:
                    raise RuntimeError("multiple edge {}-{}:{}".format(k, rk, rv))
                else:
                    edge_cache[(kid, rkid)][0] = rv
            else:
                key = (min(kid, rkid), max(kid, rkid))
                flag = 1 if kid > rkid else 2
                rec = edge_cache[key]
                if rec[1] & flag != 0:
                    raise RuntimeError("Duplicated edge ({}, {})".format(kid, rkid))
                if rec[1] != 0 and rec[0] != rv:
                    raise RuntimeError("Not a valid undirected graph, ({}, {})={}, ({}, {})={}"
                                       .format(kid, rkid, rv, rkid, kid, rec[0]))
                rec[1] |= flag
                rec[0] = rv
    edgearr = np.zeros((len(edge_cache), 3))
    for i, (k, v) in enumerate(edge_cache.items()):
        if not convert_to and v[1] != 3:  # both directions must appear
            raise RuntimeError("Edge ({}, {}) appears in only one direction".format(k[0], k[1]))
        edgearr[i] = [k[0], k[1], v[0]]
    g.add_edge_list(edgearr, eprops=[weight])
    g.edge_properties['weight'] = weight

    assert g.is_directed() == convert_to
    return g


def load_mygraph(fn, directed=True, nodename=None, nametype='string', convert_to=None):
    if nodename is not None:
        raise NotImplementedError("given nodename is not supported in load_mygraph")

    data = utils.open_datasrc(fn).read().split('\n')[1:]  # skip the first line counting number of vertices
    if data[-1] == '' and len(data) % 2 == 1:  # try to fix length problem
        data = data[:-1]
    assert len(data) % 2 == 0, "{} {}".format(len(data), str(data))

    vertices = data[::2]
    if nametype == 'string':
        vertices = [v[v.find('@') + 1:] for v in vertices]
    elist = data[1::2]

    def str2elist(s):
        arr = s.split()[1:]  # discard edge cnt at the beginning of this line
        if nametype == 'string':
            evertices = [v[v.find('@') + 1:] for v in arr[::2]]
        else:
            evertices = arr[::2]
        return izip(evertices, arr[1::2])

    vid2elist = utils.KeyDefaultDict(lambda x: str2elist(elist[x]))
    return load_mygraph_core(vertices, vid2elist, directed=directed, nametype=nametype, weighttype='float',
                             convert_to=convert_to, check=True)

# def load_mygraph(fn, directed=True, nodename=None, nametype='string', convert_to=None):
#     if directed is None:
#         directed = True
#     if not directed:
#         raise RuntimeError("mygraph format is always directed at the moment, maybe you mean convert_to='undirected'?")
#     if convert_to == 'directed':
#         convert_to = True
#     elif convert_to == 'undirected':
#         convert_to = False
#     else:
#         convert_to = directed
#     if nodename is not None:
#         raise NotImplementedError("given nodename is not supported in load_mygraph")
#
#     g = gt.Graph(directed=convert_to)  # because mygraph support only directed graphs
#
#     # we do not rstrip('\n') here because there is a possible blank line at the end
#     data = utils.open_datasrc(fn).read().split('\n')
#     if data[-1] == '' and len(data) % 2 == 1:  # try to fix length problem
#         data = data[:-1]
#     assert len(data) % 2 == 0, "{} {}".format(len(data), str(data))
#     g.add_vertex(len(data) / 2)
#
#     nodepytype = type2python(nametype)
#     names = g.new_vertex_property(nametype)
#     for i, k in enumerate(data[0::2]):
#         names[i] = nodepytype(k)
#     g.vertex_properties['name'] = names
#     name2id = {v: i for i, v in enumerate(names)}
#
#     weight = g.new_edge_property('float')
#     edge_cache = defaultdict(lambda: [0, 0])  # weight, flag: two bits for two edge directions
#     for k, v in zip(data[0::2], data[1::2]):
#         k = nodepytype(k)
#         recs = v.split()[1:]  # discard edge count
#         for rk, rv in zip(recs[0::2], recs[1::2]):
#             kid, rkid = name2id[k], name2id[nodepytype(rk)]
#             if kid == rkid:
#                 print("[debug] loopback edge detected: ({}, {})".format(k, rk))
#             # check for conflict
#             if convert_to:
#                 if (kid, rkid) in edge_cache:
#                     raise RuntimeError("multiple edge {}-{}:{}".format(k, rk, rv))
#                 else:
#                     edge_cache[(kid, rkid)][0] = float(rv)
#             else:
#                 key = (min(kid, rkid), max(kid, rkid))
#                 flag = 1 if kid > rkid else 2
#                 rec = edge_cache[key]
#                 if rec[1] & flag != 0:
#                     raise RuntimeError("Duplicated edge ({}, {})".format(kid, rkid))
#                 if rec[1] != 0 and rec[0] != float(rv):
#                     raise RuntimeError("Not a valid undirected graph, ({}, {})={}, ({}, {})={}"
#                                        .format(kid, rkid, float(rv), rkid, kid, rec[0]))
#                 rec[1] |= flag
#                 rec[0] = float(rv)
#     edgearr = np.zeros((len(edge_cache), 3))
#     for i, (k, v) in enumerate(edge_cache.items()):
#         if not convert_to and v[1] != 3:  # both directions must appear
#             raise RuntimeError("Edge ({}, {}) appears in only one direction".format(k[0], k[1]))
#         edgearr[i] = [k[0], k[1], v[0]]
#     g.add_edge_list(edgearr, eprops=[weight])
#     g.edge_properties['weight'] = weight
#
#     assert g.is_directed() == convert_to
#     return g


# load mygraph format, which is always directed
# if directed=False, the directed mygraph format is converted to non-directed if possible
def load_graph(fn, fmt='mygraph', directed=None, nodename=None, nametype='string', convert_to=None):
    if fmt == 'mygraph':
        return load_mygraph(fn, directed=directed, nodename=nodename, nametype=nametype, convert_to=convert_to)
    elif fmt == 'edgelist':
        return load_edge_list(fn, directed=directed, nodename=nodename, nametype=nametype, convert_to=convert_to)
    else:
        raise NotImplementedError


def save_graph(g, fn, fmt='adjlist', weight=None):
    if fmt == 'adjlist':
        save_adjlist(g, fn, weight=weight)
    elif fmt == 'edgelist':
        save_edgelist(g, fn, weight=weight)
    elif fmt == 'TNE':
        save_TNE(g, fn, weight=weight)
    else:
        raise RuntimeError("Unkonwn graph format {}".format(fmt))


def save_adjlist(g, fn, weight=None):
    fh = open(fn, 'w')
    nodeidx = list(sorted([int(v) for v in g.vertices()]))
    for i in nodeidx:
        if g.is_directed():
            nbrs = [int(n) for n in g.vertex(i).out_neighbours()]
        else:
            nbrs = [int(n) for n in g.vertex(i).all_neighbours()]
        if weight is None:
            strnbr = ' '.join([str(n) for n in nbrs])
        else:
            w = [weight[g.edge(i, n)] for n in nbrs]
            assert len(nbrs) == len(w)
            strnbr = ' '.join([str(e) for e in itertools.chain.from_iterable(zip(nbrs, w))])
        print("{} {}".format(i, strnbr), file=fh)
    fh.close()


def save_edgelist(g, fn, weight=None):
    fh = open(fn, 'w')
    # if we don't care about order
    for e in g.edges():
        if weight is None:
            print("{} {}".format(int(e.source()), int(e.target())), file=fh)
        else:
            print("{} {} {}".format(int(e.source()), int(e.target()), weight[e]), file=fh)
    fh.close()


# TNE format is an undirected format defined here
# https://github.com/linhongseba/Temporal-Network-Embedding
def save_TNE(g, fn, weight=None):
    assert not g.is_directed()
    
    fh = open(fn, 'w')
    # in order to speed up edge access
    edge_cache = {}
    
    if weight is None:  # in this format, a weight must be given
        weight = defaultdict(lambda x: 1.0)

    for e in g.edges():
        isrc, itgt = int(e.source()), int(e.target())
        # isrc, itgt = min(isrc, itgt), max(isrc, itgt)
        edge_cache[(isrc, itgt)] = weight[e]
        edge_cache[(itgt, isrc)] = weight[e]
    # w = None

    print(g.num_vertices(), file=fh)
    for i in range(g.num_vertices()):
        outnbr = [int(v) for v in list(g.vertex(i).out_neighbours())]
        #if len(outnbr) == 0:  # for debug
        #    continue
        outnbr = list(sorted(outnbr))  # in ascending order
        w = [edge_cache[(i, v)] for v in outnbr]
        fields = ['{},{}'.format(i, len(outnbr))] + ["{},{}".format(a, b) for a, b in zip(outnbr, w)]
        print(':'.join(fields), file=fh)
    fh.close()


# rawnames: either an iterable (including a property map), or a list of iterables
# in the first case, it is a global name list, and a list of local name lists in the second case
# weights: a list of weights, where each weight is a dict from edge to float
def merge_graph(graphs, rawnames, weights=None, directed=False, name_type=None, weight_type=None):
    # if directed:
    #     raise NotImplementedError("merge_graph support only merging from undirected to undirected")

    name2id = None
    if hasattr(next(iter(rawnames)), '__iter__') and not isinstance(rawnames[0], str):
        # each element is a namemap, in this case, rawnames are considered unique IDs of nodes
        if name_type is None:
            try:
                name_type = rawnames[0].value_type()
            except AttributeError:
                name_type = python2type(rawnames[0][0])
        unified_names = list(sorted(set([n for namelist in rawnames for n in namelist])))
        name2id = {n: i for i, n in enumerate(unified_names)}
    else:
        # rawnames is a unified namemap, in this case, index is considered the unique ID of a node,
        # while rawnames are only ordinary attributes
        if name_type is None:
            try:
                name_type = rawnames.value_type()
            except AttributeError:
                name_type = python2type(type(next(iter(rawnames))))
        unified_names = list(rawnames)

    # we consider all unweighted edges as weight=1, and a weight property is forcibly added
    if weights is None:
        weights = utils.ConstantDict(utils.ConstantDict(1))
        if weight_type is None:
            weight_type = 'float'
    else:
        if weight_type is None:
            weight_type = weights[0].value_type()

    nodecnt = len(unified_names)

    g = gt.Graph(directed=directed)
    g.add_vertex(nodecnt)
    w = g.new_ep(weight_type)
    g.edge_properties['weight'] = w
    name = g.new_vp(name_type)
    for i in range(g.num_vertices()):
        name[i] = unified_names[i]  # do not use rawnames.a in case rawnames.value_type() == 'string'
    g.vertex_properties['name'] = name

    for i in range(len(graphs)):
        if graphs[i].is_directed() != directed:
            raise RuntimeError("graph {} has different directional property from given directed={}".format(i, directed))

    edge_cache = defaultdict(lambda: 0)
    for i in range(len(graphs)):
        wi = weights[i]
        for e in graphs[i].edges():
            isrc, itgt = int(e.source()), int(e.target())
            if name2id is not None:  # map to unified names
                isrc, itgt = name2id[rawnames[i][isrc]], name2id[rawnames[i][itgt]]
            if not directed:
                isrc, itgt = min(isrc, itgt), max(isrc, itgt)
            edge_cache[(isrc, itgt)] += wi[e]
    edgearr = np.zeros((len(edge_cache), 3))
    for i, (k, v) in enumerate(edge_cache.items()):
        edgearr[i] = [k[0], k[1], v]
    g.add_edge_list(edgearr, eprops=[w])

    # for debug
    if __gtutils_debug:
        for e in g.edges():
            isrc, itgt = int(e.source()), int(e.target())
            if not directed:
                isrc, itgt = min(isrc, itgt), max(isrc, itgt)
            assert w[e] == edge_cache[(isrc, itgt)]
        print("Merged into graph (nsize: {}, esize: {}, weight_type: {}, name_type: {}, directed: {})"
              .format(nodecnt, g.num_edges(), weight_type, name_type, directed))

    return g
