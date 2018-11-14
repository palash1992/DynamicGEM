from __future__ import print_function

import dynamicgem.dynamictriad.core.graphtool_utils as gtutils
import dynamicgem.dynamictriad.core.utils as utils
# import dynamicgem.dynamictriad.core.mygraph

import ctypes
import os
# mygraph=ctypes.cdll.LoadLibrary(os.path.realpath('')+'/dynamicgem/dynamictriad/core/mygraph.so')
import dynamicgem.dynamictriad.core.mygraph as mygraph
import dynamicgem.dynamictriad.core.mygraph_utils as mgutils

def graphtool2mygraph(g, **_):
    names = g.vp.get('name')
    if names:
        try:
            name_type = mgutils.format_type(names.value_type())
        except ValueError as e:
            print("Auto resolving type alias failed, try resolving with graph tool type system: " + e.message)
            name_type = mgutils.python2type(gtutils.type2python(names.value_type()))
    else:
        names = range(g.num_vertices())
        name_type = 'int'

    weight = g.ep.get('weight')
    if weight:
        try:
            weight_type = mgutils.format_type(names.value_type())
        except ValueError as e:
            print("Auto resolving type alias failed, try resolving with graph tool type system: " + e.message)
            weight_type = mgutils.python2type(gtutils.type2python(weight.value_type()))
    else:
        weight = utils.ConstantDict(1.0)
        weight_type = 'float'

    names = list(names)  # get rid of sluggish gt.vertex_properties

    mg = mygraph.Graph(name_type, weight_type)

    for n in names:
        mg.add_vertex(n)

    for e in g.edges():
        n1, n2 = names[int(e.source())], names[int(e.target())]
        # n1, n2 = names[e.source()], names[e.target()]
        if g.is_directed():
            mg.inc_edge(n1, n2, weight[e])
        else:
            mg.inc_edge(n1, n2, weight[e])
            mg.inc_edge(n2, n1, weight[e])

    return mg


# TODO: modify mygraph to make it support non-directed graph, and remove convert_to arg here
def mygraph2graphtool(g, convert_to=None, **_):
    vertices = g.vertices()
    ret = gtutils.load_mygraph_core(vertices, utils.KeyDefaultDict(lambda x: g.get(vertices[x])), directed=True,
                                    nametype=gtutils.python2type(mgutils.type2python(g.node_type())),
                                    weighttype=gtutils.python2type(mgutils.type2python(g.weight_type())),
                                    convert_to=convert_to, check=True)
    print("converting into graph {}".format(gtutils.graph_summary(ret)))
    return ret
