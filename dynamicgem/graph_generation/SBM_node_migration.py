import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import random
import networkx as nx
import operator
import SBM_graph

function_mapping = {'degree': nx.degree_centrality,
                    'eigenvector': nx.eigenvector_centrality,
                    'katz': nx.katz_centrality,
                    'closeness': nx.closeness_centrality,
                    'betweenness': nx.betweenness_centrality,
                    'load': nx.load_centrality,
                    'harmonic': nx.harmonic_centrality}


def get_node_color(node_community):
    cnames = [item[0] for item in matplotlib.colors.cnames.iteritems()]
    node_colors = [cnames[c] for c in node_community]
    return node_colors, cnames


def drawGraph(node_num, community_num, central_measure, topK, dc_id):
    my_graph = SBM_graph.SBMGraph(node_num, community_num)
    my_graph.sample_graph()
    G = my_graph._graph
    try:
        function = function_mapping[central_measure]
        if central_measure == 'katz':
            G_cen = function(G, alpha=0.01)
        else:
            G_cen = function(G)
    except KeyError:
        print(central_measure, 'is an invalid input using default cenratility measure: degree_centrality')
        G_cen = nx.degree_centrality(G)
        pass
    G_cen = sorted(G_cen.items(), key=operator.itemgetter(1), reverse=True)
    node_colors, _ = get_node_color(range(my_graph._community_num))

    pos = nx.spring_layout(G)
    nodelist = [[] for i in range(my_graph._community_num)]

    plt.figure()
    centralNodes = []
    count = 0
    ii = 0
    while count < topK:
        if my_graph._node_community[G_cen[ii][0]] == dc_id:
            centralNodes.append(G_cen[ii][0])
            count += 1
        ii += 1

    for i in range(my_graph._community_num):
        for j in range(my_graph._node_num):
            if my_graph._node_community[j] == i:
                nodelist[i].append(j)

    for i in range(my_graph._community_num):
        nx.draw_networkx_nodes(G, pos, nodelist=nodelist[i], node_color=node_colors[i], node_size=40, with_labels=False)
        nx.draw_networkx_edges(G, pos, width=0.1, arrows=False, alpha=0.8)

    for i in centralNodes:
        nodes_draw = nx.draw_networkx_nodes(G, pos, nodelist=[i], node_color=node_colors[my_graph._node_community[i]],
                                            node_size=500, with_labels=False)
        nodes_draw.set_edgecolor('w')

    labels = {}
    for i, nid in enumerate(centralNodes):
        labels[nid] = str("{0:.4f}".format(G_cen[i][1]))
    nx.draw_networkx_labels(G, pos, labels, nodelist=centralNodes, font_size=5, font_color='w')

    plt.title("Centrality measure: " + central_measure)
    plt.show()


if __name__ == '__main__':
    node_num = 1000
    community_num = 2
    node_change_num = 10
    length = 5
    criterias = ['degree',
                 'eigenvector',
                 'katz',
                 'betweenness',
                 'load',
                 'harmonic']

    for central_measure in criterias:
        drawGraph(node_num, community_num, central_measure, topK=5, dc_id=0)
