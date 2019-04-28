import matplotlib.pyplot as plt
import numpy as np
import random
import networkx as nx
import operator
import sys

sys.path.append('./')
from dynamicgem.graph_generation import SBM_graph
from dynamicgem.utils import graph_util

function_mapping = {'degree': nx.degree_centrality,
                    'eigenvector': nx.eigenvector_centrality,
                    'katz': nx.katz_centrality,
                    'closeness': nx.closeness_centrality,
                    'betweenness': nx.betweenness_centrality,
                    'load': nx.load_centrality,
                    'harmonic': nx.harmonic_centrality}


def _resample_egde_for_node(sbm_graph, node_id):
    if sbm_graph._graph is None:
        sbm_graph.sample_graph()
    else:
        n = sbm_graph._node_num
        for i in range(n):
            if i == node_id:
                continue
            if sbm_graph._graph.has_edge(node_id, i):
                sbm_graph._graph.remove_edge(node_id, i)
                sbm_graph._graph.remove_edge(i, node_id)
            prob = sbm_graph._B[sbm_graph._node_community[node_id], sbm_graph._node_community[i]]
            if np.random.uniform() <= prob:
                sbm_graph._graph.add_edge(node_id, i)
                sbm_graph._graph.add_edge(i, node_id)


def _resample_egde_for_node_v2(sbm_graph, node_id):
    if sbm_graph._graph is None:
        sbm_graph.sample_graph()
    else:
        n = sbm_graph._node_num
        for i in range(n):
            if i == node_id or sbm_graph._node_community[i] == sbm_graph._node_community[node_id]:
                if np.random.uniform() <= 0.04 and not sbm_graph._graph.has_edge(node_id, i):
                    sbm_graph._graph.add_edge(node_id, i)
                    sbm_graph._graph.add_edge(i, node_id)
                continue
            if sbm_graph._graph.has_edge(node_id, i):
                prob = sbm_graph._B[sbm_graph._node_community[node_id], sbm_graph._node_community[i]]
                if np.random.uniform() >= prob:
                    sbm_graph._graph.remove_edge(node_id, i)
                    sbm_graph._graph.remove_edge(i, node_id)
            # prob = sbm_graph._B[sbm_graph._node_community[node_id], sbm_graph._node_community[i]] 
            # if np.random.uniform() <= prob:
            #     sbm_graph._graph.add_edge(node_id, i)
            #     sbm_graph._graph.add_edge(i, node_id)                


def dyn_node_chng(sbm_graph, node_id):
    if sbm_graph._graph is None:
        sbm_graph.sample_graph()
    else:
        n = sbm_graph._node_num
        for i in range(n):
            if i == node_id:
                continue
            if sbm_graph._node_community[i] != sbm_graph._node_community[node_id]:
                if not sbm_graph._graph.has_edge(node_id, i):
                    prob = 0.1
                    if np.random.uniform() <= prob:
                        sbm_graph._graph.add_edge(node_id, i)
                        sbm_graph._graph.add_edge(i, node_id)
            else:
                if sbm_graph._graph.has_edge(node_id, i):
                    prob = 0.1
                    if np.random.uniform() <= prob:
                        sbm_graph._graph.remove_edge(node_id, i)
                        sbm_graph._graph.remove_edge(i, node_id)


def dyn_node_chng_v2(sbm_graph, node_id):
    if sbm_graph._graph is None:
        sbm_graph.sample_graph()
    else:
        n = sbm_graph._node_num
        othercommnodes = [i for i in range(n) if sbm_graph._node_community[i] != sbm_graph._node_community[node_id] if
                          not sbm_graph._graph.has_edge(node_id, i)]
        edgesnodes = random.sample(othercommnodes, 30)
        for i in edgesnodes:
            sbm_graph._graph.add_edge(node_id, i)
            sbm_graph._graph.add_edge(i, node_id)

        for i in range(n):
            if i == node_id:
                continue
            if sbm_graph._node_community[i] == sbm_graph._node_community[node_id]:
                if sbm_graph._graph.has_edge(node_id, i):
                    prob = 0.1
                    if np.random.uniform() <= prob:
                        sbm_graph._graph.remove_edge(node_id, i)
                        sbm_graph._graph.remove_edge(i, node_id)


def random_node_perturbation(sbm_graph, nodes_to_purturb):
    n = sbm_graph._node_num
    # Add a function to give perturbed_nodes based on adifferent criterias
    perturb_nodes = random.sample(range(n), nodes_to_purturb)
    for node_id in perturb_nodes:
        new_community = sbm_graph._node_community[node_id]
        while new_community == sbm_graph._node_community[node_id]:
            new_community = random.sample(range(sbm_graph._community_num), 1)[0]
        print('Node %d change from community %d to %d' % (node_id, sbm_graph._node_community[node_id], new_community))
        sbm_graph._node_community[node_id] = new_community
    for node_id in perturb_nodes:
        _resample_egde_for_node(sbm_graph, node_id)
    return perturb_nodes


def diminish_community(sbm_graph, community_id, nodes_to_purturb, criteria, criteria_r):
    n = sbm_graph._node_num
    community_nodes = [i for i in range(n) if sbm_graph._node_community[i] == community_id]
    nodes_to_purturb = min(len(community_nodes), nodes_to_purturb)
    labels = {}
    try:
        function = function_mapping[criteria]
        if criteria == 'katz':
            G_cen = function(sbm_graph._graph, alpha=0.01)
        else:
            G_cen = function(sbm_graph._graph)
    except KeyError:
        print(criteria, 'is an invalid input! Using degree_centrality instead.')
        G_cen = nx.degree_centrality(sbm_graph._graph)
        pass

    G_cen = sorted(G_cen.items(), key=operator.itemgetter(1), reverse=criteria_r)
    perturb_nodes = []
    count = 0
    i = 0
    while count < nodes_to_purturb:
        if sbm_graph._node_community[G_cen[i][0]] == community_id:
            perturb_nodes.append(G_cen[i][0])
            count += 1
        i += 1

    node_plot = []
    count = 0
    i = 0
    while count < 20:
        if sbm_graph._node_community[G_cen[i][0]] == community_id:
            node_plot.append(G_cen[i][0])
            count += 1
        i += 1

    node_plot_reverse = []
    count = 0
    i = len(G_cen) - 1
    while count < 20:
        if sbm_graph._node_community[G_cen[i][0]] == community_id:
            node_plot_reverse.append(G_cen[i][0])
            count += 1
        i -= 1

    for i, nid in enumerate(perturb_nodes):
        labels[nid] = str("{0:.2f}".format(G_cen[i][1]))
    del G_cen
    # perturb_nodes = random.sample(community_nodes, nodes_to_purturb)

    left_communitis = [i for i in range(sbm_graph._community_num) if i != community_id]
    for node_id in perturb_nodes:
        new_community = random.sample(left_communitis, 1)[0]
        print('Node %d change from community %d to %d' % (node_id,
                                                          sbm_graph._node_community[node_id],
                                                          new_community))
        sbm_graph._node_community[node_id] = new_community
    for node_id in perturb_nodes:
        _resample_egde_for_node(sbm_graph, node_id)

    return perturb_nodes, labels, node_plot, node_plot_reverse


def diminish_community_v2(sbm_graph, community_id, nodes_to_purturb, chngnodes):
    n = sbm_graph._node_num
    community_nodes = [i for i in range(n) if sbm_graph._node_community[i] == community_id]
    nodes_to_purturb = min(len(community_nodes), nodes_to_purturb)

    perturb_nodes = chngnodes

    # pos=nx.spring_layout(sbm_graph._graph)
    # color=['y','b']
    # plt.figure()
    # plt.subplot(311)
    # nx.draw_networkx_nodes(sbm_graph._graph,pos,node_size=500,node_color=[color[sbm_graph._node_community[p]] for p in sbm_graph._graph.nodes()])
    # nx.draw_networkx_edges(sbm_graph._graph,pos,arrows=False,width=1.0,alpha=0.5)
    # nx.draw_networkx_labels(sbm_graph._graph,pos,font_size=8)

    left_communitis = [i for i in range(sbm_graph._community_num) if i != community_id]
    for node_id in perturb_nodes:
        new_community = random.sample(left_communitis, 1)[0]
        print('Node %d change from community %d to %d' % (node_id,
                                                          sbm_graph._node_community[node_id],
                                                          new_community))
        sbm_graph._node_community[node_id] = new_community
    for node_id in perturb_nodes:
        _resample_egde_for_node_v2(sbm_graph, node_id)

    # plt.subplot(312)
    # nx.draw_networkx_nodes(sbm_graph._graph,pos,node_size=500,node_color=[color[sbm_graph._node_community[p]] for p in sbm_graph._graph.nodes()])
    # nx.draw_networkx_edges(sbm_graph._graph,pos,arrows=False,width=1.0,alpha=0.5)  
    # nx.draw_networkx_labels(sbm_graph._graph,pos,font_size=8)  
    # G_cen= nx.degree_centrality(sbm_graph._graph) 
    # G_cen = sorted(G_cen.items(), key=operator.itemgetter(1),reverse = False)
    # chngnodes=[]
    # count = 0
    # i     = 0
    # while count<nodes_to_purturb:
    #     if sbm_graph._node_community[G_cen[i][0]]==community_id:
    #         chngnodes.append(G_cen[i][0])
    #         count+=1
    #     i+=1

    nodes = [i for i in range(n) if sbm_graph._node_community[i] == community_id]

    chngnodes = random.sample(nodes, nodes_to_purturb)
    for node_id in chngnodes:
        dyn_node_chng_v2(sbm_graph, node_id)

    # print("Changed Nodes: ",chngnodes)   
    # plt.subplot(313)
    # nx.draw_networkx_nodes(sbm_graph._graph,pos,node_size=500,node_color=[color[sbm_graph._node_community[p]] for p in sbm_graph._graph.nodes()])
    # nx.draw_networkx_edges(sbm_graph._graph,pos,arrows=False,width=1.0,alpha=0.5) 
    # nx.draw_networkx_labels(sbm_graph._graph,pos,font_size=8)
    # plt.show()

    return perturb_nodes, chngnodes


def get_random_perturbation_series(node_num, community_num, length, nodes_to_purturb):
    my_graph = SBM_graph.SBMGraph(node_num, community_num)
    my_graph.sample_graph()

    graphs = [my_graph._graph.copy()]
    nodes_comunities = [my_graph._node_community[:]]
    perturbations = [[]]

    for i in range(length - 1):
        print('Step %d' % i)
        perturb_nodes = random_node_perturbation(my_graph, nodes_to_purturb)
        graphs.append(my_graph._graph.copy())
        nodes_comunities.append(my_graph._node_community[:])
        perturbations.append(perturb_nodes)

    return zip(graphs, nodes_comunities, perturbations)


def get_community_diminish_series(node_num,
                                  community_num,
                                  length,
                                  community_id,
                                  nodes_to_purturb,
                                  criteria,
                                  criteria_r):
    my_graph = SBM_graph.SBMGraph(node_num, community_num, community_id, nodes_to_purturb)
    my_graph.sample_graph_v3()
    chngnodes = my_graph._chngnodes

    graphs = [my_graph._graph.copy()]
    nodes_comunities = [my_graph._node_community[:]]
    perturbations = [[]]
    nodes_plot = [[]]
    nodes_plot_reverse = [[]]
    labels = [[]]
    for i in range(length - 1):
        print('Step %d' % i)

        perturb_nodes, label, node_plot, node_plot_reverse, chngnodes = diminish_community_v2(my_graph,
                                                                                              community_id,
                                                                                              nodes_to_purturb,
                                                                                              criteria,
                                                                                              criteria_r,
                                                                                              chngnodes)
        print("purturbed nodes")
        print(perturb_nodes)
        print("changed nodes")
        print(chngnodes)

        graphs.append(my_graph._graph.copy())
        nodes_comunities.append(my_graph._node_community[:])
        perturbations.append(perturb_nodes)
        labels.append(label)
        nodes_plot.append(node_plot)
        nodes_plot_reverse.append(node_plot_reverse)

    return zip(graphs, nodes_comunities, perturbations, labels, nodes_plot, nodes_plot_reverse)


def get_community_diminish_series_v2(node_num,
                                     community_num,
                                     length,
                                     community_id,
                                     nodes_to_purturb,
                                     ):
    my_graph = SBM_graph.SBMGraph(node_num, community_num, community_id, nodes_to_purturb)
    my_graph.sample_graph_v3()
    chngnodes = my_graph._chngnodes

    graphs = [my_graph._graph.copy()]
    nodes_comunities = [my_graph._node_community[:]]
    perturbations = [[]]
    dyn_change_nodes = [[]]
    for i in range(length - 1):
        print('Step %d' % i)

        print("Migrating Nodes")
        print(chngnodes)
        perturb_nodes, chngnodes = diminish_community_v2(my_graph,
                                                         community_id,
                                                         nodes_to_purturb,
                                                         chngnodes)
        print("Dynamically changed nodes")
        print(chngnodes)
        perturbations.append(perturb_nodes)
        dyn_change_nodes.append(chngnodes)

        graphs.append(my_graph._graph.copy())
        nodes_comunities.append(my_graph._node_community[:])

    return zip(graphs, nodes_comunities, perturbations, dyn_change_nodes)


def drawGraph(node_num, community_num):
    my_graph = SBM_graph.SBMGraph(node_num, community_num)
    my_graph.sample_graph()
    graphs = [my_graph._graph.copy()]
    nx.draw(graphs)


if __name__ == '__main__':
    node_num = 100
    community_num = 2
    node_change_num = 5
    length = 5
    get_community_diminish_series_v2(50,
                                     2,
                                     4,
                                     1,
                                     5)
    plt.show()
    # drawGraph(node_num, community_num)

    # prefix = 'data/synthetic/dynamic_SBM/node_pertuabtion_%d_%d_%d' % (node_num, community_num, node_change_num)
    # dynamic_sbm_series = get_random_perturbation_series(node_num, community_num, length, node_change_num)
    # graph_util.saveDynamicSBmGraph(prefix, dynamic_sbm_series)

    # prefix = 'data/synthetic/dynamic_SBM/community_diminish_%d_%d_%d' % (node_num, community_num, node_change_num)
    # dynamic_sbm_series = get_community_diminish_series(node_num, community_num, length, 1, node_change_num)
    # graph_util.saveDynamicSBmGraph(prefix, dynamic_sbm_series)
