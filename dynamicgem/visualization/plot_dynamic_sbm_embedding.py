disp_avlbl = True
import os

if os.name == 'posix' and 'DISPLAY' not in os.environ:
    disp_avlbl = False
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import networkx as nx
import numpy as np
from sklearn.manifold import TSNE
import sys

sys.path.append('./')
import pdb

from dynamicgem.graph_generation import dynamic_SBM_graph
from dynamicgem.utils import graph_util
from dynamicgem.utils import plot_util
from .plot_static_embedding import *
from matplotlib import rc
import seaborn

font = {'family': 'serif', 'serif': ['computer modern roman']}
rc('text', usetex=True)
rc('font', weight='bold')
rc('font', size=20)
rc('lines', markersize=10)
rc('xtick', labelsize=12)
rc('ytick', labelsize=12)
rc('axes', labelsize='x-large')
rc('axes', labelweight='bold')
rc('axes', titlesize='x-large')
rc('axes', linewidth=3)
plt.rc('font', **font)
seaborn.set_style("darkgrid")


def plot_single_step(node_pos, graph_info, graph_info_next, changed_node):
    node_colors = plot_util.get_node_color(graph_info_next[1])
    node_num, embedding_dimension = node_pos.shape
    pos = {}
    for i in range(node_num):
        pos[i] = node_pos[i, :]
    unchanged_nodes = list(set(range(node_num)) - set(changed_node))

    nodes_draw = nx.draw_networkx_nodes(graph_info[0],
                                        pos,
                                        nodelist=unchanged_nodes,
                                        node_color=[node_colors[p] for p in unchanged_nodes],
                                        node_size=40,
                                        with_labels=False)
    nodes_draw.set_edgecolor('w')

    # nodes_draw=nx.draw_networkx_nodes(graph_info[0], 
    #                        pos, 
    #                        nodelist=graph_info_next[4], 
    #                        node_color='r', 
    #                        node_size=50, 
    #                        with_labels=False)
    # nodes_draw.set_edgecolor('w')

    # nodes_draw=nx.draw_networkx_nodes(graph_info[0], 
    #                        pos, 
    #                        nodelist=graph_info_next[5], 
    #                        node_color='g', 
    #                        node_size=50, 
    #                        with_labels=False)
    # nodes_draw.set_edgecolor('w')

    nodes_draw = nx.draw_networkx_nodes(graph_info[0],
                                        pos,
                                        nodelist=changed_node,
                                        node_color='r',
                                        node_size=80,
                                        with_labels=False)
    nodes_draw.set_edgecolor('k')


def plot_single_step_v2(node_pos, graph_info, graph_info_next, purturbed_nodes, dyn_changed_node):
    node_colors = plot_util.get_node_color(graph_info_next[1])
    node_num, embedding_dimension = node_pos.shape
    pos = {}
    for i in range(node_num):
        pos[i] = node_pos[i, :]
    # unchanged_nodes = list(set(range(node_num)) - set(purturbed_nodes) - set(dyn_changed_node))
    unchanged_nodes = list(set(range(node_num)) - set(dyn_changed_node))

    nodes_draw = nx.draw_networkx_nodes(graph_info[0],
                                        pos,
                                        nodelist=unchanged_nodes,
                                        node_color=[node_colors[p] for p in unchanged_nodes],
                                        node_size=40,
                                        with_labels=False)
    nodes_draw.set_edgecolor('w')

    # nodes_draw=nx.draw_networkx_nodes(graph_info[0], 
    #                        pos, 
    #                        nodelist=graph_info_next[4], 
    #                        node_color='r', 
    #                        node_size=50, 
    #                        with_labels=False)
    # nodes_draw.set_edgecolor('w')

    # nodes_draw=nx.draw_networkx_nodes(graph_info[0], 
    #                        pos, 
    #                        nodelist=graph_info_next[5], 
    #                        node_color='g', 
    #                        node_size=50, 
    #                        with_labels=False)
    # nodes_draw.set_edgecolor('w')

    # nodes_draw=nx.draw_networkx_nodes(graph_info[0], 
    #                                   pos, 
    #                                   nodelist=purturbed_nodes, 
    #                                   node_color='r', 
    #                                   node_size=80, 
    #                                   with_labels=False)
    # nodes_draw.set_edgecolor('k')

    nodes_draw = nx.draw_networkx_nodes(graph_info[0],
                                        pos,
                                        nodelist=dyn_changed_node,
                                        node_color='r',
                                        node_size=80,
                                        with_labels=False)
    nodes_draw.set_edgecolor('k')


def plot_dynamic_sbm_embedding(nodes_pos_list, dynamic_sbm_series):
    # print("dynamic_sbm_series: ", len(dynamic_sbm_series)) 
    # print("nodes_pos_list: ", len(nodes_pos_list)) 
    length = len(dynamic_sbm_series) - 1
    # length = len(dynamic_sbm_series)
    node_num, dimension = nodes_pos_list[0].shape

    if dimension > 2:
        print("Embedding dimension greater than 2, using tSNE to reduce it to 2")
        model = TSNE(n_components=2, random_state=42)
        nodes_pos_list = [model.fit_transform(X) for X in nodes_pos_list]

    pos = 1
    for t in range(length):
        # print(t)
        plt.subplot(220 + pos)
        pos += 1

        plot_single_step(nodes_pos_list[t],
                         dynamic_sbm_series[t],
                         dynamic_sbm_series[t + 1],
                         dynamic_sbm_series[t + 1][2])


def plot_dynamic_sbm_embedding_v2(nodes_pos_list, dynamic_sbm_series):
    # print("dynamic_sbm_series: ", len(dynamic_sbm_series)) 
    # print("nodes_pos_list: ", len(nodes_pos_list)) 
    length = len(dynamic_sbm_series) - 1
    # length = len(dynamic_sbm_series)
    node_num, dimension = nodes_pos_list[0].shape

    if dimension > 2:
        print("Embedding dimension greater than 2, using tSNE to reduce it to 2")
        model = TSNE(n_components=2, random_state=42)
        nodes_pos_list = [model.fit_transform(X) for X in nodes_pos_list]

    pos = 1
    for t in range(length):
        # print(t)
        plt.subplot(220 + pos)
        pos += 1

        plot_single_step_v2(nodes_pos_list[t],
                            dynamic_sbm_series[t],
                            dynamic_sbm_series[t + 1],
                            dynamic_sbm_series[t][2],
                            dynamic_sbm_series[t][3])


if __name__ == '__main__':
    pass
    # prefix = 'data/synthetic/dynamic_SBM/community_diminish_%d_%d_%d' % (256, 3, 10)
    # dynamic_sbm_series = graph_util.loadDynamicSBmGraph(prefix, 5)
    # # get the embedding

    # nodes_pos_list = []
    # for i in xrange(len(dynamic_sbm_series)):
    #     static_embedding = lapEig_static.LaplacianEigenmaps(2)
    #     nodes_pos_list.append(static_embedding.learn_embedding(dynamic_sbm_series[i][0]))

    # plot_dynamic_sbm_embedding(nodes_pos_list, dynamic_sbm_series)
    # plt.savefig('result/visualization_sbm.png')
    # plt.show()
