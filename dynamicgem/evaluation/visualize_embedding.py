import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE

import sys
sys.path.insert(0, './')
from dynamicgem.utils import plot_util

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


def plot_embedding2D(node_pos, node_colors=None, di_graph=None):
    node_num, embedding_dimension = node_pos.shape
    if(embedding_dimension > 2):
        print("Embedding dimension greater than 2, use tSNE to reduce it to 2")
        model = TSNE(n_components=2)
        node_pos = model.fit_transform(node_pos)

    if di_graph is None:
        # plot using plt scatter
        plt.scatter(node_pos[:, 0], node_pos[:, 1], c=node_colors)
    else:
        # plot using networkx with edge structure
        pos = {}
        for i in range(node_num):
            pos[i] = node_pos[i, :]
        if node_colors:
            nodes_draw=nx.draw_networkx_nodes(di_graph, pos,
                                   node_color=node_colors,
                                   width=0.1, node_size=40,
                                   arrows=False, alpha=0.8,
                                   font_size=5)
            nodes_draw.set_edgecolor('w')
        else:
            nodes_draw=nx.draw_networkx(di_graph, pos, node_color=node_colors,
                             width=0.1, node_size=40, arrows=False,
                             alpha=0.8, font_size=12)
            nodes_draw.set_edgecolor('w')


def expVis(X, res_pre, m_summ, node_labels=None, di_graph=None):
    print('\tGraph Visualization:')
    if node_labels:
        node_colors = plot_util.get_node_color(node_labels)
    else:
        node_colors = None
    plot_embedding2D(X, node_colors=node_colors,
                     di_graph=di_graph)
    plt.savefig('%s_%s_vis.pdf' % (res_pre, m_summ), dpi=300,
                format='pdf', bbox_inches='tight')
    plt.figure()

def plot_single_step(node_pos, graph_info,  dyn_changed_node):
    node_colors= plot_util.get_node_color(graph_info[1])
    node_num, embedding_dimension = node_pos.shape
    pos = {}
    for i in range(node_num):
        pos[i] = node_pos[i, :]
    unchanged_nodes = list(set(range(node_num)) - set(dyn_changed_node))

    nodes_draw=nx.draw_networkx_nodes(graph_info[0], 
                           pos, 
                           nodelist=unchanged_nodes, 
                           node_color=[node_colors[p] for p in unchanged_nodes], 
                           node_size=40, 
                           with_labels=False)
    nodes_draw.set_edgecolor('w')

    nodes_draw=nx.draw_networkx_nodes(graph_info[0], 
                                      pos, 
                                      nodelist=dyn_changed_node, 
                                      node_color='r', 
                                      node_size=80, 
                                      with_labels=False)
#     nodes_draw.set_edgecolor('k')

def plot_static_sbm_embedding(nodes_pos_list, dynamic_sbm_series):
    length = len(dynamic_sbm_series)
    node_num, dimension = nodes_pos_list[0].shape

    if dimension > 2:
        print("Embedding dimension greater than 2, using tSNE to reduce it to 2")
        model = TSNE(n_components=2, random_state=42)
        nodes_pos_list = [model.fit_transform(X) for X in nodes_pos_list]

    pos = 1
    plt.figure()
    for t in range(length):
        plt.subplot(220 + pos)
        pos += 1

        plot_single_step(nodes_pos_list[t], 
                         dynamic_sbm_series[t], 
                         dynamic_sbm_series[t][3]) 
    plt.show()    
     
