import matplotlib.pyplot as plt
import numpy as np
import random
import networkx as nx
import operator
import sys
sys.path.append('./')
# from graph_generation import SBM_graph
from utils import graph_util
import SBM_graph
from matplotlib import rc
import seaborn
import os

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

def dyn_node_chng_v2(sbm_graph, node_id):
      
    if sbm_graph._graph is None:
        sbm_graph.sample_graph()
    else:
        n = sbm_graph._node_num
        othercommnodes =[i for i in range(n) if sbm_graph._node_community[i]!=sbm_graph._node_community[node_id] if not sbm_graph._graph.has_edge(node_id, i)]
        edgesnodes = random.sample(othercommnodes,30)
        for i in edgesnodes:
            sbm_graph._graph.add_edge(node_id, i)
            sbm_graph._graph.add_edge(i, node_id)
        
        
        for i in range(n):
            if i == node_id:
                continue
            if sbm_graph._node_community[i]==sbm_graph._node_community[node_id]:    
                if sbm_graph._graph.has_edge(node_id, i):
                    prob = 0.1
                    if np.random.uniform() <= prob:
                        sbm_graph._graph.remove_edge(node_id, i)
                        sbm_graph._graph.remove_edge(i, node_id)                                            


def diminish_community_v2(sbm_graph, community_id, nodes_to_purturb, chngnodes):
    n = sbm_graph._node_num
    community_nodes = [i for i in range(n) if sbm_graph._node_community[i] == community_id]
    nodes_to_purturb = min(len(community_nodes), nodes_to_purturb)
    
    perturb_nodes=chngnodes

    pos=nx.spring_layout(sbm_graph._graph)
    color=['#FFD700','#4B0082']
    plt.figure()
    plt.subplot(231)
    nodes_draw=nx.draw_networkx_nodes(sbm_graph._graph,pos,node_size=40,node_color=[color[sbm_graph._node_community[p]] for p in sbm_graph._graph.nodes()])
    nx.draw_networkx_edges(sbm_graph._graph,pos,arrows=False,width=0.5,alpha=0.5)
    nodes_draw.set_edgecolor('w')
    nodes_draw=nx.draw_networkx_nodes(sbm_graph._graph,pos, 
                                      nodelist=perturb_nodes, 
                                      node_color='r', 
                                      node_size=50, 
                                      with_labels=False)
    nodes_draw.set_edgecolor('k')
    nx.draw_networkx_labels(sbm_graph._graph,pos,font_size=8)
    
    

    left_communitis = [i for i in range(sbm_graph._community_num) if i != community_id]
    for node_id in perturb_nodes:
        new_community = random.sample(left_communitis, 1)[0]
        print ('Node %d change from community %d to %d' % (node_id,
                                                          sbm_graph._node_community[node_id], 
                                                          new_community))
        sbm_graph._node_community[node_id] = new_community
    for node_id in perturb_nodes:
        _resample_egde_for_node(sbm_graph, node_id)

    # pos=nx.spring_layout(sbm_graph._graph)
    plt.subplot(232)
    nodes_draw=nx.draw_networkx_nodes(sbm_graph._graph,pos,node_size=40,node_color=[color[sbm_graph._node_community[p]] for p in sbm_graph._graph.nodes()])
    nx.draw_networkx_edges(sbm_graph._graph,pos,arrows=False,width=0.5,alpha=0.5)  
    nx.draw_networkx_labels(sbm_graph._graph,pos,font_size=8)  
    nodes_draw.set_edgecolor('w')
    nodes_draw=nx.draw_networkx_nodes(sbm_graph._graph,pos, 
                                      nodelist=perturb_nodes, 
                                      node_color='r', 
                                      node_size=50, 
                                      with_labels=False)
    nodes_draw.set_edgecolor('k')

    nodes=[i for i in range(n) if sbm_graph._node_community[i] == community_id ]

    chngnodes = random.sample(nodes, nodes_to_purturb)
    for node_id in chngnodes:
        dyn_node_chng_v2(sbm_graph, node_id)

    print("Changed Nodes: ",chngnodes)   
    plt.subplot(233)
    nodes_draw=nx.draw_networkx_nodes(sbm_graph._graph,pos,node_size=40,node_color=[color[sbm_graph._node_community[p]] for p in sbm_graph._graph.nodes()])
    nx.draw_networkx_edges(sbm_graph._graph,pos,arrows=False,width=0.5,alpha=0.5) 
    nx.draw_networkx_labels(sbm_graph._graph,pos,font_size=8)
    nodes_draw.set_edgecolor('w')
    nodes_draw=nx.draw_networkx_nodes(sbm_graph._graph,pos,
                                      nodelist=chngnodes, 
                                      node_color='r', 
                                      node_size=50, 
                                      with_labels=False)
    nodes_draw.set_edgecolor('k')
    
    if not os.path.exists('./figures'):
        os.mkdir('./figures')
    count=[ files for files in os.listdir('./figures') if '.pdf' in files]    
    plt.savefig('./figures/motivation'+str(len(count)+1)+'.pdf',bbox_inches='tight',dpi=600)
    plt.show()
    return perturb_nodes, chngnodes



def get_community_diminish_series_v2(node_num, 
                                  community_num, 
                                  length, 
                                  community_id, 
                                  nodes_to_purturb,
                                  ):

    my_graph = SBM_graph.SBMGraph(node_num, community_num,community_id,nodes_to_purturb)
    my_graph.sample_graph_v3()
    chngnodes=my_graph._chngnodes

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
    node_change_num = 2
    length = 2
    get_community_diminish_series_v2(node_num, 
                                  community_num, 
                                  length, 
                                  1, 
                                  node_change_num)
    plt.show()
    # drawGraph(node_num, community_num)
    
    # prefix = 'data/synthetic/dynamic_SBM/node_pertuabtion_%d_%d_%d' % (node_num, community_num, node_change_num)
    # dynamic_sbm_series = get_random_perturbation_series(node_num, community_num, length, node_change_num)
    # graph_util.saveDynamicSBmGraph(prefix, dynamic_sbm_series)

    # prefix = 'data/synthetic/dynamic_SBM/community_diminish_%d_%d_%d' % (node_num, community_num, node_change_num)
    # dynamic_sbm_series = get_community_diminish_series(node_num, community_num, length, 1, node_change_num)
    # graph_util.saveDynamicSBmGraph(prefix, dynamic_sbm_series)

