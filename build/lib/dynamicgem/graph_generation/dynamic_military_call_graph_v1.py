import cPickle as pickle
import numpy as np
import networkx as nx
import pdb
import pandas as pd

import static_military_call_graph

def resample_edge_for_node(call_graph, node_id):
    if call_graph._graph is None:
        call_graph.sample_graph()
    else:
        n = call_graph._n_nodes
        for i in xrange(n):
            if i == node_id:
                continue
            if call_graph._graph.has_edge(node_id, i):
                call_graph._graph.remove_edge(node_id, i)
                call_graph._graph.remove_edge(i, node_id)
            prob = call_graph._B[call_graph._node_community[node_id], call_graph._node_community[i]]
            if np.random.uniform() <= prob:
                call_graph._graph.add_edge(node_id, i)
                call_graph._graph.add_edge(i, node_id)

def resample_nodes(call_graph, k):
    ''' Resample edges for k randomly selected nodes
    '''
    sampled_nodes = np.random.choice(call_graph._n_nodes, k)
    for i in sampled_nodes:
       resample_edge_for_node(call_graph, node_id) 

def resample_edges(call_graph, percent_resampled_edges):
    ''' Resample k random edges
    '''
    n_edges = call_graph._graph.number_of_edges()
    k = n_edges*percent_resampled_edges/100.0
    sampled_edge_indices = np.random.choice(n_edges, int(k))

    idx = 0
    idx2 = 0
    call_graph_copy = call_graph._graph.copy()
    for v_i, v_j in call_graph_copy.edges_iter():
        if idx == sampled_edge_indices[idx2]:
            prob = call_graph._B[call_graph._node_community[v_i], call_graph._node_community[v_j]]
            if np.random.uniform() > prob:
                call_graph._graph.remove_edge(v_i, v_j)
                call_graph._graph.remove_edge(v_j, v_i)
            else:
                call_graph._graph.add_edge(v_i, v_j)
                call_graph._graph.add_edge(v_j, v_i)
            idx2 += 1
        idx += 1

def increment_commander_commander_calls(call_graph, inc_factor):
    n_commanders = call_graph._n_commanders
    p = inc_factor*call_graph._crossblock_commander_p
    for i in xrange(n_commanders):
        for j in xrange(i):
            if np.random.uniform() <= p:
                call_graph._graph.add_edge(i, j)
                call_graph._graph.add_edge(j, i)

def increment_commander_soldier_calls(call_graph, inc_factor):
    n_commanders = call_graph._n_commanders
    n_soldiers = call_graph._n_soldiers
    p = inc_factor*call_graph._commander_to_soldier_p
    for i in xrange(n_commanders):
        for j in xrange(n_commanders, n_commanders + n_soldiers):
            if np.random.uniform() <= p:
                call_graph._graph.add_edge(i, j)
                call_graph._graph.add_edge(j, i)

def increment_soldier_soldier_calls(call_graph, inc_factor):
    n_commanders = call_graph._n_commanders
    n_soldiers = call_graph._n_soldiers
    p = inc_factor*call_graph._crossblock_soldier_p
    for i in xrange(n_commanders, n_commanders+n_soldiers):
        for j in xrange(n_commanders, i):
            if np.random.uniform() <= p:
                call_graph._graph.add_edge(i, j)
                call_graph._graph.add_edge(j, i)


def get_dynamic_military_call_graph(percent_resampled_edges, inc_factor, n_peace, n_plan, n_order, n_event):
    # Start time series with Phase I - Peace
    # n_soldiers, n_commanders, n_commander_comms, n_family_mem_per_person, inblock_prob, crossblock_prob, commander_to_soldier_p, to_family_p
    graph_current = static_military_call_graph.StaticMilitaryGraph(1000, 100, 10, 1, [0.1]*3, [0.01]*3, 0.1, 0.1)# Fill it with initial param mitras
    graph_current.sample_graph()
    graph_series = [graph_current._graph.copy()]

    # PHASE I - PEACE
    n_peace_timesteps = np.random.randint(n_peace//2, n_peace)
    for t in xrange(n_peace):
        resample_edges(graph_current, percent_resampled_edges)
        graph_series.append(graph_current._graph.copy())

    # PHASE II - PLANNING
    n_planning_timesteps = np.random.randint(n_plan//2, n_plan)
    for t in xrange(n_plan):
        resample_edges(graph_current, percent_resampled_edges)
        increment_commander_commander_calls(graph_current, inc_factor)
        graph_series.append(graph_current._graph.copy())

    # PHASE III - ORDER
    n_order_timesteps = np.random.randint(n_order//2, n_order)
    for t in xrange(n_order):
        resample_edges(graph_current, percent_resampled_edges)
        increment_commander_soldier_calls(graph_current, inc_factor)
        graph_series.append(graph_current._graph.copy())

    # PHASE IV - EVENT
    n_event_timesteps = np.random.randint(n_event//2, n_event)
    for t in xrange(n_event):
        resample_edges(graph_current, percent_resampled_edges)
        increment_soldier_soldier_calls(graph_current, inc_factor)
        graph_series.append(graph_current._graph.copy())

    return graph_series

if __name__ == "__main__":
    military_call_graph_series = get_dynamic_military_call_graph(1, 2, 10, 3, 3, 3)
    n_edges_arr = np.zeros(len(military_call_graph_series))
    for idx, curr_graph in enumerate(military_call_graph_series):
        n_edges_arr[idx] = curr_graph.number_of_edges()
    n_calls_df = pd.DataFrame(n_edges_arr, columns=['Daily Call Count'])
    ax = n_calls_df.plot(title='Call plot')
    fig = ax.get_figure()
    fig.savefig('callPlot.png')
    pdb.set_trace()
