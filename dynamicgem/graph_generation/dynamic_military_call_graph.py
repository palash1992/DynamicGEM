import cPickle as pickle
import numpy as np
import networkx as nx
import pdb
import pandas as pd

from graph_generation import static_military_call_graph


# def resample_edge_for_node(call_graph, node_id):
#     if call_graph._graph is None:
#         call_graph.sample_graph()
#     else:
#         n = call_graph._n_nodes
#         if node_id < call_graph._n_commanders:
#             for i in xrange(call_graph._n_commanders):
#                 if i == node_id:
#                     continue
#                 if call_graph._graph.has_edge(node_id, i):
#                     call_graph._graph.remove_edge(node_id, i)
#                     call_graph._graph.remove_edge(i, node_id)
#                 i_comm = call_graph._commander_subgraph._node_community[i]
#                 node_id_comm = call_graph._commander_subgraph._node_community[node_id]
#                 prob = call_graph._commander_subgraph._B[i_comm, node_id_comm]
#                 if np.random.uniform() <= prob:
#                     call_graph._graph.add_edge(node_id, i)
#                     call_graph._graph.add_edge(i, node_id)
#             for soldier_idx in xrange(call_graph._n_soldiers):
#                 commander_idx  = call_graph._soldier_subraph._node_community[soldier_idx]
#                 if commander_idx == node_id:
#                     if call_graph._graph.has_edge(node_id, soldier_idx+call_graph._n_commanders):
#                         call_graph._graph.remove_edge(node_id, soldier_idx+call_graph._n_commanders)
#                         call_graph._graph.remove_edge(soldier_idx+call_graph._n_commanders, node_id)
#                     if np.random.uniform() <= call_graph._commander_to_soldier_p:
#                         call_graph._graph.add_edge(node_id, soldier_idx+call_graph._n_commanders)
#                         call_graph._graph.add_edge(soldier_idx+call_graph._n_commanders, node_id)
#             for family_idx in xrange(call_graph._n_family_members):
#                 military_idx  = call_graph._family_subraph._node_community[soldier_idx]
#                 if military_idx == node_id:
#                     if call_graph._graph.has_edge(node_id, family_idx+call_graph._n_military):
#                         call_graph._graph.remove_edge(node_id, family_idx+call_graph._n_military)
#                         call_graph._graph.remove_edge(family_idx+call_graph._n_military, node_id)
#                     if np.random.uniform() <= call_graph._commander_to_soldier_p:
#                         call_graph._graph.add_edge(node_id, family_idx+call_graph._n_military)
#                         call_graph._graph.add_edge(family_idx+call_graph._n_military, node_id)


#         for i in xrange(n):
#             if i == node_id:
#                 continue
#             if call_graph._graph.has_edge(node_id, i):
#                 call_graph._graph.remove_edge(node_id, i)
#                 call_graph._graph.remove_edge(i, node_id)
#             prob = call_graph._B[call_graph._node_community[node_id], call_graph._node_community[i]]
#             if np.random.uniform() <= prob:
#                 call_graph._graph.add_edge(node_id, i)
#                 call_graph._graph.add_edge(i, node_id)

# def resample_nodes(call_graph, k):
#     ''' Resample edges for k randomly selected nodes
#     '''
#     sampled_nodes = np.random.choice(call_graph._n_nodes, k)
#     for i in sampled_nodes:
#        resample_edge_for_node(call_graph, node_id) 

def _resample_subgraph(call_graph, sampled_edge_indices, subgraph, idx, idx2, startIdx, n_subgraph):
    for i in range(n_subgraph):
        for j in range(i):
            if idx == sampled_edge_indices[idx2]:
                if call_graph._graph.has_edge(startIdx + i, startIdx + j):
                    call_graph._graph.remove_edge(startIdx + i, startIdx + j)
                    call_graph._graph.remove_edge(startIdx + j, startIdx + i)
                i_comm = subgraph._node_community[i]
                j_comm = subgraph._node_community[j]
                prob = subgraph._B[i_comm, j_comm]
                if np.random.uniform() <= prob:
                    call_graph._graph.add_edge(startIdx + i, startIdx + j)
                    call_graph._graph.add_edge(startIdx + j, startIdx + i)
                idx2 += 1
            idx += 1
    return call_graph, idx, idx2


def resample_edges(call_graph, percent_resampled_edges):
    n = call_graph._n_nodes
    n_max_edges = (n * (n - 1)) // 2
    k = n_max_edges * percent_resampled_edges // 100
    sampled_edge_indices = sorted(np.random.choice(n_max_edges, k))
    idx = 0
    idx2 = 0
    call_graph, idx, idx2 = _resample_subgraph(call_graph, sampled_edge_indices, call_graph._commander_subgraph, idx,
                                               idx2, 0, call_graph._n_commanders)
    call_graph, idx, idx2 = _resample_subgraph(call_graph, sampled_edge_indices, call_graph._soldier_subgraph, idx,
                                               idx2, call_graph._n_commanders, call_graph._n_soldiers)
    call_graph, idx, idx2 = _resample_subgraph(call_graph, sampled_edge_indices, call_graph._family_subgraph, idx, idx2,
                                               call_graph._n_military, call_graph._n_family_members)
    for soldier_idx in range(call_graph._n_soldiers):
        if idx == sampled_edge_indices[idx2]:
            commander_idx = call_graph._soldier_subgraph._node_community[soldier_idx]
            if call_graph._graph.has_edge(commander_idx, call_graph._n_commanders + soldier_idx):
                call_graph._graph.remove_edge(commander_idx, call_graph._n_commanders + soldier_idx)
                call_graph._graph.remove_edge(call_graph._n_commanders + soldier_idx, commander_idx)
            if np.random.uniform() <= call_graph._commander_to_soldier_p:
                call_graph.add_edge(commander_idx, call_graph._n_commanders + soldier_idx)
                call_graph.add_edge(call_graph._n_commanders + soldier_idx, commander_idx)
            idx2 += 1
        idx += 1

    for family_idx in range(call_graph._n_family_members):
        if idx == sampled_edge_indices[idx2]:
            military_idx = call_graph._family_subgraph._node_community[family_idx]
            if call_graph._graph.has_edge(military_idx, call_graph._n_military + family_idx):
                call_graph.remove_edge(military_idx, call_graph._n_military + family_idx)
                call_graph.remove_edge(call_graph._n_military + family_idx, military_idx)
            if np.random.uniform() <= call_graph._to_family_p:
                call_graph.add_edge(military_idx, call_graph._n_military + family_idx)
                call_graph.add_edge(call_graph._n_military + family_idx, military_idx)
            idx2 += 1
        idx += 1


def increment_commander_commander_calls(call_graph, inc_factor):
    n_commanders = call_graph._n_commanders
    p = inc_factor * call_graph._crossblock_commander_p
    for i in range(n_commanders):
        for j in range(i):
            if np.random.uniform() <= p:
                call_graph._graph.add_edge(i, j)
                call_graph._graph.add_edge(j, i)


def increment_commander_soldier_calls(call_graph, inc_factor):
    n_commanders = call_graph._n_commanders
    n_soldiers = call_graph._n_soldiers
    p = inc_factor * call_graph._commander_to_soldier_p
    for i in range(n_soldiers):
        commander_idx = call_graph._soldier_subgraph._node_community[i]
        if np.random.uniform() <= p:
            call_graph._graph.add_edge(i + n_commanders, commander_idx)
            call_graph._graph.add_edge(commander_idx, i + n_commanders)


def increment_soldier_soldier_calls(call_graph, inc_factor):
    n_commanders = call_graph._n_commanders
    n_soldiers = call_graph._n_soldiers
    p = inc_factor * call_graph._crossblock_soldier_p
    for i in range(n_commanders, n_commanders + n_soldiers):
        for j in range(n_commanders, i):
            if np.random.uniform() <= p:
                call_graph._graph.add_edge(i, j)
                call_graph._graph.add_edge(j, i)


def get_dynamic_military_call_graph(percent_resampled_edges, inc_factor, n_peace, n_plan, n_order, n_event):
    # Start time series with Phase I - Peace
    # n_soldiers, n_commanders, n_commander_comms, n_family_mem_per_person, inblock_prob, crossblock_prob, commander_to_soldier_p, to_family_p
    graph_current = static_military_call_graph.StaticMilitaryGraph(1000, 100, 10, 1, [0.2] * 3, [0.01] * 3, 0.1,
                                                                   0.1)  # Fill it with initial param mitras
    graph_current.sample_graph()
    graph_series = [graph_current._graph.copy()]

    # PHASE I - PEACE
    # n_peace_timesteps = np.random.randint(n_peace//2, n_peace)
    for t in range(n_peace):
        resample_edges(graph_current, percent_resampled_edges)
        graph_series.append(graph_current._graph.copy())

    # PHASE II - PLANNING
    # n_planning_timesteps = np.random.randint(n_plan//2, n_plan)
    for t in range(n_plan):
        resample_edges(graph_current, percent_resampled_edges)
        increment_commander_commander_calls(graph_current, inc_factor)
        graph_series.append(graph_current._graph.copy())

    # PHASE III - ORDER
    # n_order_timesteps = np.random.randint(n_order//2, n_order)
    for t in range(n_order):
        resample_edges(graph_current, percent_resampled_edges)
        increment_commander_soldier_calls(graph_current, inc_factor)
        graph_series.append(graph_current._graph.copy())

    # PHASE IV - EVENT
    # n_event_timesteps = np.random.randint(n_event//2, n_event)
    for t in range(n_event):
        resample_edges(graph_current, percent_resampled_edges)
        increment_soldier_soldier_calls(graph_current, inc_factor)
        graph_series.append(graph_current._graph.copy())

    return graph_series


if __name__ == "__main__":
    military_call_graph_series = get_dynamic_military_call_graph(10, 2, 10, 3, 3, 3)
    n_edges_arr = np.zeros(len(military_call_graph_series))
    for idx, curr_graph in enumerate(military_call_graph_series):
        n_edges_arr[idx] = curr_graph.number_of_edges()
    n_calls_df = pd.DataFrame(n_edges_arr, columns=['Daily Call Count'])
    ax = n_calls_df.plot(title='Call plot')
    fig = ax.get_figure()
    fig.savefig('callPlot.png')
    print(n_edges_arr)
    pdb.set_trace()
