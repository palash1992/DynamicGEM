try: import cPickle as pickle
except: import pickle
from .metrics import *
from dynamicgem.utils import evaluation_util, graph_util
import networkx as nx
import numpy as np


def evaluateStaticGraphReconstruction(digraph, 
                                      graph_embedding,
                                      X_stat, 
                                      node_l=None, 
                                      sample_ratio_e=None, 
                                      file_suffix=None,
                                      is_undirected=True,
                                      is_weighted=False):
    node_num = digraph.number_of_nodes()
    # evaluation
    if sample_ratio_e:
        eval_edge_pairs = evaluation_util.getRandomEdgePairs(
            node_num,
            sample_ratio_e,
            is_undirected
        )
    else:
        eval_edge_pairs = None
    if file_suffix is None:
        estimated_adj = graph_embedding.get_reconstructed_adj(X_stat, node_l)
    else:
        estimated_adj = graph_embedding.get_reconstructed_adj(
            X_stat,
            node_l,
            file_suffix
            
        )
    predicted_edge_list = evaluation_util.getEdgeListFromAdjMtx(
        estimated_adj,
        is_undirected=is_undirected,
        edge_pairs=eval_edge_pairs
    )
    MAP = metrics.computeMAP(predicted_edge_list, digraph)
    prec_curv, _ = metrics.computePrecisionCurve(predicted_edge_list, digraph)
    # If weighted, compute the error in reconstructed weights of observed edges
    if is_weighted:
        digraph_adj = nx.to_numpy_matrix(digraph)
        estimated_adj[digraph_adj == 0] = 0
        err = np.linalg.norm(digraph_adj - estimated_adj)
        err_baseline = np.linalg.norm(digraph_adj)
    else:
        err = None
        err_baseline = None
    return (MAP, prec_curv, err, err_baseline)


def expGR(digraph, 
          graph_embedding,
          X, 
          n_sampled_nodes, 
          rounds,
          res_pre, 
          m_summ,
          file_suffix=None,
          is_undirected=True,
          sampling_scheme="rw"):
    print('\tGraph Reconstruction')
    n_sampled_nodes = int(n_sampled_nodes)
    summ_file = open('%s_%s.grsumm' % (res_pre, m_summ), 'w')
    summ_file.write('Method\t%s\n' % metrics.getMetricsHeader())
    if digraph.number_of_nodes() <= n_sampled_nodes:
        rounds = 1
    MAP = [None] * rounds
    prec_curv = [None] * rounds
    err = [None] * rounds
    err_b = [None] * rounds
    n_nodes = [None] * rounds
    n_edges = [None] * rounds
    for round_id in range(rounds):
        if sampling_scheme == "u_rand":
            sampled_digraph, node_l = graph_util.sample_graph(
                digraph,
                n_sampled_nodes=n_sampled_nodes
            )
        else:
            sampled_digraph, node_l = graph_util.sample_graph_rw_int(
                digraph,
                n_sampled_nodes=n_sampled_nodes
            )
        n_nodes[round_id] = sampled_digraph.number_of_nodes()
        n_edges[round_id] = sampled_digraph.number_of_edges()
        print('\t\tRound: %d, n_nodes: %d, n_edges:%d\n' % (round_id,
                                                            n_nodes[round_id],
                                                            n_edges[round_id]))
        sampled_X = X[node_l]
        # sampled_X = np.expand_dims(sampled_X,axis=1)
        MAP[round_id], prec_curv[round_id], err[round_id], err_b[round_id] = \
            evaluateStaticGraphReconstruction(sampled_digraph, 
                                              graph_embedding,
                                              sampled_X, 
                                              node_l,
                                              file_suffix= file_suffix,
                                              is_undirected=is_undirected
                                              )
    try:
        summ_file.write('Err: %f/%f\n' % (np.mean(err), np.std(err)))
        summ_file.write('Err_b: %f/%f\n' % (np.mean(err_b), np.std(err_b)))
    except TypeError:
        pass
    summ_file.write('%f/%f\t%s\n' % (np.mean(MAP), np.std(MAP),
                                     metrics.getPrecisionReport(prec_curv[0],
                                                                n_edges[0])))
    pickle.dump([n_nodes,
                 n_edges,
                 MAP,
                 prec_curv,
                 err,
                 err_b],
                open('%s_%s.gr' % (res_pre, m_summ), 'wb'))
    return np.mean(np.array(MAP))
