try: import cPickle as pickle
except: import pickle
from .metrics import *
from dynamicgem.utils import evaluation_util
from dynamicgem.utils import graph_util
import numpy as np
import networkx as nx
import pdb
import sys
sys.path.insert(0, './')
from dynamicgem.utils import embed_util


def evaluateDynamicLinkPrediction(graph, 
                                  embedding,
                                 rounds,
                                 n_sample_nodes=None,
                                 no_python=False,
                                 is_undirected=True,
                                 sampling_scheme="u_rand"):
    node_l = None
    if n_sample_nodes:
        if sampling_scheme == "u_rand":
            test_digraph, node_l = graph_util.sample_graph(
                graph,
                n_sample_nodes
            )
        else:
            test_digraph, node_l = graph_util.sample_graph_rw_int(
                graph,
                n_sample_nodes
            )
    estimated_adj = embedding.predict_next_adj(node_l)
    print(len(estimated_adj),np.shape(estimated_adj))

    predicted_edge_list = evaluation_util.getEdgeListFromAdjMtx(
        estimated_adj,
        is_undirected=is_undirected,
        edge_pairs=None
    )
    print(len(predicted_edge_list), np.shape(predicted_edge_list) ,len(test_digraph.edges()),np.shape(test_digraph.edges()))
    # pdb.set_trace()

    MAP = metrics.computeMAP(predicted_edge_list, test_digraph)
    prec_curv, _ = metrics.computePrecisionCurve(
        predicted_edge_list,
        test_digraph
    )
    return (MAP, prec_curv)

def evaluateDynamicLinkPrediction_TIMERS(graph, 
                                  embedding,t,
                                 rounds,
                                 n_sample_nodes=None,
                                 no_python=False,
                                 is_undirected=True,
                                 sampling_scheme="u_rand"):
    node_l = None
    if n_sample_nodes:
        if sampling_scheme == "u_rand":
            test_digraph, node_l = graph_util.sample_graph(
                graph,
                n_sample_nodes
            )
        else:
            test_digraph, node_l = graph_util.sample_graph_rw_int(
                graph,
                n_sample_nodes
            )
    estimated_adj = embedding.predict_next_adj(t,node_l)

    predicted_edge_list = evaluation_util.getEdgeListFromAdjMtx(
        estimated_adj,
        is_undirected=is_undirected,
        edge_pairs=None
    )

    MAP = metrics.computeMAP(predicted_edge_list, test_digraph)
    prec_curv, _ = metrics.computePrecisionCurve(
        predicted_edge_list,
        test_digraph
    )
    return (MAP, prec_curv)


def expLP(graphs, 
          embedding, 
          rounds,
          res_pre, 
          m_summ,
          n_sample_nodes=1000, 
          train_ratio_init=0.5,
          no_python=False, 
          is_undirected=True,
          sampling_scheme="u_rand"):
    n_sample_nodes = int(n_sample_nodes)
    print('\tDynamic Link Prediction')
    summ_file = open('%s%s.dlpsumm' % (res_pre, m_summ), 'w')
    summ_file.write('Method\t%s\n' % metrics.getMetricsHeader())
    summ_file.close()
    T = len(graphs)
    T_min = int(train_ratio_init * T)
    MAP = [None] * (T-T_min)
    prec_curv = [None] * (T-T_min)
    for i in range(T - T_min):
        MAP[i] = [None] * rounds
        prec_curv[i] = [None] * rounds
    for t in range(T_min, T):
        embedding.learn_embeddings(graphs[:t])
        for r_id in range(rounds):
            MAP[t-T_min][r_id], prec_curv[t-T_min][r_id] = \
                evaluateDynamicLinkPrediction(graphs[t], embedding,
                                             rounds,
                                             n_sample_nodes=n_sample_nodes,
                                             no_python=no_python,
                                             is_undirected=is_undirected,
                                             sampling_scheme=sampling_scheme)
        summ_file = open('%s%s.dlpsumm' % (res_pre, m_summ), 'a')
        summ_file.write('\tt=%d%f/%f\t%s\n' % (
            t - T_min,
            np.mean(MAP[t-T_min]),
            np.std(MAP[t-T_min]),
            metrics.getPrecisionReport(
                prec_curv[t-T_min][0],
                len(prec_curv[t-T_min][0])
            )
        ))
        summ_file.close()
    # pickle.dump([MAP, prec_curv],
    #             open('%s_%s_%s.lp' % (res_pre, m_summ, sampling_scheme),
    #                  'wb'))
    return np.mean(np.array(MAP))

def exp_changedLP(graphs, 
          embedding, 
          rounds,
          res_pre, 
          m_summ,
          n_sample_nodes=1000, 
          train_ratio_init=0.5,
          no_python=False, 
          is_undirected=True,
          sampling_scheme="u_rand"):
    n_sample_nodes = int(n_sample_nodes)
    print('\tDynamic Link Prediction')
    summ_file = open('%s%s.dlpsumm' % (res_pre, m_summ), 'w')
    summ_file.write('Method\t%s\n' % metrics.getMetricsHeader())
    summ_file.close()
    T = len(graphs)
    T_min = int(train_ratio_init * T)
    MAP = [None] * (T-T_min)
    prec_curv = [None] * (T-T_min)
    for i in range(T - T_min):
        MAP[i] = [None] * rounds
        prec_curv[i] = [None] * rounds
    for t in range(T_min, T):
        edges_add,edges_rm = getchangedlinks(graphs[t-1],graphs[t])
        embedding.learn_embeddings(graphs[:t])
        for r_id in range(rounds):
            MAP[t-T_min][r_id], prec_curv[t-T_min][r_id] = \
                evaluateDynamic_changed_LinkPrediction(graphs[t], embedding,
                                             rounds,
                                             edges_add,edges_rm,
                                             # dynamic_sbm_series[t][3],
                                             n_sample_nodes=n_sample_nodes,
                                             no_python=no_python,
                                             is_undirected=is_undirected,
                                             sampling_scheme=sampling_scheme)
        summ_file = open('%s%s.dlpsumm' % (res_pre, m_summ), 'a')
        summ_file.write('\tt=%d%f/%f\t%s\n' % (
            t - T_min,
            np.mean(MAP[t-T_min]),
            np.std(MAP[t-T_min]),
            metrics.getPrecisionReport(
                prec_curv[t-T_min][0],
                len(prec_curv[t-T_min][0])
            )
        ))
        summ_file.close()
    # pickle.dump([MAP, prec_curv],
    #             open('%s_%s_%s.lp' % (res_pre, m_summ, sampling_scheme),
    #                  'wb'))
    return np.mean(np.array(MAP))


def evaluateDynamic_changed_LinkPrediction(graph, 
                                  embedding,
                                 rounds,
                                 edges_add,edges_rm,
                                 n_sample_nodes=None,
                                 no_python=False,
                                 is_undirected=True,
                                 sampling_scheme="u_rand"):
    nodes=[]
    for e in edges_add[0]:
      nodes.append(e[0])
      nodes.append(e[1])

    # for e in edges_rm[0]:
    #   nodes.append(e[0])
    #   nodes.append(e[1])  

    nodes=list(np.unique(nodes))  
    # pdb.set_trace()

    test_digraph, node_l =  graph_util.sample_graph(graph,  len(nodes), nodes)
    estimated_adj = embedding.predict_next_adj(node_l)

    predicted_edge_list = evaluation_util.getEdgeListFromAdjMtx(
        estimated_adj,
        is_undirected=is_undirected,
        edge_pairs=None
    )

    MAP = metrics.computeMAP(predicted_edge_list, test_digraph)
    prec_curv, _ = metrics.computePrecisionCurve(
        predicted_edge_list,
        test_digraph
    )

    return (MAP, prec_curv)



def evaluateDynamic_changed_LinkPrediction_v2(graph, 
                                  embedding,
                                 rounds,
                                 edges_add,edges_rm,
                                 n_sample_nodes=None,
                                 no_python=False,
                                 is_undirected=True,
                                 sampling_scheme="u_rand"):
    nodes=[]
    for e in edges_add[0]:
      nodes.append(e[0])
      nodes.append(e[1])

    # for e in edges_rm[0]:
    #   nodes.append(e[0])
    #   nodes.append(e[1])  

    nodes=list(np.unique(nodes))  
    # pdb.set_trace()

    test_digraph, node_dict =  graph_util.sample_graph_nodes(graph,  nodes)
    estimated_adj = embedding.predict_next_adj(node_l)

    predicted_edge_list = evaluation_util.getEdgeListFromAdjMtx(
        estimated_adj,
        is_undirected=is_undirected,
        edge_pairs=None
    )

    MAP = metrics.computeMAP(predicted_edge_list, test_digraph,node_dict, edges_rm)
    
    node_edges_rm = []
    for i in range(len(edges_rm[0])):
        node_edges_rm.append([])
    for st, ed in edges_rm[0]:
        node_edges_rm[node_dict[st]].append((node_dict[st], node_dict[ed], 1))  
    node_edges_rm=[node_edges_rm[i] for i in xrange(len(node_edges_rm)) if len(node_edges_rm[i])>0]
     
    # pdb.set_trace()
    prec_curv, _ = metrics.computePrecisionCurve(
        predicted_edge_list,
        test_digraph,node_edges_rm
    )
    # pdb.set_trace()

    return (MAP, prec_curv)


def getchangedlinks(G,Gnew):
  #get all the changed links
  edges_add=[]
  Gdiff=nx.difference(Gnew,G)
  edges_add.append(Gdiff.edges())

  edges_rm=[]
  Gdiff=nx.difference(G,Gnew)
  edges_rm.append(Gdiff.edges())

  # pdb.set_trace()

  # for e in edges:
  #   nodes.append(e[0])
  #   nodes.append(e[1])


  return edges_add, edges_rm

def expstatic_changedLP(dynamic_sbm_series,
          graphs, 
          embedding, 
          rounds,
          res_pre, 
          m_summ,
          n_sample_nodes=1000, 
          train_ratio_init=0.5,
          no_python=False, 
          is_undirected=True,
          sampling_scheme="u_rand"):
    n_sample_nodes = int(n_sample_nodes)
    print('\tDynamic Link Prediction')
    summ_file = open('%s%s.dlpsumm' % (res_pre, m_summ), 'w')
    summ_file.write('Method\t%s\n' % metrics.getMetricsHeader())
    summ_file.close()
    T = len(graphs)
    # T_min = int(train_ratio_init * T)
    MAP = [None] * (T-1)
    prec_curv = [None] * (T-1)
    for i in range(T - 1):
        MAP[i] = [None] * rounds
        prec_curv[i] = [None] * rounds
    for t in range(T-1):
        embedding.learn_embeddings(graphs[t])
        edges_add,edges_rm = getchangedlinks(graphs[t],graphs[t+1])
        for r_id in range(rounds):
            MAP[t][r_id], prec_curv[t][r_id] = \
                evaluateDynamic_changed_LinkPrediction(graphs[t+1], embedding,
                                             rounds,
                                             edges_add,edges_rm,
                                             # dynamic_sbm_series[t][3],
                                             n_sample_nodes=n_sample_nodes,
                                             no_python=no_python,
                                             is_undirected=is_undirected,
                                             sampling_scheme=sampling_scheme)
        summ_file = open('%s%s.dlpsumm' % (res_pre, m_summ), 'a')
        summ_file.write('\tt=%d%f/%f\t%s\n' % (
            t,
            np.mean(MAP[t]),
            np.std(MAP[t]),
            metrics.getPrecisionReport(
                prec_curv[t][0],
                len(prec_curv[t][0])
            )
        ))
        summ_file.close()
    # pickle.dump([MAP, prec_curv],
    #             open('%s_%s_%s.lp' % (res_pre, m_summ, sampling_scheme),
    #                  'wb'))
    return np.mean(np.array(MAP))

def expstaticLP(dynamic_sbm_series,
          graphs, 
          embedding, 
          rounds,
          res_pre, 
          m_summ,
          n_sample_nodes=1000, 
          train_ratio_init=0.5,
          no_python=False, 
          is_undirected=True,
          sampling_scheme="u_rand"):
    n_sample_nodes = int(n_sample_nodes)
    print('\tDynamic Link Prediction')
    summ_file = open('%s%s.dlpsumm' % (res_pre, m_summ), 'w')
    summ_file.write('Method\t%s\n' % metrics.getMetricsHeader())
    summ_file.close()
    T = len(graphs)
    # T_min = int(train_ratio_init * T)
    MAP = [None] * (T-1)
    prec_curv = [None] * (T-1)
    for i in range(T - 1):
        MAP[i] = [None] * rounds
        prec_curv[i] = [None] * rounds
    for t in range(T-1):
        embedding.learn_embeddings(graphs[t])
        for r_id in range(rounds):
            MAP[t][r_id], prec_curv[t][r_id] = \
                evaluateDynamicLinkPrediction(graphs[t+1], embedding,
                                             rounds,
                                             # dynamic_sbm_series[t][3],
                                             n_sample_nodes=n_sample_nodes,
                                             no_python=no_python,
                                             is_undirected=is_undirected,
                                             sampling_scheme=sampling_scheme)
        summ_file = open('%s%s.dlpsumm' % (res_pre, m_summ), 'a')
        summ_file.write('\tt=%d%f/%f\t%s\n' % (
            t,
            np.mean(MAP[t]),
            np.std(MAP[t]),
            metrics.getPrecisionReport(
                prec_curv[t][0],
                len(prec_curv[t][0])
            )
        ))
        summ_file.close()
    # pickle.dump([MAP, prec_curv],
    #             open('%s_%s_%s.lp' % (res_pre, m_summ, sampling_scheme),
    #                  'wb'))
    return np.mean(np.array(MAP))


def expstaticLP_TIMERS(dynamic_sbm_series,
          graphs, 
          embedding, 
          rounds,
          res_pre, 
          m_summ,
          n_sample_nodes=1000, 
          train_ratio_init=0.5,
          no_python=False, 
          is_undirected=True,
          sampling_scheme="u_rand"):
    n_sample_nodes = int(n_sample_nodes)
    print('\tDynamic Link Prediction')
    summ_file = open('%s%s.dlpsumm' % (res_pre, m_summ), 'w')
    summ_file.write('Method\t%s\n' % metrics.getMetricsHeader())
    summ_file.close()
    T = len(graphs)
    # T_min = int(train_ratio_init * T)
    MAP = [None] * (T-1)
    prec_curv = [None] * (T-1)
    for i in range(T - 1):
        MAP[i] = [None] * rounds
        prec_curv[i] = [None] * rounds
    for t in range(T-1):
        # embedding.learn_embeddings(t)
        for r_id in range(rounds):
            MAP[t][r_id], prec_curv[t][r_id] = \
                evaluateDynamicLinkPrediction_TIMERS(graphs[t+1], embedding,t,
                                             rounds,
                                             # dynamic_sbm_series[t][3],
                                             n_sample_nodes=n_sample_nodes,
                                             no_python=no_python,
                                             is_undirected=is_undirected,
                                             sampling_scheme=sampling_scheme)
        summ_file = open('%s%s.dlpsumm' % (res_pre, m_summ), 'a')
        summ_file.write('\tt=%d%f/%f\t%s\n' % (
            t,
            np.mean(MAP[t]),
            np.std(MAP[t]),
            metrics.getPrecisionReport(
                prec_curv[t][0],
                len(prec_curv[t][0])
            )
        ))
        summ_file.close()
    # pickle.dump([MAP, prec_curv],
    #             open('%s_%s_%s.lp' % (res_pre, m_summ, sampling_scheme),
    #                  'wb'))
    return np.mean(np.array(MAP))

def expstaticLP_TRIAD(dynamic_sbm_series,
          graphs, 
          embedding, 
          rounds,
          res_pre, 
          m_summ,
          n_sample_nodes=1000, 
          train_ratio_init=0.5,
          no_python=False, 
          is_undirected=True,
          sampling_scheme="u_rand"):
    n_sample_nodes = int(n_sample_nodes)
    print('\tDynamic Link Prediction')
    summ_file = open('%s%s.dlpsumm' % (res_pre, m_summ), 'w')
    summ_file.write('Method\t%s\n' % metrics.getMetricsHeader())
    summ_file.close()
    T = len(graphs)
    # T_min = int(train_ratio_init * T)
    MAP = [None] * (T-1)
    prec_curv = [None] * (T-1)
    for i in range(T - 1):
        MAP[i] = [None] * rounds
        prec_curv[i] = [None] * rounds
    for t in range(T-1):
        
        embedding.link_predict(graphs[t],t) 
        for r_id in range(rounds):
            MAP[t][r_id], prec_curv[t][r_id] = \
                evaluateDynamicLinkPrediction_TIMERS(graphs[t+1], embedding,t,
                                             rounds,
                                             # dynamic_sbm_series[t][3],
                                             n_sample_nodes=n_sample_nodes,
                                             no_python=no_python,
                                             is_undirected=is_undirected,
                                             sampling_scheme=sampling_scheme)
        summ_file = open('%s%s.dlpsumm' % (res_pre, m_summ), 'a')
        summ_file.write('\tt=%d%f/%f\t%s\n' % (
            t,
            np.mean(MAP[t]),
            np.std(MAP[t]),
            metrics.getPrecisionReport(
                prec_curv[t][0],
                len(prec_curv[t][0])
            )
        ))
        summ_file.close()
    # pickle.dump([MAP, prec_curv],
    #             open('%s_%s_%s.lp' % (res_pre, m_summ, sampling_scheme),
    #                  'wb'))
    return np.mean(np.array(MAP))    