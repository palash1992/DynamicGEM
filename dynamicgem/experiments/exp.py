try: import cPickle as pickle
except: import pickle
from time import time
from argparse import ArgumentParser
import importlib
import json
import cPickle
import networkx as nx
import itertools
import pdb
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, './')

from dynamicgem.graph_generation import dynamic_SBM_graph
from dynamicgem.utils      import graph_util, plot_util
from dynamicgem.evaluation.evaluate_graph_reconstruction import expGR
from dynamicgem.evaluation.evaluate_link_prediction import expLP

methClassMap = {"dynAE": "DynAE",
                "dynAERNN": "DynAERNN",
                "dynRNN": "DynRNN",
                "rand": "RandDynamic",
                }
expMap = {"gf": "GF MAP", "lp": "LP MAP",
          "nc": "NC MAP"}


def learn_emb(MethObj, graphs, params, res_pre, m_summ):
    if params["experiments"] == ["lp"]:
        X = None
    else:
        print 'Learning Embedding: %s' % m_summ
        if not bool(int(params["load_emb"])):
            X, learn_t = MethObj.learn_embeddings(graphs)
            print '\tTime to learn embedding: %f sec' % learn_t
            pickle.dump(
                X,
                open('%s_%s_%d.emb' % (res_pre, m_summ, len(graphs)), 'wb')
            )
            pickle.dump(learn_t,
                        open('%s_%s_%d.learnT' % (res_pre, m_summ, len(graphs)), 'wb'))
        else:
            X = pickle.load(
                open('%s_%s_%d.emb' % (res_pre, m_summ, len(graphs)),
                                 'rb')
                )
            try:
                learn_t = pickle.load(
                    open('%s_%s_%d.learnT' % (res_pre, m_summ, len(graphs)),
                                           'rb')
                )
                print '\tTime to learn emb.: %f sec' % learn_t
            except IOError:
                print '\tTime info not found'
    return X


def run_exps(MethObj, meth, dim, graphs, data_set, params):
    m_summ = '%s_%d' % (meth, dim)
    res_pre = "results/%s" % data_set
    n_r = params["rounds"]
    T = len(graphs)
    X = [None] * (T - T // 2)
    for t in range(T // 2, T):
        X[t - T // 2] = learn_emb(
            MethObj, graphs[:t], params, res_pre, m_summ
        )
    gr, lp = [0] * (T - T // 2), [0] * (T - T // 2)
    if "gr" in params["experiments"]:
        for t in range(T // 2, T):
            gr[t - T // 2] = expGR(
                graphs[t], MethObj,
                X[t - T // 2], params["n_sample_nodes"],
                n_r, res_pre,
                m_summ, file_suffix=data_set+'_'+str(dim),is_undirected=params["is_undirected"],
                sampling_scheme=params["samp_scheme"]
            )

    if "lp" in params["experiments"]:
        lp = expLP(graphs, MethObj,
                   n_r, res_pre,
                   m_summ, params["n_sample_nodes"],
                   is_undirected=params["is_undirected"],
                   sampling_scheme=params["samp_scheme"])
    return gr, lp


def get_max(val, val_max, idx, idx_max):
    if val > val_max:
        return val, idx
    else:
        return val_max, idx_max


def choose_best_hyp(data_set, graphs, params):
    # Load range of hyper parameters to test on
    try:
        model_hyp_range = json.load(
            open('experiments/config/%s_hypRange2.conf' % data_set, 'r')
        )
    except IOError:
        model_hyp_range = json.load(
            open('experiments/config/default_hypRange.conf', 'r')
        )

    # Test each hyperparameter for each method and store the best
    for meth in params["methods"]:
        dim = 128
        MethClass = getattr(
            importlib.import_module("embedding.%s" % meth),
            methClassMap[meth]
        )
        meth_hyp_range = model_hyp_range[meth]
        gr_max, lp_max = 0, 0
        gr_hyp, lp_hyp = 0, 0
        gr_hyp, lp_hyp = {meth: {}}, {meth: {}}

        # Test each hyperparameter
        ev_cols = ["GR MAP", "LP MAP"]
        hyp_df = pd.DataFrame(
            columns=meth_hyp_range.keys() + ev_cols + ["Round Id"]
        )
        hyp_r_idx = 0
        for hyp in itertools.product(*meth_hyp_range.values()):
            hyp_d = {"d": dim}
            hyp_d.update(dict(zip(meth_hyp_range.keys(), hyp)))
            print hyp_d
            if meth == "dynAE" or meth == "dynRNN" or meth == "dynAERNN":
                hyp_d.update({
                    "modelfile": [
                        "./intermediate/encoder_model_%s_%d.json" % (data_set, dim),
                        "./intermediate/decoder_model_%s_%d.json" % (data_set, dim)
                    ],
                    "weightfile": [
                        "./intermediate/encoder_weights_%s_%d.hdf5" % (data_set, dim),
                        "./intermediate/decoder_weights_%s_%d.hdf5" % (data_set, dim)
                    ]
                })
            # elif meth == "gf" or meth == "node2vec":
            #     hyp_d.update({"data_set": data_set})
            MethObj = MethClass(hyp_d)
            gr, lp = run_exps(MethObj, meth, dim, graphs,
                                  data_set, params)
            gr_m, lp_m = np.mean(gr), np.mean(lp)
            gr_max, gr_hyp[meth] = get_max(gr_m, gr_max, hyp_d, gr_hyp[meth])
            lp_max, lp_hyp[meth] = get_max(lp_m, lp_max, hyp_d, lp_hyp[meth])
            hyp_df_row = dict(zip(meth_hyp_range.keys(), hyp))
            for r_id in range(params["rounds"]):
                hyp_df.loc[hyp_r_idx, meth_hyp_range.keys()] = \
                    pd.Series(hyp_df_row)
                # hyp_df.loc[hyp_r_idx, ev_cols + ["Round Id"]] = \
                #     [gr[min(r_id, len(gr) -1)], lp[r_id], r_id]
                hyp_df.loc[hyp_r_idx, ev_cols + ["Round Id"]] = \
                    [np.mean(np.array(gr)), np.mean(np.array(lp)), r_id]
                hyp_r_idx += 1
        exp_param = params["experiments"]
        for exp in exp_param:
            hyp_df.to_hdf(
                "intermediate/%s_%s_%s_%s_hyp.h5" % (data_set, meth,
                                                         exp,
                                                         params["samp_scheme"]),
                "df"
            )
        plot_util.plot_hyp(meth_hyp_range.keys(), exp_param,
                           meth, data_set, s_sch=params["samp_scheme"])

        # Store the best hyperparameter
        opt_hyp_f_pre = 'experiments/config/%s_%s_%s' % (
            data_set,
            meth,
            params["samp_scheme"]
        )
        if gr_max:
            with open('%s_gr.conf' % opt_hyp_f_pre, 'w') as f:
                f.write(json.dumps(gr_hyp, indent=4))
        if lp_max:
            with open('%s_lp.conf' % opt_hyp_f_pre, 'w') as f:
                f.write(json.dumps(lp_hyp, indent=4))


def call_plot_hyp(data_set, params):
    # Load range of hyper parameters tested on to plot
    try:
        model_hyp_range = json.load(
            open('experiments/config/%s_hypRange.conf' % data_set, 'r')
        )
    except IOError:
        model_hyp_range = json.load(
            open('experiments/config/default_hypRange.conf', 'r')
        )
    for meth in params["methods"]:
            meth_hyp_range = model_hyp_range[meth]
            exp_param = params["experiments"]
            plot_util.plot_hyp(meth_hyp_range.keys(), exp_param,
                               meth, data_set,
                               s_sch=params["samp_scheme"])


def call_plot_hyp_all(data_sets, params):
    # Load range of hyper parameters tested on to plot
    try:
        model_hyp_range = json.load(
            open('experiments/config/%s_hypRange.conf' % data_sets[0], 'r')
        )
    except IOError:
        model_hyp_range = json.load(
            open('experiments/config/default_hypRange.conf', 'r')
        )
    for meth in params["methods"]:
            meth_hyp_range = model_hyp_range[meth]
            exp_param = params["experiments"]
            plot_util.plot_hyp_all(meth_hyp_range.keys(), exp_param,
                                   meth, data_sets,
                                   s_sch=params["samp_scheme"])


def call_exps(params, data_set, n_graphs):

    # Load Dataset
    print('Dataset: %s' % data_set)
    if data_set == "sbm":
        node_num = 500
        community_num = 2
        node_change_num = 5
        length = n_graphs
        sbm_gs = dynamic_SBM_graph.get_community_diminish_series_v2(
            node_num,
            community_num,
            length,
            1,
            node_change_num
        )
        graphs = [g[0] for g in sbm_gs]
    else:
        graphs = []
        for t in range(n_graphs):
            G = nx.read_gpickle(
                'data/%s/graph_%d.gpickle' % (data_set, t)
            )
            G, nodeListMap = graph_util.get_lcc(G)
            graphs.append(G)
            print('Graph %d:' % t)
            graph_util.print_graph_stats(G)

    # Search through the hyperparameter space
    if params["find_hyp"]:
        choose_best_hyp(data_set, graphs, params)

    # Load best hyperparameter and test it again on new test data
    for d, meth, exp in itertools.product(
        params["dimensions"],
        params["methods"],
        params["experiments"]
    ):
        dim = int(d)
        MethClass = getattr(
            importlib.import_module("embedding.%s" % meth),
            methClassMap[meth]
        )
        opt_hyp_f_pre = 'experiments/config/%s_%s_%s' % (
            data_set,
            meth,
            params["samp_scheme"]
        )
        try:
            model_hyp = json.load(
                open('%s_%s.conf' % (opt_hyp_f_pre, exp), 'r')
            )
        except IOError:
            print('Default hyperparameter of the method chosen')
            model_hyp = json.load(
                open('experiments/config/%s.conf' % meth, 'r')
            )
        hyp = {}
        hyp.update(model_hyp[meth])
        hyp.update({"d": dim})
        if meth == "dynAE" or meth == "dynAERNN" or meth == "dynRNN":
                hyp.update({
                    "modelfile": [
                        "./intermediate/encoder_model_%s_%d.json" % (data_set, dim),
                        "./intermediate/decoder_model_%s_%d.json" % (data_set, dim)
                    ],
                    "weightfile": [
                        "./intermediate/encoder_weights_%s_%d.hdf5" % (data_set, dim),
                        "./intermediate/decoder_weights_%s_%d.hdf5" % (data_set, dim)
                    ]
                })
        elif meth == "gf" or meth == "node2vec":
            hyp.update({"data_set": data_set})
        MethObj = MethClass(hyp)
        run_exps(MethObj, meth, dim, graphs, data_set, params)


if __name__ == '__main__':
    ''' Sample usage
    python experiments/exp.py -data sbm -dim 128 -meth sdne -exp gr,lp
    '''
    t1 = time()
    parser = ArgumentParser(description='Graph Embedding Experiments')
    parser.add_argument('-data', '--data_sets',
                        help='dataset names (default: sbm)')
    parser.add_argument('-dim', '--dimensions',
                        help='embedding dimensions list(default: 2^1 to 2^8)')
    parser.add_argument('-meth', '--methods',
                        help='method list (default: all methods)')
    parser.add_argument('-exp', '--experiments',
                        help='exp list (default: gr,lp)')
    parser.add_argument('-lemb', '--load_emb',
                        help='load saved embeddings (default: False)')
    parser.add_argument('-lexp', '--load_exp',
                        help='load saved experiment results (default: False)')
    parser.add_argument('-rounds', '--rounds',
                        help='number of rounds (default: 5)')
    parser.add_argument('-plot', '--plot',
                        help='plot the results (default: True)')
    parser.add_argument('-plot_d', '--plot_d',
                        help='plot the results wrt dims(default: True)')
    parser.add_argument('-hyp_plot', '--hyp_plot',
                        help='plot the hyperparameter results (default: True)')
    parser.add_argument('-hyp_plot_all', '--hyp_plot_all',
                        help='plot the hyperparameter results (all) (default: True)')
    parser.add_argument('-find_hyp', '--find_hyp',
                        help='find best hyperparameters (default: False)')
    parser.add_argument('-saveMAP', '--save_MAP',
                        help='save MAP in a latex table (default: False)')
    parser.add_argument('-n_samples', '--n_sample_nodes',
                        help='number of sampled nodes (default: 1024)')
    parser.add_argument('-s_sch', '--samp_scheme',
                        help='sampling scheme (default: u_rand)')
    parser.add_argument('-n_graphs', '--n_graphs',
                        help='# of graphs (default: 5)')

    params = json.load(open('experiments/config/params.conf', 'r'))
    args = vars(parser.parse_args())
    print args
    for k, v in args.iteritems():
        if v is not None:
            params[k] = v
    params["experiments"] = params["experiments"].split(',')
    params["data_sets"] = params["data_sets"].split(',')
    params["rounds"] = int(params["rounds"])
    params["n_sample_nodes"] = int(params["n_sample_nodes"])
    params["is_undirected"] = bool(int(params["is_undirected"]))
    params["plot_d"] = bool(int(params["plot_d"]))
    params["plot"] = bool(int(params["plot"]))
    params["hyp_plot"] = bool(int(params["hyp_plot"]))
    params["hyp_plot_all"] = bool(int(params["hyp_plot_all"]))
    t_pred = int(params["n_graphs"]) - int(params["n_graphs"]) // 2
    if params["methods"] == "all":
        params["methods"] = methClassMap.keys()
    else:
        params["methods"] = params["methods"].split(',')
    params["dimensions"] = params["dimensions"].split(',')
    print params
    for data_set in params["data_sets"]:
        if not int(params["load_exp"]):
            call_exps(params, data_set, int(params["n_graphs"]))
        if int(params["plot"]):
            res_pre = "results/%s" % data_set
            plot_util.plotExpRes(res_pre, params["methods"],
                                 params["experiments"], params["dimensions"],
                                 'plots/%s_%s' % (data_set, params["samp_scheme"]),
                                 params["rounds"], params["plot_d"], t_pred,
                                 params["samp_scheme"])
        if int(params["hyp_plot"]):
            call_plot_hyp(data_set, params)
    if int(params["hyp_plot_all"]):
            call_plot_hyp_all(params["data_sets"], params)
