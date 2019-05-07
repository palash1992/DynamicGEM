try: import cPickle as pickle
except: import pickle
import os
if os.name == 'posix' and 'DISPLAY' not in os.environ:
    import matplotlib
    matplotlib.use('Agg')
import matplotlib    
import matplotlib.pyplot as plt
import itertools
from matplotlib import rc
import random
import seaborn
import numpy as np
import pandas as pd
import pdb
from argparse import ArgumentParser

font = {'family': 'serif', 'serif': ['computer modern roman']}
rc('text', usetex=False)
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

figsize_d = {2: (5, 2),
             4: (9, 2)}

m_name_l = {"dynAE": "DynAE",
                "dynRNN": "DynRNN",
                "rand": "RandDynamic",
                }

expMap = {"gr": "GR MAP", "lp": "LP MAP",
          "nc": "NC F1 score"}
expMap2 = {"gr": "GR MAP", "lp": "LP P@100",
           "nc": "NC F1 score"}


def get_node_color(node_community):
    cnames = [item[0] for item in matplotlib.colors.cnames.items()]
    node_colors = [cnames[c] for c in node_community]
    return node_colors


def plot(x_s, y_s, fig_n, x_lab, y_lab,
         file_save_path, title, legendLabels=None, show=False):
    plt.rcParams.update({'font.size': 16, 'font.weight': 'bold'})
    markers = ['o', '*', 'v', 'D', '<', 's', '+', '^', '>']
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    series = []
    plt.figure(fig_n)
    i = 0
    for i in range(len(x_s)):
        # n_points = len(x_s[i])
        # n_points = int(n_points/10) + random.randint(1,100)
        # x = x_s[i][::n_points]
        # y = y_s[i][::n_points]
        x = x_s[i]
        y = y_s[i]
        series.append(plt.plot(x, y, color=colors[i],
                               linewidth=2, marker=markers[i],
                               markersize=8))
        plt.xlabel(x_lab, fontsize=16, fontweight='bold')
        plt.ylabel(y_lab, fontsize=16, fontweight='bold')
        plt.title(title, fontsize=16, fontweight='bold')
    if legendLabels:
        plt.legend([s[0] for s in series], legendLabels)
    plt.savefig(file_save_path)
    if show:
        plt.show()


def plot_ts(ts_df, plot_title, eventDates,
            eventLabels=None, save_file_name=None,
            xLabel=None, yLabel=None, show=False):
    ax = ts_df.plot(title=plot_title, marker='*',
                    markerfacecolor='red', markersize=10,
                    linestyle='solid')
    colors = ['r', 'g', 'c', 'm', 'y', 'b', 'k']
    if not eventLabels:
        for eventDate in eventDates:
            # Show event as a red vertical line
            ax.axvline(eventDate, color='r', linestyle='--', lw=2)
    else:
        for idx in range(len(eventDates)):
            ax.axvline(eventDates[idx], color=colors[idx],
                       linestyle='--', lw=2, label=eventLabels[idx])
            ax.legend()
    if xLabel:
        ax.set_xlabel(xLabel, fontweight='bold')
    if yLabel:
        ax.set_ylabel(yLabel, fontweight='bold')
    fig = ax.get_figure()
    if save_file_name:
        fig.savefig(save_file_name, bbox_inches='tight')
    if show:
        fig.show()


def turn_latex(key_str):
    if key_str in ['mu', 'rho', 'beta', 'alpha', 'gamma']:
        return '$\%s$' % key_str
    else:
        return '$%s$' % key_str.upper()


def plot_hyp_data2(hyp_keys, exp_param,
                   meths, data,
                   s_sch="u_rand",
                   dim=2):
    font = {'family': 'serif', 'serif': ['computer modern roman']}
    rc('text', usetex=True)
    rc('font', weight='bold')
    rc('font', size=8)
    rc('lines', markersize=2.5)
    rc('lines', linewidth=0.5)
    rc('xtick', labelsize=6)
    rc('ytick', labelsize=6)
    rc('axes', labelsize='small')
    rc('axes', labelweight='bold')
    rc('axes', titlesize='small')
    rc('axes', linewidth=1)
    plt.rc('font', **font)
    seaborn.set_style("darkgrid")
    for exp in exp_param:
        df_all = pd.DataFrame()
        n_meths = 0
        for meth in meths:
            try:
                df = pd.read_hdf(
                    "intermediate/%s_%s_%s_%s_dim_%d_data_hyp.h5" % (data, meth, exp, s_sch, dim),
                    "df"
                )
                n_meths += 1
            except:
                print ('%s_%s_%s_%s_dim_%d_data_hyp.h5 not found. Ignoring data set' % (data, meth, exp, s_sch, dim))
                continue
            # Check if experiment is in the dataframe
            if expMap[exp] not in df:
                continue
            df["Method"] = m_name_l[meth]
            # pdb.set_trace()
            df_all = df_all.append(df).reset_index()
            df_all = df_all.drop(['index'], axis=1)
        if df_all.empty:
            continue
        df = df_all
        col_names = df.columns
        col_rename_d = {}
        for col_name in col_names:
            col_rename_d[col_name] = col_name.replace('_', '\ ')
        df.rename(columns=col_rename_d, inplace=True)
        for hyp_key in hyp_keys:
            # hyp_key_ren = hyp_key.replace('_', '\ ')
            df_trun = df[hyp_keys + ["Round Id", expMap[exp], expMap2[exp], "Method"]]
            df_grouped = df_trun
            rem_hyp_keys = list(set(hyp_keys) - {hyp_key})
            val_lists = [df_grouped[r_k].unique() for r_k in rem_hyp_keys]
            n_cols = len(list(itertools.product(*val_lists)))
            if len(df_grouped[hyp_key].unique()) < 3:
                continue
            plot_shape = (1, n_cols)
            fin1, axarray1 = plt.subplots(1, n_cols, figsize=figsize_d[n_cols])
            fin2, axarray2 = plt.subplots(1, n_cols, figsize=figsize_d[n_cols])
            for plt_idx, hyp_vals in enumerate(itertools.product(*val_lists)):
                plot_idx = np.unravel_index(plt_idx, plot_shape)
                hyp_dict = dict(zip(rem_hyp_keys, hyp_vals))
                hyp_str = ', '.join(
                    "%s:%r" % (turn_latex(key), val) for (key, val) in hyp_dict.iteritems() if len(df_grouped[key].unique()) > 1
                )
                df_temp = df_grouped
                for hyp_idx, hyp_val in enumerate(hyp_vals):
                    df_temp = df_temp[df_temp[rem_hyp_keys[hyp_idx]] == hyp_val]
                if len(df_temp[hyp_key].unique()) < 3:
                    continue
                print('Plotting %s: %s' % (exp, hyp_key))
                try:
                    ax = seaborn.tsplot(time=hyp_key, value=expMap[exp],
                                        unit="Round Id", condition="Method",
                                        data=df_temp,
                                        ax=axarray1[plot_idx[0], plot_idx[1]])
                    if plot_idx[1]:
                        ax.set_ylabel('')
                    if not plot_idx[0]:
                        ax.set_xlabel('')
                except IndexError:
                    try:
                        ax = seaborn.tsplot(time=hyp_key, value=expMap[exp],
                                            unit="Round Id", condition="Method",
                                            data=df_temp,
                                            ax=axarray1[plt_idx])
                    except:
                        import pdb
                        pdb.set_trace()
                    if plt_idx:
                        ax.set_ylabel('')
                ax.set_title(hyp_str)
                hyp_values = df_grouped[hyp_key].unique()
                l_diff = hyp_values[-1] - hyp_values[-2]
                f_diff = hyp_values[1] - hyp_values[0]
                l_f_diff_r = l_diff / f_diff
                if l_f_diff_r > 1:
                    log_base = pow(l_f_diff_r, 1.0 / (len(hyp_values) - 2))
                    ax.set_xscale('log', basex=round(log_base))
                marker = ["o", "s", "D", "^", "v", "8", "*", "p", "1", "h"]
                for line_i in range(len(ax.lines)):
                    ax.lines[line_i].set_marker(marker[line_i])
                # ax.grid()
                ax.legend_.remove()

                try:
                    ax = seaborn.tsplot(time=hyp_key, value=expMap2[exp],
                                        unit="Round Id", condition="Method",
                                        data=df_temp,
                                        ax=axarray2[plot_idx[0], plot_idx[1]])
                    if plot_idx[1]:
                        ax.set_ylabel('')
                    if not plot_idx[0]:
                        ax.set_xlabel('')
                except IndexError:
                    ax = seaborn.tsplot(time=hyp_key, value=expMap2[exp],
                                        unit="Round Id", condition="Method",
                                        data=df_temp,
                                        ax=axarray2[plt_idx])
                    if plt_idx:
                        ax.set_ylabel('')
                ax.set_title(hyp_str)
                if l_f_diff_r > 1:
                    log_base = pow(l_f_diff_r, 1.0 / (len(hyp_values) - 2))
                    ax.set_xscale('log', basex=round(log_base))
                marker = ["o", "s", "D", "^", "v", "8", "*", "p", "1", "h"]
                for line_i in range(len(ax.lines)):
                    ax.lines[line_i].set_marker(marker[line_i])
                # ax.grid()
                ax.legend_.remove()
            for col_idx in range(axarray1.shape[0]):
                box = axarray1[col_idx].get_position()
                axarray1[col_idx].set_position(
                    [box.x0,
                     box.y0 + box.height * 0.1,
                     box.width,
                     box.height * 0.9]
                )
                box = axarray2[col_idx].get_position()
                axarray2[col_idx].set_position(
                    [box.x0,
                     box.y0 + box.height * 0.1,
                     box.width,
                     box.height * 0.9]
                )
            fin1.legend(loc='lower center', bbox_to_anchor=(0.45, -0.01),
                        ncol=n_meths, fancybox=True, shadow=True)
            fin2.legend(loc='lower center', bbox_to_anchor=(0.45, -0.01),
                        ncol=n_meths, fancybox=True, shadow=True)
            fin1.savefig(
                'plots/data_hyp/%s_%s_%s_%d_%s.pdf' % (data, exp, s_sch, dim, hyp_key),
                dpi=300, format='pdf', bbox_inches='tight'
            )
            fin2.savefig(
                'plots/data_hyp/%s_%s_%s_%d_%s_p100.pdf' % (data, exp, s_sch, dim, hyp_key),
                dpi=300, format='pdf', bbox_inches='tight'
            )
            fin1.clf()
            fin2.clf()


def plot_hyp_data(hyp_keys, exp_param,
                  meths, data,
                  s_sch="u_rand",
                  dim=2):
    for exp in exp_param:
        df_all = pd.DataFrame()
        for meth in meths:
            try:
                df = pd.read_hdf(
                    "intermediate/%s_%s_%s_%s_dim_%d_data_hyp.h5" % (data, meth, exp, s_sch, dim),
                    "df"
                )
            except:
                print('%s_%s_%s_%s_dim_%d_data_hyp.h5 not found. Ignoring data set' % (data, meth, exp, s_sch, dim))
                continue
            # Check if experiment is in the dataframe
            if expMap[exp] not in df:
                continue
            df["Method"] = m_name_l[meth]
            # pdb.set_trace()
            df_all = df_all.append(df).reset_index()
            df_all = df_all.drop(['index'], axis=1)
        if df_all.empty:
            continue
        df = df_all
        col_names = df.columns
        col_rename_d = {}
        for col_name in col_names:
            col_rename_d[col_name] = col_name.replace('_', '\ ')
        df.rename(columns=col_rename_d, inplace=True)
        for hyp_key in hyp_keys:
            # hyp_key_ren = hyp_key.replace('_', '\ ')
            df_trun = df[hyp_keys + ["Round Id", expMap[exp], expMap2[exp], "Method"]]
            df_grouped = df_trun
            rem_hyp_keys = list(set(hyp_keys) - {hyp_key})
            val_lists = [df_grouped[r_k].unique() for r_k in rem_hyp_keys]
            for hyp_vals in itertools.product(*val_lists):
                hyp_dict = dict(zip(rem_hyp_keys, hyp_vals))
                hyp_str = '_'.join("%s=%r" % (key,val) for (key,val) in hyp_dict.iteritems())
            
                df_temp = df_grouped
                for hyp_idx, hyp_val in enumerate(hyp_vals):
                    df_temp = df_temp[df_temp[rem_hyp_keys[hyp_idx]] == hyp_val]
                if len(df_temp[hyp_key].unique()) < 3:
                    continue
                print('Plotting %s: %s' % (exp, hyp_key))
                ax = seaborn.tsplot(time=hyp_key, value=expMap[exp],
                                    unit="Round Id", condition="Method",
                                    data=df_temp)
                hyp_values = df_grouped[hyp_key].unique()
                l_diff = hyp_values[-1] - hyp_values[-2]
                f_diff = hyp_values[1] - hyp_values[0]
                l_f_diff_r = l_diff / f_diff
                if l_f_diff_r > 1:
                    log_base = pow(l_f_diff_r, 1.0 / (len(hyp_values) - 2))
                    ax.set_xscale('log', basex=round(log_base))
                marker = ["o", "s", "D", "^", "v", "8", "*", "p", "1", "h"]
                for line_i in range(len(ax.lines)):
                    ax.lines[line_i].set_marker(marker[line_i])
                # ax.grid()
                ax.legend()
                plt.savefig(
                    'plots/data_hyp/%s_%s_%s_%d_%s.pdf' % (data, exp, s_sch, dim, hyp_str),
                    dpi=300, format='pdf', bbox_inches='tight'
                )
                plt.clf()
                ax = seaborn.tsplot(time=hyp_key, value=expMap2[exp],
                    unit="Round Id", condition="Method",
                    data=df_temp)
                hyp_values = df_grouped[hyp_key].unique()
                l_diff = hyp_values[-1] - hyp_values[-2]
                f_diff = hyp_values[1] - hyp_values[0]
                l_f_diff_r = l_diff / f_diff
                if l_f_diff_r > 1:
                    log_base = pow(l_f_diff_r, 1.0 / (len(hyp_values) - 2))
                    ax.set_xscale('log', basex=round(log_base))
                marker = ["o", "s", "D", "^", "v", "8", "*", "p", "1", "h"]
                for line_i in range(len(ax.lines)):
                    ax.lines[line_i].set_marker(marker[line_i])
                # ax.grid()
                ax.legend()
                plt.savefig(
                    'plots/data_hyp/%s_%s_%s_%d_%s_p_100.pdf' % (data, exp, s_sch, dim, hyp_str),
                    dpi=300, format='pdf', bbox_inches='tight'
                )
                plt.clf()


def plot_hyp(hyp_keys, exp_param, meth, data,
             s_sch="u_rand"):
    for exp in exp_param:
        df = pd.read_hdf(
            "intermediate/%s_%s_%s_%s_hyp.h5" % (data, meth, exp, s_sch),
            "df"
        )
        col_names = df.columns
        col_rename_d = {}
        for col_name in col_names:
            col_rename_d[col_name] = col_name.replace('_', '\ ')
        df.rename(columns=col_rename_d, inplace=True)
        for hyp_key in hyp_keys:
            hyp_key_ren = hyp_key.replace('_', '\ ')
            df_trun = df[[hyp_key_ren, "Round Id", expMap[exp]]]
            try:
                df_grouped = df_trun.groupby([hyp_key_ren, "Round Id"]).max().reset_index()
            except TypeError:
                df_trun[hyp_key_ren + "2"] = \
                    df_trun[hyp_key_ren].apply(lambda x: str(x))
                df_trun[hyp_key_ren] = df_trun[hyp_key_ren + "2"].copy()
                df_trun = df_trun.drop([hyp_key_ren + "2"], axis=1)
                df_grouped = df_trun.groupby([hyp_key_ren, "Round Id"]).max().reset_index()
            if len(df_grouped[hyp_key_ren].unique()) < 3:
                continue
            try:
                print('Plotting %s: %s' % (exp, hyp_key))
                ax = seaborn.tsplot(time=hyp_key_ren, value=expMap[exp],
                                    unit="Round Id", data=df_grouped)
                hyp_values = df_grouped[hyp_key_ren].unique()
                l_diff = hyp_values[-1] - hyp_values[-2]
                f_diff = hyp_values[1] - hyp_values[0]
                l_f_diff_r = l_diff / f_diff
                if l_f_diff_r > 1:
                    log_base = pow(l_f_diff_r, 1.0 / (len(hyp_values) - 2))
                    ax.set_xscale('log', basex=round(log_base))
                marker = ["o", "s", "D", "^", "v", "8", "*", "p", "1", "h"]
                for line_i in range(len(ax.lines)):
                    ax.lines[line_i].set_marker(marker[line_i])
                # ax.grid()
                ax.legend()
            except ValueError:
                ax = seaborn.barplot(x=hyp_key_ren, y=expMap[exp], data=df_grouped)
            except ZeroDivisionError:
                print( 'Only 2 points provided to plot hyperparameters')
                continue
            plt.savefig(
                'plots/hyp/%s_%s_%s_%s_%s.pdf' % (data, meth, exp, s_sch, hyp_key),
                dpi=300, format='pdf', bbox_inches='tight'
            )
            plt.clf()


def plot_hyp_all(hyp_keys, exp_param, meth, data_sets,
                 s_sch="u_rand"):
    for exp in exp_param:
        df_all = pd.DataFrame()
        for data in data_sets:
            try:
                df = pd.read_hdf(
                    "intermediate/%s_%s_%s_%s_hyp.h5" % (data, meth, exp, s_sch),
                    "df"
                )
            except:
                print( '%s_%s_%s_%s_hyp.h5 not found. Ignoring data set' % (data, meth, exp, s_sch))
                continue
            # Check if experiment is in the dataframe
            if expMap[exp] not in df:
                continue
            df["Data"] = data
            # pdb.set_trace()
            df_all = df_all.append(df).reset_index()
            df_all = df_all.drop(['index'], axis=1)
        if df_all.empty:
            continue
        col_names = df_all.columns
        col_rename_d = {}
        for col_name in col_names:
            col_rename_d[col_name] = col_name.replace('_', '\ ')
        df_all.rename(columns=col_rename_d, inplace=True)
        for hyp_key in hyp_keys:
            hyp_key_ren = hyp_key.replace('_', '\ ')
            df_trun = df_all[[hyp_key_ren, "Round Id", expMap[exp], "Data"]]
            try:
                df_grouped = \
                    df_trun.groupby([hyp_key_ren, "Round Id", "Data"]).max().reset_index()
            except TypeError:
                df_trun[hyp_key_ren + "2"] = \
                    df_trun[hyp_key_ren].apply(lambda x: str(x))
                df_trun[hyp_key_ren] = df_trun[hyp_key_ren + "2"].copy()
                df_trun = df_trun.drop([hyp_key_ren + "2"], axis=1)
                df_grouped = df_trun.groupby([hyp_key_ren, "Round Id", "Data"]).max().reset_index()
            if len(df_grouped[df_grouped['Data'] == data_sets[0]][hyp_key_ren].unique()) < 3:
                continue
            try:
                print ('Plotting %s: %s' % (exp, hyp_key))
                if hyp_key_ren == 'inout\ p':
                    hyp_key_ren = 'q'
                elif hyp_key_ren == 'ret\ p':
                    hyp_key_ren = 'p'
                df_grouped.rename(columns={expMap[exp]: m_name_l[meth]}, inplace=True)
                try:
                    df_grouped.rename(columns={'inout\ p': 'q'}, inplace=True)
                except:
                    pass
                try:
                    df_grouped.rename(columns={'ret\ p': 'p'}, inplace=True)
                except:
                    pass
                ax = seaborn.tsplot(time=hyp_key_ren, value=m_name_l[meth],
                                    unit="Round Id", condition="Data",
                                    data=df_grouped)
                hyp_values = df_grouped[hyp_key_ren].unique()
                l_diff = hyp_values[-1] - hyp_values[-2]
                f_diff = hyp_values[1] - hyp_values[0]
                l_f_diff_r = l_diff / f_diff
                if l_f_diff_r > 1:
                    log_base = pow(l_f_diff_r, 1.0 / (len(hyp_values) - 2))
                    ax.set_xscale('log', basex=round(log_base))
                marker = ["o", "s", "D", "^", "v", "8", "*", "p", "1", "h"]
                for line_i in range(len(ax.lines)):
                    ax.lines[line_i].set_marker(marker[line_i])
                # ax.grid()
                ax.legend()
            except ValueError:
                ax = seaborn.barplot(x="Data", y=m_name_l[meth],
                                     hue=hyp_key_ren, data=df_grouped)
            except ZeroDivisionError:
                print('Only 2 points provided to plot hyperparameters')
                continue
            except:
                pdb.set_trace()
            plt.savefig(
                'plots/hyp/%s_%s_%s_%s.pdf' % (meth, exp, s_sch, hyp_key),
                dpi=300, format='pdf', bbox_inches='tight'
            )
            plt.clf()


def plot_p_at_k(res_pre, res_suffix, exp_type, m_names_f,
                m_names, d_arr, n_rounds, save_fig_name, T_pred, length, testData, nm, epochs,
                K=27089, plot_d=False, s_sch="u_rand"): #k=32768
    log_K = int(np.log2(K)) + 1
    num_k = log_K - 3
    df_map = pd.DataFrame(
        np.zeros((n_rounds * len(m_names) * len(d_arr) * T_pred, 5)),
        columns=['d', 'Method', 'Round id', 'MAP', 't']
    )
    df_p_100 = pd.DataFrame(
        np.zeros((n_rounds * len(m_names) * len(d_arr) * T_pred, 5)),
        columns=['d', 'Method', 'Round id', 'P@100', 't']
    )
    df_p_100_idx = 0
    df_map_idx = 0
    MAP = [None] * len(d_arr)
    for d_idx, d in enumerate(d_arr):
        d = int(d)
        df_prec = pd.DataFrame(
            np.zeros((n_rounds * len(m_names) * num_k, 4)),
            columns=['k', 'Method', 'Round id', 'precision@k']
        )
        df_idx = 0
        MAP[d_idx] = [None] * len(m_names_f)
        k_range = [2**i for i in range(3, log_K)]
        p_at_k_ind = [2**i - 1 for i in range(3, log_K)]
        for m_idx, method in enumerate(m_names_f):
            try:
                f = open('%s/%s/%s/nm%d_l%d_emb%d_%s%s' % (res_pre, testData, method, nm, length, d_arr[0],s_sch, res_suffix), 'rb')
            except IOError:
                print('file not found :%s/%s/%s/nm%d_l%d_emb%d_%s%s' % (res_pre, testData, method,  nm, length, d_arr[0],s_sch, res_suffix))
                continue
            if exp_type == 'gr':
                [_, _, MAP[d_idx][m_idx], prec_curv, _, _] = \
                    pickle.load(f)
            else:
                [MAP[d_idx][m_idx], prec_curv] = pickle.load(f)
                try:
                    assert len(MAP[d_idx][m_idx]) >= T_pred-1
                except:
                    # T_pred=len(MAP[d_idx][m_idx])

                    pdb.set_trace()
                    # continue
            
            for rid in range(min(n_rounds, len(prec_curv[0]))):
                for t in range(T_pred-1):
                    try:
                        p_at_k = np.array(prec_curv[t][rid][:K])
                    except:
                        pdb.set_trace()    
                    if p_at_k.shape[0] == 0:
                        print('%s_%s_%d%s: Encountered missing precision curve' \
                              % (res_pre, method, d, res_suffix))
                        continue
                    df_map.loc[df_map_idx, 'd'] = d
                    df_map.loc[df_map_idx, 't'] = t
                    df_map.loc[df_map_idx, 'MAP'] = MAP[d_idx][m_idx][t][rid]
                    df_map.loc[df_map_idx, 'Method'] = m_names[m_idx]
                    df_map.loc[df_map_idx, 'Round id'] = rid
                    df_map_idx += 1

                    df_p_100.loc[df_p_100_idx, 'd'] = d
                    df_p_100.loc[df_p_100_idx, 't'] = t
                    df_p_100.loc[df_p_100_idx, 'P@100'] = p_at_k[100]
                    df_p_100.loc[df_p_100_idx, 'Method'] = m_names[m_idx]
                    df_p_100.loc[df_p_100_idx, 'Round id'] = rid
                    df_p_100_idx += 1

                df_prec.loc[df_idx:df_idx + num_k - 1, 'k'] = k_range
                df_prec.loc[df_idx:df_idx + num_k - 1, 'precision@k'] = \
                    p_at_k[p_at_k_ind]
                df_prec.loc[df_idx:df_idx + num_k - 1, 'Method'] = \
                    m_names[m_idx]
                df_prec.loc[df_idx:df_idx + num_k - 1, 'Round id'] = \
                    rid
                df_idx += num_k
            f.close()
        if d == 32:
            df_prec = df_prec[:df_idx]
            # seaborn.FacetGrid.set(xticks=[2**i for i in range(3, log_K)])
            # ax = seaborn.factorplot(x='k', y='precision@k',
            # hue='Method', units='Round id',
            # data=df_prec)
            # pdb.set_trace()
            plt.figure()
            try:
                ax = seaborn.tsplot(time='k', value='precision@k',
                                    unit='Round id', condition='Method',
                                    data=df_prec)
            except:
                pdb.set_trace()
            # ax.set_xscale('log', basex=2)
            marker = ["o", "s", "D", "^", "v", "8", "*", "p", "1", "h"]
            for line_i in range(len(ax.lines)):
                ax.lines[line_i].set_marker(marker[line_i])
            # ax.grid()
            # ax.legend_.remove()
            plt.show()
            # pdb.set_trace()
            # return
            plt.savefig('%s_d_%d.pdf' % (save_fig_name, d),
                        dpi=300, format='pdf', bbox_inches='tight')
            plt.clf()

            df_map = df_map[:df_map_idx]
            ax = seaborn.tsplot(time='t', value='MAP',
                    unit='Round id', condition='Method',
                    data=df_map)
            # ax = seaborn.barplot(x="Method", y="MAP", data=df_map)
            plt.savefig('%s_d_%d_map.pdf' % (save_fig_name, d),
                        dpi=300, format='pdf', bbox_inches='tight')
            plt.savefig('%s_d_%d_map.png' % (save_fig_name, d),
                        dpi=300, bbox_inches='tight')
            plt.clf()

            df_p_100 = df_p_100[:df_p_100_idx]
            ax = seaborn.tsplot(time='t', value='P@100',
                    unit='Round id', condition='Method',
                    data=df_p_100)
            # ax = seaborn.barplot(x="Method", y="MAP", data=df_p_100)
            plt.savefig('%s_d_%d_p100.pdf' % (save_fig_name, d),
                        dpi=300, format='pdf', bbox_inches='tight')
            plt.savefig('%s_d_%d_p100.png' % (save_fig_name, d),
                        dpi=300, bbox_inches='tight')
            plt.clf()

    if plot_d and len(d_arr) > 1:
        df_map = df_map[:df_map_idx]
        ax = seaborn.tsplot(time='d', value='MAP', unit='Round id',
                            condition='Method', data=df_map)
        ax.set_xscale('log', basex=2)
        marker = ["o", "s", "D", "^", "v", "8", "*", "p", "1", "h"]
        for line_i in range(len(ax.lines)):
            ax.lines[line_i].set_marker(marker[line_i])
        # ax.grid()
        # ax.legend_.remove()
        ax.legend()
        plt.savefig('%s_map.pdf' % save_fig_name,
                    dpi=300, format='pdf', bbox_inches='tight')
        plt.savefig('%s_map.png' % save_fig_name,
                    dpi=300, bbox_inches='tight')
        plt.clf()
        df_p_100 = df_p_100[:df_p_100_idx]
        ax = seaborn.tsplot(time='d', value='P@100', unit='Round id',
                            condition='Method', data=df_p_100)
        ax.set_xscale('log', basex=2)
        marker = ["o", "s", "D", "^", "v", "8", "*", "p", "1", "h"]
        for line_i in range(len(ax.lines)):
            ax.lines[line_i].set_marker(marker[line_i])
        # ax.grid()
        # ax.legend_.remove()
        ax.legend()
        plt.savefig('%s_p_100.pdf' % save_fig_name,
                    dpi=300, format='pdf', bbox_inches='tight')
        plt.savefig('%s_p_100.png' % save_fig_name,
                    dpi=300, bbox_inches='tight')
        plt.clf()
    return MAP


def plot_F1(res_pre, res_suffix, exp_type,
            m_names_f, m_names, d_arr, n_rounds,
            save_fig_name, K=1024, plot_d=False):
    df_f1_glob = pd.DataFrame(
        np.zeros((n_rounds * len(m_names) * len(d_arr), 5)),
        columns=['d', 'Method', 'Round id',
                 'Micro-F1 score', 'Macro-F1 score']
    )
    df_f1_glob_idx = 0
    for d in d_arr:
        d = int(d)
        df = pd.DataFrame(np.zeros((n_rounds * len(m_names) * K, 5)),
                          columns=['Train ratio', 'Method', 'Round id',
                                   'Micro-F1 score', 'Macro-F1 score'])
        df_idx = 0
        for idx, method in enumerate(m_names_f):
            try:
                with open('%s_%s_%d%s' % (res_pre, method, d, res_suffix), 'rb') as f:
                    [test_ratio_arr, micro, macro] = pickle.load(f)
                    n_xlabels = len(test_ratio_arr)
                    for round_id in range(min(n_rounds, len(micro))):
                        microF1 = micro[round_id]
                        macroF1 = macro[round_id]

                        df_f1_glob.loc[df_f1_glob_idx, 'd'] = d
                        df_f1_glob.loc[df_f1_glob_idx, 'Micro-F1 score'] = \
                            microF1[len(test_ratio_arr) // 2]
                        df_f1_glob.loc[df_f1_glob_idx, 'Macro-F1 score'] = \
                            macroF1[len(test_ratio_arr) // 2]
                        df_f1_glob.loc[df_f1_glob_idx, 'Method'] = m_names[idx]
                        df_f1_glob.loc[df_f1_glob_idx, 'Round id'] = round_id
                        df_f1_glob_idx += 1

                        df.loc[df_idx:df_idx + n_xlabels - 1, 'Train ratio'] = \
                            [(1.0 - test_r) for test_r in test_ratio_arr]
                        df.loc[df_idx:df_idx + n_xlabels - 1, 'Micro-F1 score'] = \
                            microF1
                        df.loc[df_idx:df_idx + n_xlabels - 1, 'Macro-F1 score'] = \
                            macroF1
                        df.loc[df_idx:df_idx + n_xlabels - 1, 'Method'] = \
                            m_names[idx]
                        df.loc[df_idx:df_idx + n_xlabels -
                               1, 'Round id'] = round_id
                        df_idx += n_xlabels
            except IOError:
                print('File %s_%s_%d%s not found. Ignoring it for NC plot' \
                    % (res_pre, method, d, res_suffix))
                continue
        if d == 32:
            df = df[:df_idx]
            ax = seaborn.tsplot(time='Train ratio', value='Micro-F1 score',
                                unit='Round id', condition='Method', data=df)
            marker = ["o", "s", "D", "^", "v", "8", "*", "p", "1", "h"]
            for line_i in range(len(ax.lines)):
                ax.lines[line_i].set_marker(marker[line_i])
            # ax.grid()
            ax.legend_.remove()
            plt.savefig('%s_d_%d_micro.pdf' % (save_fig_name, d),
                        dpi=300, format='pdf', bbox_inches='tight')
            plt.show()
            plt.clf()
            ax = seaborn.tsplot(time='Train ratio', value='Macro-F1 score',
                                unit='Round id', condition='Method', data=df)
            for line_i in range(len(ax.lines)):
                ax.lines[line_i].set_marker(marker[line_i])
            # ax.grid()
            ax.legend_.remove()
            plt.savefig('%s_d_%d_macro.pdf' % (save_fig_name, d),
                        dpi=300, format='pdf', bbox_inches='tight')
            plt.show()
            plt.clf()

    if plot_d and len(d_arr) > 1:
        df_f1_glob = df_f1_glob[:df_f1_glob_idx]
        ax = seaborn.tsplot(time='d', value='Micro-F1 score',
                            unit='Round id', condition='Method',
                            data=df_f1_glob)
        ax.set_xscale('log', basex=2)
        marker = ["o", "s", "D", "^", "v", "8", "*", "p", "1", "h"]
        for line_i in range(len(ax.lines)):
            ax.lines[line_i].set_marker(marker[line_i])
        # ax.grid()
        ax.legend_.remove()
        plt.savefig('%s_micro.pdf' % (save_fig_name),
                    dpi=300, format='pdf', bbox_inches='tight')
        plt.clf()
        ax = seaborn.tsplot(time='d', value='Macro-F1 score',
                            unit='Round id', condition='Method',
                            data=df_f1_glob)
        ax.set_xscale('log', basex=2)
        for line_i in range(len(ax.lines)):
            ax.lines[line_i].set_marker(marker[line_i])
        # ax.grid()
        ax.legend_.remove()
        plt.savefig('%s_macro.pdf' % (save_fig_name),
                    dpi=300, format='pdf', bbox_inches='tight')
        plt.clf()


def plotExpRes(res_pre, methods, exp,
               d_arr, save_fig_pre,
               n_rounds, plot_d, T_pred,length,
               testData,nm=None,epochs=None,
               samp_scheme="u_rand", K=1000):
    # m_names = [m_name_l[meth] for meth in methods]
    m_names=methods
    map_gr = None
    map_lp = None
    if "gr" in exp:
        map_gr = plot_p_at_k(res_pre, '.gr', 'gr',
                             methods, m_names, d_arr,
                             n_rounds, '%s_gr' % save_fig_pre, T_pred,
                             K=K, plot_d=plot_d,
                             s_sch=samp_scheme)
    if "lp" in exp:
        map_lp = plot_p_at_k(res_pre, '.lp',
                             'lp', methods, m_names, d_arr,
                             n_rounds, '%s_lp' % save_fig_pre, T_pred, length,testData, nm, epochs,
                             K=K, plot_d=plot_d,
                             s_sch=samp_scheme)
    if "nc" in exp:
        plot_F1(res_pre, '.nc', 'nc', methods, m_names,
                d_arr, n_rounds, '%s_nc' % save_fig_pre,
                K=K, plot_d=plot_d)
    return map_gr, map_lp

if __name__=='__main__':

    parser = ArgumentParser(description='Learns node embeddings for a sequence of graph snapshots')
    parser.add_argument('-t', '--testDataType', default='sbm_cd', type=str,help='Type of data to test the code')
    parser.add_argument('-l', '--timelength', default=7, type=int, help='Number of time series graph to generate')
    parser.add_argument('-nm', '--nodemigration', default=10, type=int, help='number of nodes to migrate')
    parser.add_argument('-emb', '--embeddimension', default=128, type=int, help='embedding dimension')
    parser.add_argument('-rd', '--resultdir', type=str, default='./results_link_all', help="result directory name")
    parser.add_argument('-sm',  '--samples',  default = 5000,type = int, help= 'samples for test data')
    parser.add_argument('-iter', '--epochs', default=250, type=int, help='number of epochs')

    args     = parser.parse_args()

    methods   = ['incrementalSVD','rerunSVD','optimalSVD', 'dynTRIAD', 'staticAE']
    method_d = [ 'dynAE', 'dynRNN', 'dynAERNN']

    testData  = ['sbm_cd']#, 'academic', 'hep', 'AS']
    t_pred   = args.timelength-args.timelength//2

    plotExpRes(args.resultdir, methods,
                                 'lp', [args.embeddimension],
                                 'plots/%s' % (args.testDataType),
                                 1, True, t_pred,args.timelength,
                                  args.testDataType, args.nodemigration, args.epochs,"u_rand")

# python utils/plot_util.py -t sbm_cd -l 7 -nm 5 -iter 250 -emb 32 -rd /media/Data/graph-learning/dynamicgem/results_link_all
