try:
    import cPickle as pickle
except:
    import pickle
import os
if os.name == 'posix' and 'DISPLAY' not in os.environ:
    import matplotlib

    matplotlib.use('Agg')
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools
from matplotlib import rc
import random
import seaborn as sns
import numpy as np
import pandas as pd
import pdb
from argparse import ArgumentParser
import subprocess
import six

font = {'family': 'serif', 'serif': ['computer modern roman']}
rc('text', usetex=True)
rc('font', weight='bold')
rc('font', size=9)
rc('lines', markersize=14)
rc('xtick', labelsize=14)
rc('ytick', labelsize=14)
rc('axes', labelsize='x-large')
rc('axes', labelweight='bold')
rc('axes', titlesize='x-large')
rc('axes', linewidth=3)
plt.rc('font', **font)
sns.set_style("darkgrid")

figsize_d = {2: (5, 2),
             4: (9, 2)}

columns_sbm = ['Method', 'data', 'Time Step', 'Mean MAP', 'P@2', 'P@10', 'P@100', 'P@200', 'P@300', 'P@500', 'P@1000',
               'P@EdgeNum']
columns_sbmhyp = ['Method', 'batch', 'epochs', 'learning rate', 'embedding', 'lookback', 'data', 'Mean MAP', 'P@2',
                  'P@10', 'P@100', 'P@200', 'P@300', 'P@500', 'P@1000', 'P@EdgeNum']
columns_sbm1 = ['Method', 'data', 'Mean MAP', 'P@2', 'P@10', 'P@100', 'P@200', 'P@300', 'P@500', 'P@1000', 'P@EdgeNum']
columns_sbm3 = ['Method', 'Data', 'Embedding Size', 'Mean MAP', 'P@2', 'P@10', 'P@100', 'P@200', 'P@300', 'P@500',
                'P@1000', 'P@EdgeNum']
columns_sbm2 = ['Method', 'data', 'lookback', 'Mean MAP', 'P@2', 'P@10', 'P@100', 'P@200', 'P@300', 'P@500', 'P@1000',
                'P@EdgeNum']
columns_sbm_mean = ['Mean MAP', 'P@2', 'P@10', 'P@100', 'P@200', 'P@300', 'P@500', 'P@1000', 'P@EdgeNum']
length = {'academic': 20, 'hep': 50, 'AS': 50}


def render_table(data, col_width=2.0, row_height=0.625, font_size=20,
                 header_color='#CD3333', row_colors=['#f1f1f2', 'w'], edge_color='k',
                 bbox=[0, 0, 1, 1], header_columns=1,
                 ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        # pdb.set_trace()
        fig, ax = plt.subplots(figsize=size)

        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, cellLoc='center', **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])
    return ax


def render_table2(data, col_width=2.0, row_height=0.625, font_size=20,
                  header_color='#CD3333', row_colors=['#f1f1f2', 'w'], edge_color='k',
                  bbox=[0, 0, 1, 1], header_columns=1,
                  ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        # pdb.set_trace()
        fig, ax = plt.subplots(figsize=size)
        # fig(figsize=(5, 5)) 
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, cellLoc='center', **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        # pdb.set_trace()
        if k[1] == 0:
            cell.set_width(0.3)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])
    return ax


def main():
    parser = ArgumentParser(description='plotting tools')
    parser.add_argument('-t', '--testDataType', default='sbm_cd', type=str, help='Type of data to test the code')
    parser.add_argument('-ts', '--subDataType', default='academic', type=str, help='sub dataype')
    parser.add_argument('-nm', '--nodemigration', default=10, type=int, help='number of nodes to migrate')
    parser.add_argument('-emb', '--embeddimension', default=64, type=int, help='embedding dimension')
    parser.add_argument('-rd', '--resultdir', type=str, default='./results_link_all', help="result directory name")
    parser.add_argument('-sm', '--samples', default=2000, type=int, help='samples for test data')
    parser.add_argument('-iter', '--epochs', default=250, type=int, help='number of epochs')
    parser.add_argument('-l', '--timelength', default=10, type=int, help='Number of time series graph to generate')
    parser.add_argument('-fig', '--figure', default=1, type=int, help='1 for figure 0 for table ')
    parser.add_argument('-fs', '--show', default=1, type=int, help='show figure ')
    parser.add_argument('-hp', '--hyperparameter', default=0, type=int, help='show hyper parameter figure figure ')
    parser.add_argument('-mn', '--mean', default=0, type=int, help='mean results')
    parser.add_argument('-lb', '--lookback', default=0, type=int, help='mean results')
    parser.add_argument('-me', '--meanemb', default=0, type=int, help='mean results')
    parser.add_argument('-tb', '--table', default=0, type=int, help='mean results')

    args = parser.parse_args()

    method_s = ['incrementalSVD', 'rerunSVD', 'optimalSVD', 'dynTRIAD', 'staticAE']
    method_hyp = ['dynAE', 'dynRNN', 'dynAERNN']
    method_d = ['dynAE', 'dynRNN', 'dynAERNN']
    method_all = ['incrementalSVD', 'rerunSVD', 'optimalSVD', 'dynTRIAD', 'staticAE', 'dynAE', 'dynRNN', 'dynAERNN']
    method_dict = {'incrementalSVD': 'incSVD', 'rerunSVD': 'rerunSVD', 'optimalSVD': 'optimalSVD',
                   'dynTRIAD': 'dynTriad', 'staticAE': 'dynGEM', 'dynAE': 'dyngraph2vecAE', 'dynRNN': 'dyngraph2vecRNN',
                   'dynAERNN': 'dyngraph2vecAERNN'}

    func = lambda x: float(x) if not '-' == x else np.NaN
    # sbm_cd data comparision
    if args.testDataType == 'sbm_cd' and args.hyperparameter == 0 and args.mean == 0 and args.lookback == 0 and args.meanemb == 0 and args.table == 0:
        resultdir = args.resultdir + '/sbm_cd'

        df = pd.DataFrame(columns=columns_sbm)
        i = 0
        for method in method_s:
            filename = resultdir + '/' + method + '/nm' + str(args.nodemigration) + '_l' + str(
                args.timelength) + '_emb' + str(args.embeddimension) + '.dlpsumm'
            with open(filename, 'r') as f:
                data = f.read().splitlines()[1:]
                if method == 'incrementalSVD':
                    method = 'incSVD'
                for t, lines in enumerate(data):
                    val = lines.split('\t')[1:]
                    df.loc[i] = [method, 'sbm_cd',
                                 int(t + 1),
                                 float('.' + (val[0].split('/')[0]).split('.')[1]),
                                 func(val[1]),
                                 func(val[2]),
                                 func(val[3]),
                                 func(val[4]),
                                 func(val[5]),
                                 func(val[6]),
                                 func(val[7]),
                                 func(val[8])]
                    i += 1

        df2 = pd.DataFrame(columns=columns_sbm)
        i = 0
        for method in method_d:
            filename = resultdir + '/' + method + '/nm' + str(args.nodemigration) + '_l' + str(
                args.timelength) + '_emb' + str(args.embeddimension) + '.dlpsumm'
            with open(filename, 'r') as f:
                data = f.read().splitlines()[1:]
                if method == 'incrementalSVD':
                    method = 'incSVD'
                for t, lines in enumerate(data):
                    val = lines.split('\t')[1:]
                    df2.loc[i] = [method_dict[method],
                                  'sbm_cd',
                                  int(t + 1),
                                  float('.' + (val[0].split('/')[0]).split('.')[1]),
                                  func(val[1]),
                                  func(val[2]),
                                  func(val[3]),
                                  func(val[4]),
                                  func(val[5]),
                                  func(val[6]),
                                  func(val[7]),
                                  func(val[8])]
                    i += 1
        df = df.round(4)
        if args.figure == 1:
            df3 = df[['Method', 'Time Step', 'Mean MAP']].loc[df['Time Step'] < 4]
            df4 = df2[['Method', 'Time Step', 'Mean MAP']].loc[df2['Time Step'] < 4]
            # print(df3,df4)

            frames = [df3, df4]
            df5 = pd.concat(frames, keys=['Method', 'Time Step', 'Mean MAP'])

            print(df5)
            plt.figure(figsize=(8, 5))
            flatui = ["#d46a7e", "#d5b60a", "#9b59b6", "#3498db", "#95a5a6", "#34495e", "#2ecc71", "#e74c3c"]
            # sns.color_palette(flatui)
            g = sns.barplot(x='Time Step', y='Mean MAP', hue='Method', palette=flatui, data=df5)
            g.legend(loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=4)
            g.set_ylim(0.1, 1)
            plt.savefig('./figures/sbm_emb' + str(args.embeddimension) + '_nm' + str(args.nodemigration) + '.pdf',
                        bbox_inches='tight', dpi=300)
            if args.show == 1:
                plt.show()

        if args.figure == 0:
            df3 = df.loc[df['Time Step'] == 1]
            df4 = df2.loc[df2['Time Step'] == 1]

            df3 = df3[['Method', 'P@2', 'P@10', 'P@100', 'P@200', 'P@300', 'P@500', 'P@1000']]
            df4 = df4[['Method', 'P@2', 'P@10', 'P@100', 'P@200', 'P@300', 'P@500', 'P@1000']]

            frames = [df3, df4]
            df5 = pd.concat(frames, keys=['Method', 'P@2', 'P@10', 'P@100', 'P@200', 'P@300', 'P@500', 'P@1000'])
            df5.fillna(-1)

            print(df5)
            # ax = sns.heatmap(df5, annot=True, fmt="d")
            # plt.show()

            render_table2(df5, header_columns=0, col_width=2)
            plt.savefig(
                './figures/sbm_Ptable_emb' + str(args.embeddimension) + '_nm' + str(args.nodemigration) + '.pdf',
                bbox_inches='tight', dpi=300)
            if args.show == 1:
                plt.show()

    elif args.testDataType == 'all':
        testDataType = ['academic', 'hep', 'AS']

        resultdir_dict = {'incrementalSVD': 'resultsTIMERS',
                          'rerunSVD': 'resultsTIMERS',
                          'optimalSVD': 'resultsTIMERS',
                          'dynTriad': args.resultdir,
                          'staticAE': args.resultdir,
                          'dynAE': args.resultdir,
                          'dynRNN': args.resultdir,
                          'dynAERNN': args.resultdir}

        df = pd.DataFrame(columns=columns_sbm)
        i = 0
        for datatype in testDataType:

            for method in method_all:
                resultdir = args.resultdir + '/' + datatype

                filename = resultdir + '/' + method + '/l' + str(length[datatype]) + '_emb' + str(
                    args.embeddimension) + '_samples' + str(args.samples) + '.dlpsumm'

                with open(filename, 'r') as f:
                    data = f.read().splitlines()[1:]

                    for t, lines in enumerate(data):
                        val = lines.split('\t')[1:]
                        df.loc[i] = [method_dict[method],
                                     datatype,
                                     int(t + 1),
                                     float('.' + (val[0].split('/')[0]).split('.')[1]),
                                     func(val[1]),
                                     func(val[2]),
                                     func(val[3]),
                                     func(val[4]),
                                     func(val[5]),
                                     func(val[6]),
                                     func(val[7]),
                                     func(val[8])]
                        i += 1

                        # print(df)

        df3 = df.loc[df['data'] == args.subDataType]

        ###########FIGURE############## 
        if args.figure == 1:
            df3 = df3[['Method', 'data', 'Time Step', 'Mean MAP']].loc[df3['Time Step'] < 4]
            print(df3)
            plt.figure(figsize=(8, 5))
            flatui = ["#d46a7e", "#d5b60a", "#9b59b6", "#3498db", "#95a5a6", "#34495e", "#2ecc71", "#e74c3c"]
            # flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
            # sns.color_palette(flatui)
            g = sns.barplot(x='Time Step', y='Mean MAP', hue='Method', palette=flatui, data=df3)
            g.legend(loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=4)
            try:
                g.set_ylim(0.0, (df3.max())['Mean MAP'] + 0.05)
                plt.savefig('./figures/' + args.subDataType + '_emb' + str(args.embeddimension) + '_sm' + str(
                    args.samples) + '.pdf', bbox_inches='tight', dpi=300)
                if args.show == 1:
                    plt.show()
            except:
                pdb.set_trace()

                ###########TABLE##############
        if args.figure == 0:
            df3 = df3.loc[df3['Time Step'] == 1]
            df3 = df3[['Method', 'P@2', 'P@10', 'P@100', 'P@200', 'P@300', 'P@500', 'P@1000']]
            df3.fillna(-1)
            print(df3)

            render_table2(df3, header_columns=0, col_width=2)
            plt.savefig('./figures/' + args.subDataType + '_Ptable_emb' + str(args.embeddimension) + '_sm' + str(
                args.samples) + '.pdf', bbox_inches='tight', dpi=300)
            if args.show == 1:
                plt.show()

    elif args.mean == 1 and args.meanemb == 1:
        resultdir = args.resultdir + '/' + args.testDataType

        df = pd.DataFrame(columns=columns_sbm3)
        i = 0
        for method in method_all:
            for emb in [64, 128, 256]:
                # resultdir=resultdir_dict[method]+'/'+datatype
                if args.testDataType == 'sbm_cd':
                    filename = resultdir + '/' + method + '/nm' + str(args.nodemigration) + '_l' + str(
                        args.timelength) + '_emb' + str(emb) + '.dlpsumm'
                else:
                    filename = resultdir + '/' + method + '/l' + str(length[args.testDataType]) + '_emb' + str(
                        emb) + '_samples' + str(args.samples) + '.dlpsumm'
                with open(filename, 'r') as f:
                    data = f.read().splitlines()[1:]
                    j = 0
                    dftmp = pd.DataFrame(columns=columns_sbm_mean)
                    for t, lines in enumerate(data):
                        val = lines.split('\t')[1:]
                        if len(val) < 4:
                            continue
                        dftmp.loc[j] = [float('.' + (val[0].split('/')[0]).split('.')[1]),
                                        func(val[1]),
                                        func(val[2]),
                                        func(val[3]),
                                        func(val[4]),
                                        func(val[5]),
                                        func(val[6]),
                                        func(val[7]),
                                        func(val[8])]
                        j += 1
                    dftmp = dftmp.mean(axis=0)
                    df.loc[i] = [method_dict[method], args.testDataType, emb, dftmp['Mean MAP'], dftmp['P@2'],
                                 dftmp['P@10'], dftmp['P@100'], dftmp['P@200'], dftmp['P@300'], dftmp['P@500'],
                                 dftmp['P@1000'], dftmp['P@EdgeNum']]
                    i += 1

                    # pdb.set_trace()
        df = df.round(4)
        if args.figure == 1:
            df3 = df[['Method', 'Embedding Size', 'Mean MAP']]
            print(df3)
            rc('font', size=14)
            plt.figure(figsize=(8.5, 5))
            flatui = ["#f7a52a", "#ee4266", "#9b59b6", "#3498db", "#95a5a6", "#34495e", "#18c462", "#e74c3c"]
            # sns.color_palette(flatui)
            g = sns.barplot(x='Embedding Size', y='Mean MAP', hue='Method', palette=flatui, data=df3)
            # groupedvalues=df3.groupby('Embedding Size').sum().reset_index()    
            # for index, row in groupedvalues.iterrows():
            #     g.text(row.name,row.tip, round(row.total_bill,2), color='black', ha="center")
            # for i, v in enumerate((df3[['Mean MAP']].round(2)).values):
            #     g.text(i/10 ,v + 0.25,  str(v), color='k', fontweight='bold')
            # g.legend(loc='best',bbox_to_anchor=(0.9, 0.2))
            # g.legend(loc='upper center',bbox_to_anchor=(0.5, 1.14), ncol=4)
            g.legend(loc='best', ncol=2)
            # g.set_ylim(0.1, 1)
            g.set_ylim(0.0, (df3.max())['Mean MAP'] + 0.25)
            plt.savefig('./figures/figure' + args.testDataType + '_mean.pdf', bbox_inches='tight', dpi=300)
            if args.show == 1:
                plt.show()
            df3 = df3[['Method', 'Mean MAP']]
            print(df3)
            for method in method_all:
                print(method, df3.loc[df3['Method'] == method_dict[method]].mean())

        if args.figure == 0:
            df3 = df[['Method', 'P@100', 'P@500', 'P@1000']]
            df3.fillna(-1)
            for method in method_all:
                print(method, df3.loc[df3['Method'] == method_dict[method]].mean())

            print(df3)
            # plt.figure(figsize=(5, 5)) 
            render_table2(df3, header_columns=0, col_width=2)
            plt.savefig('./figures/' + args.testDataType + '_emb' + str(args.embeddimension) + '_table_mean.pdf',
                        bbox_inches='tight', dpi=300)
            if args.show == 1:
                plt.show()



    elif args.mean == 1:
        resultdir = args.resultdir + '/' + args.testDataType

        df = pd.DataFrame(columns=columns_sbm1)
        i = 0
        for method in method_all:
            # resultdir=resultdir_dict[method]+'/'+datatype
            if args.testDataType == 'sbm_cd':
                filename = resultdir + '/' + method + '/nm' + str(args.nodemigration) + '_l' + str(
                    args.timelength) + '_emb' + str(args.embeddimension) + '.dlpsumm'
            else:
                filename = resultdir + '/' + method + '/l' + str(length[args.testDataType]) + '_emb' + str(
                    args.embeddimension) + '_samples' + str(args.samples) + '.dlpsumm'
            with open(filename, 'r') as f:
                data = f.read().splitlines()[1:]
                j = 0
                dftmp = pd.DataFrame(columns=columns_sbm_mean)
                for t, lines in enumerate(data):
                    val = lines.split('\t')[1:]
                    dftmp.loc[j] = [float('.' + (val[0].split('/')[0]).split('.')[1]),
                                    func(val[1]),
                                    func(val[2]),
                                    func(val[3]),
                                    func(val[4]),
                                    func(val[5]),
                                    func(val[6]),
                                    func(val[7]),
                                    func(val[8])]
                    j += 1
                dftmp = dftmp.mean(axis=0)
                df.loc[i] = [method_dict[method], args.testDataType, dftmp['Mean MAP'], dftmp['P@2'], dftmp['P@10'],
                             dftmp['P@100'], dftmp['P@200'], dftmp['P@300'], dftmp['P@500'], dftmp['P@1000'],
                             dftmp['P@EdgeNum']]
                i += 1

                # pdb.set_trace()
        df = df.round(4)
        if args.figure == 1:
            df3 = df[['Method', 'Mean MAP']]
            print(df3)
            plt.figure(figsize=(8, 6))
            flatui = ["#d46a7e", "#d5b60a", "#9b59b6", "#3498db", "#95a5a6", "#34495e", "#2ecc71", "#e74c3c"]
            # sns.color_palette(flatui)
            g = sns.barplot(x=range(len(method_all)), y='Mean MAP', hue='Method', palette=flatui, data=df3)
            g.legend(loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=4)
            g.set_ylim(0.1, 1)
            g.set_ylim(0.0, (df3.max())['Mean MAP'] + 0.05)
            plt.savefig('./figures/' + args.testDataType + '_emb' + str(args.embeddimension) + '_mean.pdf',
                        bbox_inches='tight', dpi=300)
            if args.show == 1:
                plt.show()

        if args.figure == 0:
            df3 = df[['Method', 'P@2', 'P@10', 'P@100', 'P@200', 'P@300', 'P@500', 'P@1000']]
            df3.fillna(-1)
            print(df3)
            # plt.figure(figsize=(5, 5)) 
            render_table2(df3, header_columns=0, col_width=2)
            plt.savefig('./figures/' + args.testDataType + '_emb' + str(args.embeddimension) + '_table_mean.pdf',
                        bbox_inches='tight', dpi=300)
            if args.show == 1:
                plt.show()

    elif args.hyperparameter == 1:
        args.testDataType = 'sbm_cd'
        resultdir = './results_hyper/' + args.testDataType

        df = pd.DataFrame(columns=columns_sbmhyp)
        i = 0
        for method in method_hyp:
            # print(method)
            for bs in range(100, 200, 100):
                for epoch in range(100, 500, 50):
                    for eta in ['0.1', '0.01', '0.001', '0.0001', '1e-5', '1e-6']:
                        for emb in range(64, 320, 64):
                            for lb in range(1, 6, 1):
                                try:
                                    filename = resultdir + '/' + method + '/epoch' + str(epoch) + '_bs' + str(
                                        bs) + '_lb' + str(lb) + '_eta' + str(eta) + '_emb' + str(emb) + '.dlpsumm'
                                    # print(filename)
                                    with open(filename, 'r') as f:
                                        data = f.read().splitlines()[1:]
                                        # print(data)
                                        j = 0
                                        dftmp = pd.DataFrame(columns=columns_sbm_mean)
                                        for t, lines in enumerate(data):
                                            val = lines.split('\t')[1:]
                                            dftmp.loc[j] = [float('.' + (val[0].split('/')[0]).split('.')[1]),
                                                            func(val[1]),
                                                            func(val[2]),
                                                            func(val[3]),
                                                            func(val[4]),
                                                            func(val[5]),
                                                            func(val[6]),
                                                            func(val[7]),
                                                            func(val[8])]
                                            j += 1
                                        dftmp = dftmp.mean(axis=0)
                                        # pdb.set_trace()             
                                        df.loc[i] = [method_dict[method], bs, epoch, eta, emb, lb, args.testDataType,
                                                     dftmp['Mean MAP'], dftmp['P@2'], dftmp['P@10'], dftmp['P@100'],
                                                     dftmp['P@200'], dftmp['P@300'], dftmp['P@500'], dftmp['P@1000'],
                                                     dftmp['P@EdgeNum']]
                                        i += 1
                                except:
                                    print("could not open :", filename)
                                    continue

                                    #
        df = df.round(4)
        # print(df)

        # if args.figure==0:            
        df3 = df[['Method', 'learning rate', 'embedding', 'lookback', 'Mean MAP']]

        df3.fillna(-1)
        print(df3)
        pdb.set_trace()
        # plt.figure(figsize=(5, 5)) 
        render_table2(df3, header_columns=0, col_width=2)
        # plt.savefig('./figures/'+args.testDataType+'_emb'+str(args.embeddimension)+'_table_mean.pdf',bbox_inches='tight',dpi=300)  
        # if args.show==1:
        plt.show()

    elif args.lookback == 1:
        resultdir_dict = {'dynAE': './resultsdynAE_lookback',
                          'dynRNN': './resultsdynRNN_lookback',
                          'dynAERNN': './resultsdynAERNN_lookback'}
        method_all = ['dynAE', 'dynRNN', 'dynAERNN']
        datatype = args.testDataType
        df = pd.DataFrame(columns=columns_sbm2)
        i = 0
        for method in method_all:
            for datatype in ['hep', 'AS']:
                for lb in range(1, 4):
                    resultdir = resultdir_dict[method] + '/' + datatype
                    if method == 'dynAE':
                        filename = resultdir + '/' + method + '/lb_' + str(lb) + '_l' + str(
                            length[datatype]) + '_emb' + str(args.embeddimension) + '_samples' + str(
                            args.samples) + '.dlpsumm'
                    else:
                        filename = resultdir + '/' + method + '/lb' + str(lb) + '_l' + str(
                            length[datatype]) + '_emb' + str(args.embeddimension) + '_samples' + str(
                            args.samples) + '.dlpsumm'
                    with open(filename, 'r') as f:
                        data = f.read().splitlines()[1:]
                        j = 0
                        dftmp = pd.DataFrame(columns=columns_sbm_mean)
                        for t, lines in enumerate(data):
                            val = lines.split('\t')[1:]
                            if len(val) < 4:
                                continue

                            dftmp.loc[j] = [float('.' + (val[0].split('/')[0]).split('.')[1]),
                                            func(val[1]),
                                            func(val[2]),
                                            func(val[3]),
                                            func(val[4]),
                                            func(val[5]),
                                            func(val[6]),
                                            func(val[7]),
                                            func(val[8])]
                            j += 1
                        dftmp = dftmp.mean(axis=0)
                        df.loc[i] = [method_dict[method], datatype, lb, dftmp['Mean MAP'], dftmp['P@2'], dftmp['P@10'],
                                     dftmp['P@100'], dftmp['P@200'], dftmp['P@300'], dftmp['P@500'], dftmp['P@1000'],
                                     dftmp['P@EdgeNum']]
                        i += 1

                        # pdb.set_trace()
        df = df.round(4)
        print(df)
        df3 = df[['Method', 'data', 'lookback', 'Mean MAP']]
        print(df3)
        df3 = df3.loc[df3['data'] == args.testDataType]
        print(df3)
        rc('font', size=18)
        plt.figure(figsize=(6, 6))
        flatui = ["#34495e", "#2ecc71", "#e74c3c"]
        # sns.color_palette(flatui)
        g = sns.barplot(x='lookback', y='Mean MAP', hue='Method', palette=flatui, data=df3)
        g.legend(loc='best')
        # g.legend(loc='upper center',bbox_to_anchor=(0.5, 1.14), ncol=2)
        g.set_ylim(0.0, (df3.max())['Mean MAP'] + 0.4)
        plt.savefig('./figures/lookback_' + args.testDataType + '.pdf', bbox_inches='tight', dpi=300)
        plt.show()

        # if args.figure==0:            
        # df3=df[['Method','data','lookback','Mean MAP']]
        # df3.fillna(-1)
        # print(df3)
        # # plt.figure(figsize=(5, 5)) 
        # render_table2(df3, header_columns=0, col_width=4) 
        # plt.savefig('./figures/lookback_'+args.testDataType+'.pdf',bbox_inches='tight',dpi=300)  
        # plt.show()  
    elif args.table == 1:

        datatype = ['sbm_cd', 'hep', 'AS']
        df = pd.DataFrame(columns=columns_sbm3)
        i = 0
        for datat in datatype:
            resultdir = args.resultdir + '/' + datat
            for method in method_all:
                for emb in [64, 128, 256]:
                    # print(datat,method,emb)
                    # resultdir=resultdir_dict[method]+'/'+datatype
                    if datat == 'sbm_cd':
                        filename = resultdir + '/' + method + '/nm' + str(20) + '_l' + str(10) + '_emb' + str(
                            emb) + '.dlpsumm'
                    else:
                        filename = resultdir + '/' + method + '/l' + str(50) + '_emb' + str(emb) + '_samples' + str(
                            2000) + '.dlpsumm'
                    with open(filename, 'r') as f:
                        data = f.read().splitlines()[1:]
                        j = 0
                        dftmp = pd.DataFrame(columns=columns_sbm_mean)
                        for t, lines in enumerate(data):
                            val = lines.split('\t')[1:]
                            if len(val) < 4:
                                continue
                            dftmp.loc[j] = [float('.' + (val[0].split('/')[0]).split('.')[1]),
                                            func(val[1]),
                                            func(val[2]),
                                            func(val[3]),
                                            func(val[4]),
                                            func(val[5]),
                                            func(val[6]),
                                            func(val[7]),
                                            func(val[8])]
                            j += 1
                        dftmp = dftmp.mean(axis=0)
                        df.loc[i] = [method_dict[method], datat, emb, dftmp['Mean MAP'], dftmp['P@2'], dftmp['P@10'],
                                     dftmp['P@100'], dftmp['P@200'], dftmp['P@300'], dftmp['P@500'], dftmp['P@1000'],
                                     dftmp['P@EdgeNum']]
                        i += 1

                        # pdb.set_trace()
        # df=df.round(4)
        # print(df)
        df3 = df[['Method', 'Data', 'P@100', 'P@500', 'P@1000']]
        # print(df3)
        prec = ['P@100', 'P@500', 'P@1000']

        i = 0
        df5 = pd.DataFrame(columns=['Method', 'Data', 'P@100', 'P@500', 'P@1000'])
        for method in method_all:
            for datat in datatype:
                df4 = df3.loc[df3['Method'] == method_dict[method]]
                df4 = df4.loc[df4['Data'] == datat]
                df4 = df4.mean(axis=0)
                df5.loc[i] = [method_dict[method], datat, df4['P@100'], df4['P@500'], df4['P@1000']]
                i += 1
        # print(df5) 
        df5 = df5.round(4)
        for datat in datatype:
            print(df5.loc[df5['Data'] == datat])

            # if args.figure==0:
        #     df3=df[['Method','P@100','P@500','P@1000']]
        #     df3.fillna(-1)
        #     for method in method_all:

        #         print(method, df3.loc[df3['Method']==method_dict[method]].mean())  

        #     print(df3)
        #     # plt.figure(figsize=(5, 5)) 
        #     render_table2(df3, header_columns=0, col_width=2) 
        #     plt.savefig('./figures/'+args.testDataType+'_emb'+str(args.embeddimension)+'_table_mean.pdf',bbox_inches='tight',dpi=300)  
        #     if args.show==1:
        #         plt.show()               


if __name__ == '__main__':
    main()

# python utils/plot_util.py -t sbm_cd -l 7 -nm 5 -iter 250 -emb 32 -rd /media/Data/graph-learning/dynamicgem/results_link_all
# !/bin/bash
# for bs in $(seq 100 100 500) 
#     do 
#         for iter in $(seq 100 50 500) 
#             do
#                 for eta in 1e-3 1e-4 1e-5 1e-6 1e-7
#                     do
#                         for lb in $(seq 1 1 5) 
#                             do
#                                 for emb in $(seq 64 64 320) 
#                                     do
#                                     python -W ignore ./embedding/dynAE.py -iter $iter -eta $eta -bs $bs -emb $emb -lb $lb -l 10 -nm 10 -rd ./results_hyper -ht 1
#                                     python -W ignore ./embedding/dynRNN.py -iter $iter -eta $eta -bs $bs -emb $emb -lb $lb -l 10 -nm 10 -rd ./results_hyper -ht 1
#                                     python -W ignore ./embedding/dynAERNN.py -iter $iter -eta $eta -bs $bs -emb $emb -lb $lb -l 10 -nm 10 -rd ./results_hyper -ht 1
#                                     done
#                             done        
#                     done
#             done
#     done
