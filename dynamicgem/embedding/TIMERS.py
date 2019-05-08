disp_avlbl = True
import os
if os.name == 'posix' and 'DISPLAY' not in os.environ:
    disp_avlbl = False
    import matplotlib

    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import scipy.io as sio
import networkx as nx

import sys

sys.path.append('./')
sys.path.append(os.path.realpath(__file__))

from dynamicgem.embedding.static_graph_embedding import StaticGraphEmbedding
from dynamicgem.utils import graph_util, plot_util, dataprep_util
from dynamicgem.evaluation import visualize_embedding as viz
from dynamicgem.embedding.sdne_utils import *
from dynamicgem.graph_generation import SBM_graph
from dynamicgem.evaluation import evaluate_graph_reconstruction as gr
from dynamicgem.evaluation import evaluate_link_prediction as lp

from argparse import ArgumentParser
from dynamicgem.graph_generation import dynamic_SBM_graph
import TIMERS_ALL
import pdb
import operator
# from theano.printing import debugprint as dbprint, pprint
from time import time


class TIMERS(StaticGraphEmbedding):
    """ Initialize the embedding class
        Args:
            K: dimension of the embedding
            theta: threshold for rerun
            datafile: location of the data file
            length : total timesteps of the data
            nodemigraiton: number of nodes to migrate for sbm_cd datatype
            resultdir: directory to save the result
            datatype: sbm_cd, enron, academia, hep, AS
    """

    def __init__(self,  *hyper_dict, **kwargs):

        hyper_params = {
            'method_name': 'TIMERS',
            'modelfile': None,
            'weightfile': None,
            'savefilesuffix': None

        }
        hyper_params.update(kwargs)
        for key in hyper_params.keys():
            self.__setattr__('_%s' % key, hyper_params[key])
        for dictionary in hyper_dict:
            for key in dictionary:
                self.__setattr__('_%s' % key, dictionary[key])

    def get_method_name(self):
        return self._method_name

    def get_method_summary(self):
        return '%s' % (self._method_name)

    def learn_embedding(self, graph=None):
        timers = TIMERS_ALL.initialize()
        timers.TIMERS(self._datafile, self._K / 2, self._Theta, self._datatype, nargout=0)

    def plotresults(self, dynamic_sbm_series):

        plt.figure()
        plt.clf()
        viz.plot_static_sbm_embedding(self._X[-4:], dynamic_sbm_series[-4:])

        resultdir = self._resultdir + '/' + self._datatype
        if not os.path.exists(resultdir):
            os.mkdir(resultdir)

        resultdir = resultdir + '/' + self._method
        if not os.path.exists(resultdir):
            os.mkdir(resultdir)

        #         plt.savefig('./'+resultdir+'/V_'+self._method+'_nm'+str(self._nodemigration)+'_l'+str(self._length)+'_theta'+str(theta)+'_emb'+str(self._K*2)+'.pdf',bbox_inches='tight',dpi=600)
        plt.show()
        # plt.close()  

    def get_embedding(self, outdir_tmp, method):
        self._outdir_tmp = outdir_tmp
        self._method = method
        self._X = dataprep_util.getemb_TIMERS(self._outdir_tmp, int(self._length), int(self._K // 2), self._method)
        return self._X

    def get_edge_weight(self, t, i, j):
        try:
            return np.dot(self._X[t][i, :int(self._K // 2)], self._X[t][j, int(self._K // 2):])
        except Exception as e:
            print(e.message, e.args)
            pdb.set_trace()

    def get_reconstructed_adj(self, t, X=None, node_l=None):
        if X is not None:
            node_num = X.shape[0]
            # self._X = X
        else:
            node_num = self._node_num
        adj_mtx_r = np.zeros((node_num, node_num))
        for v_i in range(node_num):
            for v_j in range(node_num):
                if v_i == v_j:
                    continue
                adj_mtx_r[v_i, v_j] = self.get_edge_weight(t, v_i, v_j)
        return adj_mtx_r

    def predict_next_adj(self, t, node_l=None):
        if node_l is not None:
            return self.get_reconstructed_adj(t, node_l)
        else:
            return self.get_reconstructed_adj(t)


if __name__ == '__main__':

    parser = ArgumentParser(description='Learns static node embeddings')
    parser.add_argument('-t', '--testDataType',
                        default='sbm_cd',
                        type=str,
                        help='Type of data to test the code')
    parser.add_argument('-l', '--timelength',
                        default=10,
                        type=int,
                        help='Number of time series graph to generate')
    parser.add_argument('-nm', '--nodemigration',
                        default=10,
                        type=int,
                        help='number of nodes to migrate')
    parser.add_argument('-emb', '--embeddimension',
                        default=256,
                        type=float,
                        help='embedding dimension')
    parser.add_argument('-theta', '--theta',
                        default=0.5,  # 0.17
                        type=float,
                        help='a threshold for re-run SVD')
    parser.add_argument('-rdir', '--resultdir',
                        default='./results_link_all',  # 0.17
                        type=str,
                        help='directory for storing results')
    parser.add_argument('-sm', '--samples',
                        default=5000,
                        type=int,
                        help='samples for test data')
    parser.add_argument('-exp', '--exp',
                        default='lp',
                        type=str,
                        help='experiments (lp, emb)')

    args = parser.parse_args()
    dim_emb = args.embeddimension
    length = args.timelength
    theta = args.theta
    sample = args.samples

    if args.testDataType == 'sbm_cd':
        node_num = 1000
        community_num = 2
        node_change_num = args.nodemigration
        dynamic_sbm_series = dynamic_SBM_graph.get_community_diminish_series_v2(node_num,
                                                                                community_num,
                                                                                length,
                                                                                1,
                                                                                node_change_num)
        graphs = [g[0] for g in dynamic_sbm_series]

        datafile = dataprep_util.prep_input_TIMERS(graphs, length, args.testDataType)

        embedding = TIMERS(K=dim_emb,
                           Theta=theta,
                           datafile=datafile,
                           length=length,
                           nodemigration=args.nodemigration,
                           resultdir=args.resultdir,
                           datatype=args.testDataType
                           )
        outdir_tmp = './output'
        if not os.path.exists(outdir_tmp):
            os.mkdir(outdir_tmp)
        outdir_tmp = outdir_tmp + '/sbm_cd'
        if not os.path.exists(outdir_tmp):
            os.mkdir(outdir_tmp)
        if not os.path.exists(outdir_tmp + '/incrementalSVD'):
            os.mkdir(outdir_tmp + '/incrementalSVD')
        if not os.path.exists(outdir_tmp + '/rerunSVD'):
            os.mkdir(outdir_tmp + '/rerunSVD')
        if not os.path.exists(outdir_tmp + '/optimalSVD'):
            os.mkdir(outdir_tmp + '/optimalSVD')

        if args.exp == 'emb':
            print('plotting embedding not implemented!')

        if args.exp == 'lp':
            embedding.learn_embedding()

            outdir = args.resultdir
            if not os.path.exists(outdir):
                os.mkdir(outdir)
            outdir = outdir + '/' + args.testDataType
            if not os.path.exists(outdir):
                os.mkdir(outdir)

            embedding.get_embedding(outdir_tmp, 'incrementalSVD')
            # embedding.plotresults()  
            outdir1 = outdir + '/incrementalSVD'
            if not os.path.exists(outdir1):
                os.mkdir(outdir1)
            lp.expstaticLP_TIMERS(dynamic_sbm_series,
                                  graphs,
                                  embedding,
                                  1,
                                  outdir1 + '/',
                                  'nm' + str(args.nodemigration) + '_l' + str(length) + '_emb' + str(int(dim_emb)),
                                  )

            embedding.get_embedding(outdir_tmp, 'rerunSVD')
            outdir1 = outdir + '/rerunSVD'
            # embedding.plotresults()
            if not os.path.exists(outdir1):
                os.mkdir(outdir1)
            lp.expstaticLP_TIMERS(dynamic_sbm_series,
                                  graphs,
                                  embedding,
                                  1,
                                  outdir1 + '/',
                                  'nm' + str(args.nodemigration) + '_l' + str(length) + '_emb' + str(int(dim_emb)),
                                  )

            embedding.get_embedding(outdir_tmp, 'optimalSVD')
            # embedding.plotresults()
            outdir1 = outdir + '/optimalSVD'
            if not os.path.exists(outdir1):
                os.mkdir(outdir1)
            lp.expstaticLP_TIMERS(dynamic_sbm_series,
                                  graphs,
                                  embedding,
                                  1,
                                  outdir1 + '/',
                                  'nm' + str(args.nodemigration) + '_l' + str(length) + '_emb' + str(int(dim_emb)),
                                  )

    elif args.testDataType == 'academic':
        print("datatype:", args.testDataType)

        sample = args.samples
        if not os.path.exists('./test_data/academic/pickle'):
            os.mkdir('./test_data/academic/pickle')
            graphs, length = dataprep_util.get_graph_academic('./test_data/academic/adjlist')
            for i in range(length):
                nx.write_gpickle(graphs[i], './test_data/academic/pickle/' + str(i))
        else:
            length = len(os.listdir('./test_data/academic/pickle'))
            graphs = []
            for i in range(length):
                graphs.append(nx.read_gpickle('./test_data/academic/pickle/' + str(i)))

        G_cen = nx.degree_centrality(graphs[29])  # graph 29 in academia has highest number of edges
        G_cen = sorted(G_cen.items(), key=operator.itemgetter(1), reverse=True)
        node_l = []
        i = 0
        while i < sample:
            node_l.append(G_cen[i][0])
            i += 1
        # pdb.set_trace()
        # node_l = np.random.choice(range(graphs[29].number_of_nodes()), 5000, replace=False)
        # print(node_l)
        for i in range(length):
            graphs[i] = graph_util.sample_graph_nodes(graphs[i], node_l)
        # pdb.set_trace()
        graphs = graphs[-args.timelength:]

        datafile = dataprep_util.prep_input_TIMERS(graphs, args.timelength, args.testDataType)

        embedding = TIMERS(K=dim_emb,
                           Theta=theta,
                           datafile=datafile,
                           length=args.timelength,
                           nodemigration=args.nodemigration,
                           resultdir=args.resultdir,
                           datatype=args.testDataType
                           )
        outdir_tmp = './output'
        if not os.path.exists(outdir_tmp):
            os.mkdir(outdir_tmp)
        outdir_tmp = outdir_tmp + '/' + args.testDataType
        if not os.path.exists(outdir_tmp):
            os.mkdir(outdir_tmp)
        if not os.path.exists(outdir_tmp + '/incrementalSVD'):
            os.mkdir(outdir_tmp + '/incrementalSVD')
        if not os.path.exists(outdir_tmp + '/rerunSVD'):
            os.mkdir(outdir_tmp + '/rerunSVD')
        if not os.path.exists(outdir_tmp + '/optimalSVD'):
            os.mkdir(outdir_tmp + '/optimalSVD')

        if args.exp == 'emb':
            print('plotting embedding not implemented!')

        if args.exp == 'lp':
            embedding.learn_embedding()

            outdir = args.resultdir
            if not os.path.exists(outdir):
                os.mkdir(outdir)
            outdir = outdir + '/' + args.testDataType
            if not os.path.exists(outdir):
                os.mkdir(outdir)

            embedding.get_embedding(outdir_tmp, 'incrementalSVD')
            # embedding.plotresults()  
            outdir1 = outdir + '/incrementalSVD'
            if not os.path.exists(outdir1):
                os.mkdir(outdir1)
            lp.expstaticLP_TIMERS(None,
                                  graphs,
                                  embedding,
                                  1,
                                  outdir1 + '/',
                                  'l' + str(args.timelength) + '_emb' + str(int(dim_emb)) + '_samples' + str(sample),
                                  n_sample_nodes=sample
                                  )

            embedding.get_embedding(outdir_tmp, 'rerunSVD')
            outdir1 = outdir + '/rerunSVD'
            # embedding.plotresults()
            if not os.path.exists(outdir1):
                os.mkdir(outdir1)
            lp.expstaticLP_TIMERS(None,
                                  graphs,
                                  embedding,
                                  1,
                                  outdir1 + '/',
                                  'l' + str(args.timelength) + '_emb' + str(int(dim_emb)) + '_samples' + str(sample),
                                  n_sample_nodes=sample
                                  )

            embedding.get_embedding(outdir_tmp, 'optimalSVD')
            # embedding.plotresults()
            outdir1 = outdir + '/optimalSVD'
            if not os.path.exists(outdir1):
                os.mkdir(outdir1)
            lp.expstaticLP_TIMERS(None,
                                  graphs,
                                  embedding,
                                  1,
                                  outdir1 + '/',
                                  'l' + str(args.timelength) + '_emb' + str(int(dim_emb)) + '_samples' + str(sample),
                                  n_sample_nodes=sample
                                  )


    elif args.testDataType == 'hep':
        print("datatype:", args.testDataType)

        if not os.path.exists('./test_data/hep/pickle'):
            os.mkdir('./test_data/hep/pickle')
            files = [file for file in os.listdir('./test_data/hep/hep-th') if '.gpickle' in file]
            length = len(files)
            graphs = []
            for i in range(length):
                G = nx.read_gpickle('./test_data/hep/hep-th/month_' + str(i + 1) + '_graph.gpickle')

                graphs.append(G)
            total_nodes = graphs[-1].number_of_nodes()

            for i in range(length):
                for j in range(total_nodes):
                    if j not in graphs[i].nodes():
                        graphs[i].add_node(j)

            for i in range(length):
                nx.write_gpickle(graphs[i], './test_data/hep/pickle/' + str(i))
        else:
            length = len(os.listdir('./test_data/hep/pickle'))
            graphs = []
            for i in range(length):
                graphs.append(nx.read_gpickle('./test_data/hep/pickle/' + str(i)))

        # pdb.set_trace()            
        sample = args.samples
        G_cen = nx.degree_centrality(graphs[-1])  # graph 29 in academia has highest number of edges
        G_cen = sorted(G_cen.items(), key=operator.itemgetter(1), reverse=True)
        node_l = []
        i = 0
        while i < sample:
            node_l.append(G_cen[i][0])
            i += 1
        for i in range(length):
            graphs[i] = graph_util.sample_graph_nodes(graphs[i], node_l)

        graphs = graphs[-args.timelength:]

        datafile = dataprep_util.prep_input_TIMERS(graphs, args.timelength, args.testDataType)

        embedding = TIMERS(K=dim_emb,
                           Theta=theta,
                           datafile=datafile,
                           length=args.timelength,
                           nodemigration=args.nodemigration,
                           resultdir=args.resultdir,
                           datatype=args.testDataType
                           )
        outdir_tmp = './output'
        if not os.path.exists(outdir_tmp):
            os.mkdir(outdir_tmp)
        outdir_tmp = outdir_tmp + '/' + args.testDataType
        if not os.path.exists(outdir_tmp):
            os.mkdir(outdir_tmp)
        if not os.path.exists(outdir_tmp + '/incrementalSVD'):
            os.mkdir(outdir_tmp + '/incrementalSVD')
        if not os.path.exists(outdir_tmp + '/rerunSVD'):
            os.mkdir(outdir_tmp + '/rerunSVD')
        if not os.path.exists(outdir_tmp + '/optimalSVD'):
            os.mkdir(outdir_tmp + '/optimalSVD')

        if args.exp == 'emb':
            print('plotting embedding not implemented!')

        if args.exp == 'lp':
            embedding.learn_embedding()

            outdir = args.resultdir
            if not os.path.exists(outdir):
                os.mkdir(outdir)
            outdir = outdir + '/' + args.testDataType
            if not os.path.exists(outdir):
                os.mkdir(outdir)

            embedding.get_embedding(outdir_tmp, 'incrementalSVD')
            # embedding.plotresults()  
            outdir1 = outdir + '/incrementalSVD'
            if not os.path.exists(outdir1):
                os.mkdir(outdir1)
            lp.expstaticLP_TIMERS(None,
                                  graphs,
                                  embedding,
                                  1,
                                  outdir1 + '/',
                                  'l' + str(args.timelength) + '_emb' + str(int(dim_emb)) + '_samples' + str(sample),
                                  n_sample_nodes=sample
                                  )

            embedding.get_embedding(outdir_tmp, 'rerunSVD')
            outdir1 = outdir + '/rerunSVD'
            # embedding.plotresults()
            if not os.path.exists(outdir1):
                os.mkdir(outdir1)
            lp.expstaticLP_TIMERS(None,
                                  graphs,
                                  embedding,
                                  1,
                                  outdir1 + '/',
                                  'l' + str(args.timelength) + '_emb' + str(int(dim_emb)) + '_samples' + str(sample),
                                  n_sample_nodes=sample
                                  )

            embedding.get_embedding(outdir_tmp, 'optimalSVD')
            # embedding.plotresults()
            outdir1 = outdir + '/optimalSVD'
            if not os.path.exists(outdir1):
                os.mkdir(outdir1)
            lp.expstaticLP_TIMERS(None,
                                  graphs,
                                  embedding,
                                  1,
                                  outdir1 + '/',
                                  'l' + str(args.timelength) + '_emb' + str(int(dim_emb)) + '_samples' + str(sample),
                                  n_sample_nodes=sample
                                  )

    elif args.testDataType == 'AS':
        print("datatype:", args.testDataType)

        files = [file for file in os.listdir('./test_data/AS/as-733') if '.gpickle' in file]
        length = len(files)
        graphs = []

        for i in range(length):
            G = nx.read_gpickle('./test_data/AS/as-733/month_' + str(i + 1) + '_graph.gpickle')
            graphs.append(G)

        sample = args.samples
        G_cen = nx.degree_centrality(graphs[-1])  # graph 29 in academia has highest number of edges
        G_cen = sorted(G_cen.items(), key=operator.itemgetter(1), reverse=True)
        node_l = []
        i = 0
        while i < sample:
            node_l.append(G_cen[i][0])
            i += 1
        for i in range(length):
            graphs[i] = graph_util.sample_graph_nodes(graphs[i], node_l)

        graphs = graphs[-args.timelength:]

        datafile = dataprep_util.prep_input_TIMERS(graphs, args.timelength, args.testDataType)

        embedding = TIMERS(K=dim_emb,
                           Theta=theta,
                           datafile=datafile,
                           length=args.timelength,
                           nodemigration=args.nodemigration,
                           resultdir=args.resultdir,
                           datatype=args.testDataType
                           )
        outdir_tmp = './output'
        if not os.path.exists(outdir_tmp):
            os.mkdir(outdir_tmp)
        outdir_tmp = outdir_tmp + '/' + args.testDataType
        if not os.path.exists(outdir_tmp):
            os.mkdir(outdir_tmp)
        if not os.path.exists(outdir_tmp + '/incrementalSVD'):
            os.mkdir(outdir_tmp + '/incrementalSVD')
        if not os.path.exists(outdir_tmp + '/rerunSVD'):
            os.mkdir(outdir_tmp + '/rerunSVD')
        if not os.path.exists(outdir_tmp + '/optimalSVD'):
            os.mkdir(outdir_tmp + '/optimalSVD')

        if args.exp == 'emb':
            print('plotting embedding not implemented!')

        if args.exp == 'lp':
            embedding.learn_embedding()

            outdir = args.resultdir
            if not os.path.exists(outdir):
                os.mkdir(outdir)
            outdir = outdir + '/' + args.testDataType
            if not os.path.exists(outdir):
                os.mkdir(outdir)

            embedding.get_embedding(outdir_tmp, 'incrementalSVD')
            # embedding.plotresults()  
            outdir1 = outdir + '/incrementalSVD'
            if not os.path.exists(outdir1):
                os.mkdir(outdir1)
            lp.expstaticLP_TIMERS(None,
                                  graphs,
                                  embedding,
                                  1,
                                  outdir1 + '/',
                                  'l' + str(args.timelength) + '_emb' + str(int(dim_emb)) + '_samples' + str(sample),
                                  n_sample_nodes=sample
                                  )

            embedding.get_embedding(outdir_tmp, 'rerunSVD')
            outdir1 = outdir + '/rerunSVD'
            # embedding.plotresults()
            if not os.path.exists(outdir1):
                os.mkdir(outdir1)
            lp.expstaticLP_TIMERS(None,
                                  graphs,
                                  embedding,
                                  1,
                                  outdir1 + '/',
                                  'l' + str(args.timelength) + '_emb' + str(int(dim_emb)) + '_samples' + str(sample),
                                  n_sample_nodes=sample
                                  )

            embedding.get_embedding(outdir_tmp, 'optimalSVD')
            # embedding.plotresults()
            outdir1 = outdir + '/optimalSVD'
            if not os.path.exists(outdir1):
                os.mkdir(outdir1)
            lp.expstaticLP_TIMERS(None,
                                  graphs,
                                  embedding,
                                  1,
                                  outdir1 + '/',
                                  'l' + str(args.timelength) + '_emb' + str(int(dim_emb)) + '_samples' + str(sample),
                                  n_sample_nodes=sample
                                  )

    elif args.testDataType == 'enron':
        print("datatype:", args.testDataType)

        files = [file for file in os.listdir('./test_data/enron') if 'month' in file]
        length = len(files)
        # print(length)
        graphsall = []

        for i in range(length):
            G = nx.read_gpickle('./test_data/enron/month_' + str(i + 1) + '_graph.gpickle')
            graphsall.append(G)

        sample = graphsall[0].number_of_nodes()
        graphs = graphsall[-args.timelength:]
        # pdb.set_trace()
        datafile = dataprep_util.prep_input_TIMERS(graphs, args.timelength, args.testDataType)

        embedding = TIMERS(K=dim_emb,
                           Theta=theta,
                           datafile=datafile,
                           length=args.timelength,
                           nodemigration=args.nodemigration,
                           resultdir=args.resultdir,
                           datatype=args.testDataType
                           )
        outdir_tmp = './output'
        if not os.path.exists(outdir_tmp):
            os.mkdir(outdir_tmp)
        outdir_tmp = outdir_tmp + '/' + args.testDataType
        if not os.path.exists(outdir_tmp):
            os.mkdir(outdir_tmp)
        if not os.path.exists(outdir_tmp + '/incrementalSVD'):
            os.mkdir(outdir_tmp + '/incrementalSVD')
        if not os.path.exists(outdir_tmp + '/rerunSVD'):
            os.mkdir(outdir_tmp + '/rerunSVD')
        if not os.path.exists(outdir_tmp + '/optimalSVD'):
            os.mkdir(outdir_tmp + '/optimalSVD')

        if args.exp == 'emb':
            print('plotting embedding not implemented!')

        if args.exp == 'lp':
            embedding.learn_embedding()

            outdir = args.resultdir
            if not os.path.exists(outdir):
                os.mkdir(outdir)
            outdir = outdir + '/' + args.testDataType
            if not os.path.exists(outdir):
                os.mkdir(outdir)

            embedding.get_embedding(outdir_tmp, 'incrementalSVD')
            # embedding.plotresults()  
            outdir1 = outdir + '/incrementalSVD'
            if not os.path.exists(outdir1):
                os.mkdir(outdir1)
            lp.expstaticLP_TIMERS(None,
                                  graphs,
                                  embedding,
                                  1,
                                  outdir1 + '/',
                                  'l' + str(args.timelength) + '_emb' + str(int(dim_emb)) + '_samples' + str(sample),
                                  n_sample_nodes=sample
                                  )

            embedding.get_embedding(outdir_tmp, 'rerunSVD')
            outdir1 = outdir + '/rerunSVD'
            # embedding.plotresults()
            if not os.path.exists(outdir1):
                os.mkdir(outdir1)
            lp.expstaticLP_TIMERS(None,
                                  graphs,
                                  embedding,
                                  1,
                                  outdir1 + '/',
                                  'l' + str(args.timelength) + '_emb' + str(int(dim_emb)) + '_samples' + str(sample),
                                  n_sample_nodes=sample
                                  )

            embedding.get_embedding(outdir_tmp, 'optimalSVD')
            # embedding.plotresults()
            outdir1 = outdir + '/optimalSVD'
            if not os.path.exists(outdir1):
                os.mkdir(outdir1)
            lp.expstaticLP_TIMERS(None,
                                  graphs,
                                  embedding,
                                  1,
                                  outdir1 + '/',
                                  'l' + str(args.timelength) + '_emb' + str(int(dim_emb)) + '_samples' + str(sample),
                                  n_sample_nodes=sample
                                  )
