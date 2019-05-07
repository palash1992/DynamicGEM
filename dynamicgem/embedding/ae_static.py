import os
import networkx as nx
import sys

disp_avlbl = True
if os.name == 'posix' and 'DISPLAY' not in os.environ:
    disp_avlbl = False
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append('./')
sys.path.append(os.path.realpath(__file__))

from .static_graph_embedding import StaticGraphEmbedding
from dynamicgem.utils import graph_util, dataprep_util
from dynamicgem.evaluation import visualize_embedding as viz
from .sdne_utils import *
from dynamicgem.evaluation import evaluate_link_prediction as lp
from keras.layers import Input, Dense, Lambda, merge, Subtract
from keras.models import Model, model_from_json
from keras.optimizers import SGD, Adam
from keras import backend as KBack
import tensorflow as tf
from argparse import ArgumentParser
from dynamicgem.graph_generation import dynamic_SBM_graph
import pdb
from joblib import Parallel, delayed
import operator
from time import time


class AE(StaticGraphEmbedding):

    def __init__(self, d, *hyper_dict, **kwargs):
        """ Initialize the Autoencoder class
        Args:
            d: dimension of the embedding
            beta: penalty parameter in matrix B of 2nd order objective
            alpha: weighing hyperparameter for 1st order objective
            nu1: L1-reg hyperparameter
            nu2: L2-reg hyperparameter
            K: number of hidden layers in encoder/decoder
            n_units: vector of length K-1 containing #units in hidden
                     layers of encoder/decoder, not including the units
                     in the embedding layer
            rho: bounding ratio for number of units in consecutive layers (< 1)
            n_iter: number of sgd iterations for first embedding (const)
            n_iter_subs: number of sgd iterations for subsequent embeddings (const)
            xeta: sgd step size parameter
            n_batch: minibatch size for SGD
            modelfile: Files containing previous encoder and decoder models
            weightfile: Files containing previous encoder and decoder weights
            node_frac: Fraction of nodes to use for random walk
            n_walks_per_node: Number of random walks to do for each selected nodes
            len_rw: Length of every random walk
        """
        self._d = d
        hyper_params = {
            'method_name': 'ae',
            'actfn': 'relu',
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
        return '%s_%d' % (self._method_name, self._d)

    def learn_embeddings(self, graph=None, edge_f=None):
        # TensorFlow wizardry
        config = tf.ConfigProto()
        # Don't pre-allocate memory; allocate as-needed
        config.gpu_options.allow_growth = True
        # Only allow a total of half the GPU memory to be allocated
        config.gpu_options.per_process_gpu_memory_fraction = 0.1
        # Create a session with the above options specified.
        KBack.tensorflow_backend.set_session(tf.Session(config=config))

        if not graph and not edge_f:
            raise Exception('graph/edge_f needed')
        if not graph:
            graph = graph_util.loadGraphFromEdgeListTxt(edge_f)

        S = nx.to_scipy_sparse_matrix(graph)
        self._node_num = graph.number_of_nodes()
        t1 = time()

        # Generate encoder, decoder and autoencoder
        self._num_iter = self._n_iter
        self._encoder = get_encoder(self._node_num, self._d,
                                    self._n_units,
                                    self._nu1, self._nu2,
                                    self._actfn)
        self._decoder = get_decoder(self._node_num, self._d,
                                    self._n_units,
                                    self._nu1, self._nu2,
                                    self._actfn)
        self._autoencoder = get_autoencoder(self._encoder, self._decoder)

        # Initialize self._model
        # Input
        x_in = Input(shape=(self._node_num,), name='x_in')
        # Process inputs
        [x_hat, y] = self._autoencoder(x_in)
        # Outputs
        x_diff = Subtract()([x_hat, x_in])

        # Objectives
        def weighted_mse_x(y_true, y_pred):
            """ Hack: This fn doesn't accept additional arguments.
                      We use y_true to pass them.
                y_pred: Contains x_hat - x
                y_true: Contains b
            """
            return KBack.sum(
                KBack.square(y_pred * y_true[:, 0:self._node_num]),
                axis=-1
            )

        # Model
        self._model = Model(input=x_in, output=x_diff)
        sgd = SGD(lr=self._xeta, decay=1e-5, momentum=0.99, nesterov=True)
        adam = Adam(lr=self._xeta, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        self._model.compile(optimizer=sgd, loss=weighted_mse_x)

        history = self._model.fit_generator(
            generator=batch_generator_ae(S, self._beta, self._n_batch, True),
            nb_epoch=self._num_iter,
            samples_per_epoch=S.shape[0] // self._n_batch,
            verbose=1,
            # callbacks=[tensorboard]
            # callbacks=[callbacks.TerminateOnNaN()]
        )
        loss = history.history['loss']
        # Get embedding for all points
        if loss[0] == np.inf or np.isnan(loss[0]):
            print('Model diverged. Assigning random embeddings')
            self._Y = np.random.randn(self._node_num, self._d)
        else:
            try:
                self._Y, self._next_adj = model_batch_predictor_v2(self._autoencoder, S, self._n_batch)
            except:
                pdb.set_trace()
        t2 = time()
        # Save the autoencoder and its weights
        if self._weightfile is not None:
            saveweights(self._encoder, self._weightfile[0])
            saveweights(self._decoder, self._weightfile[1])
        if self._modelfile is not None:
            savemodel(self._encoder, self._modelfile[0])
            savemodel(self._decoder, self._modelfile[1])
        if self._savefilesuffix is not None:
            saveweights(self._encoder,
                        'encoder_weights_' + self._savefilesuffix + '.hdf5')
            saveweights(self._decoder,
                        'decoder_weights_' + self._savefilesuffix + '.hdf5')
            savemodel(self._encoder,
                      'encoder_model_' + self._savefilesuffix + '.json')
            savemodel(self._decoder,
                      'decoder_model_' + self._savefilesuffix + '.json')
            # Save the embedding
            np.savetxt('embedding_' + self._savefilesuffix + '.txt',
                       self._Y)
        return self._Y, (t2 - t1)

    def get_embedding(self, filesuffix=None):
        return self._Y if filesuffix is None else np.loadtxt(
            'embedding_' + filesuffix + '.txt'
        )

    def get_edge_weight(self, i, j, embed=None, filesuffix=None):
        if embed is None:
            if filesuffix is None:
                embed = self._Y
            else:
                embed = np.loadtxt('embedding_' + filesuffix + '.txt')
        if i == j:
            return 0
        else:
            S_hat = self.get_reconst_from_embed(embed[(i, j), :], filesuffix)
            return (S_hat[i, j] + S_hat[j, i]) / 2

    def get_reconstructed_adj(self, embed=None, node_l=None, filesuffix=None):
        if embed is None:
            if filesuffix is None:
                embed = self._Y
            else:
                embed = np.loadtxt('embedding_' + filesuffix + '.txt')
        S_hat = self.get_reconst_from_embed(embed, node_l, filesuffix)
        return graphify(S_hat)

    def get_reconst_from_embed(self, embed, node_l=None, filesuffix=None):
        if filesuffix is None:
            if node_l is not None:
                return self._decoder.predict(
                    embed,
                    batch_size=self._n_batch
                )[:, node_l]
            else:
                return self._decoder.predict(embed, batch_size=self._n_batch)
        else:
            try:
                decoder = model_from_json(
                    open('decoder_model_' + filesuffix + '.json').read())
            except:
                print('Error reading file: {0}. Cannot load previous model'.format(
                    'decoder_model_' + filesuffix + '.json'))
                exit()
            try:
                decoder.load_weights('decoder_weights_' + filesuffix + '.hdf5')
            except:
                print('Error reading file: {0}. Cannot load previous weights'.format(
                    'decoder_weights_' + filesuffix + '.hdf5'))
                exit()
            if node_l is not None:
                return decoder.predict(embed, batch_size=self._n_batch)[:, node_l]
            else:
                return decoder.predict(embed, batch_size=self._n_batch)

    def predict_next_adj(self, node_l=None):
        if node_l is not None:
            # pdb.set_trace()
            return self._next_adj[node_l]
        else:
            return self._next_adj


if __name__ == '__main__':

    parser = ArgumentParser(description='Learns static node embeddings')
    parser.add_argument('-t', '--testDataType',
                        default='sbm_cd',
                        type=str,
                        help='Type of data to test the code')
    parser.add_argument('-l', '--timelength',
                        default=5,
                        type=int,
                        help='Number of time series graph to generate')
    parser.add_argument('-nm', '--nodemigration',
                        default=10,
                        type=int,
                        help='number of nodes to migrate')
    parser.add_argument('-iter', '--epochs',
                        default=250,
                        type=int,
                        help='number of epochs')
    parser.add_argument('-emb', '--embeddimension',
                        default=128,
                        type=int,
                        help='embedding dimension')
    parser.add_argument('-sm', '--samples',
                        default=5000,
                        type=int,
                        help='samples for test data')
    parser.add_argument('-exp', '--exp',
                        default='lp',
                        type=str,
                        help='experiments (lp, emb)')
    parser.add_argument('-rd', '--resultdir',
                        type=str,
                        default='./results_link_all',
                        help="result directory name")

    args = parser.parse_args()
    epochs = args.epochs
    dim_emb = args.embeddimension
    length = args.timelength

    if args.testDataType == 'sbm_cd':
        node_num = 1000
        community_num = 2
        node_change_num = args.nodemigration
        dynamic_sbm_series = dynamic_SBM_graph.get_community_diminish_series_v2(node_num,
                                                                                community_num,
                                                                                length,
                                                                                1,
                                                                                node_change_num)
        embedding = AE(d=dim_emb,
                       beta=5,
                       nu1=1e-6,
                       nu2=1e-6,
                       K=3,
                       n_units=[500, 300, ],
                       n_iter=epochs,
                       xeta=1e-4,
                       n_batch=100,
                       modelfile=['./intermediate/enc_modelsbm.json',
                                  './intermediate/dec_modelsbm.json'],
                       weightfile=['./intermediate/enc_weightssbm.hdf5',
                                   './intermediate/dec_weightssbm.hdf5'])

        graphs = [g[0] for g in dynamic_sbm_series]
        embs = []

        outdir = args.resultdir
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        outdir = outdir + '/' + args.testDataType
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        outdir = outdir + '/staticAE'
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        if args.exp == 'emb':
            for temp_var in range(length):
                emb, _ = embedding.learn_embeddings(graphs[temp_var])
                embs.append(emb)

            result = Parallel(n_jobs=4)(
                delayed(embedding.learn_embeddings)(graphs[temp_var]) for temp_var in range(length))
            for i in range(len(result)):
                embs.append(np.asarray(result[i][0]))

            plt.figure()
            plt.clf()
            viz.plot_static_sbm_embedding(embs[-4:], dynamic_sbm_series[-4:])

            plt.savefig('./' + outdir + '/V_AE_nm' + str(args.nodemigration) + '_l' + str(length) + '_epoch' + str(
                epochs) + '_emb' + str(dim_emb) + '.pdf', bbox_inches='tight', dpi=600)
            plt.show()
            plt.close()

        if args.exp == 'lp':
            lp.expstaticLP(dynamic_sbm_series,
                           graphs,
                           embedding,
                           1,
                           outdir + '/',
                           'nm' + str(args.nodemigration) + '_l' + str(length) + '_emb' + str(dim_emb),
                           )

    elif args.testDataType == 'academic':
        print("datatype:", args.testDataType)

        embedding = AE(d=dim_emb,
                       beta=5,
                       nu1=1e-6,
                       nu2=1e-6,
                       K=3,
                       n_units=[500, 300, ],
                       n_iter=epochs,
                       xeta=1e-4,
                       n_batch=1000,
                       modelfile=['./intermediate/enc_modelacdm.json',
                                  './intermediate/dec_modelacdm.json'],
                       weightfile=['./intermediate/enc_weightsacdm.hdf5',
                                   './intermediate/dec_weightsacdm.hdf5'])

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

        for i in range(length):
            graphs[i] = graph_util.sample_graph_nodes(graphs[i], node_l)

        outdir = args.resultdir
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        outdir = outdir + '/' + args.testDataType
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        outdir = outdir + '/staticAE'
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        if args.exp == 'emb':
            print('plotting embedding not implemented!')

        if args.exp == 'lp':
            lp.expstaticLP(None,
                           graphs[-args.timelength:],
                           embedding,
                           1,
                           outdir + '/',
                           'l' + str(args.timelength) + '_emb' + str(dim_emb) + '_samples' + str(sample),
                           n_sample_nodes=graphs[i].number_of_nodes()
                           )

    elif args.testDataType == 'hep':
        print("datatype:", args.testDataType)
        embedding = AE(d=dim_emb,
                       beta=5,
                       nu1=1e-6,
                       nu2=1e-6,
                       K=3,
                       n_units=[500, 300, ],
                       n_iter=epochs,
                       xeta=1e-4,
                       n_batch=1000,
                       modelfile=['./intermediate/enc_modelhep.json',
                                  './intermediate/dec_modelhep.json'],
                       weightfile=['./intermediate/enc_weightshep.hdf5',
                                   './intermediate/dec_weightshep.hdf5'])

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

        outdir = args.resultdir
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        outdir = outdir + '/' + args.testDataType
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        outdir = outdir + '/staticAE'
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        if args.exp == 'emb':
            print('plotting embedding not implemented!')

        if args.exp == 'lp':
            lp.expstaticLP(None,
                           graphs[-args.timelength:],
                           embedding,
                           1,
                           outdir + '/',
                           'l' + str(args.timelength) + '_emb' + str(dim_emb) + '_samples' + str(sample),
                           n_sample_nodes=graphs[i].number_of_nodes()
                           )

    elif args.testDataType == 'AS':
        print("datatype:", args.testDataType)
        embedding = AE(d=dim_emb,
                       beta=5,
                       nu1=1e-6,
                       nu2=1e-6,
                       K=3,
                       n_units=[500, 300, ],
                       n_iter=epochs,
                       xeta=1e-4,
                       n_batch=1000,
                       modelfile=['./intermediate/enc_modelAS.json',
                                  './intermediate/dec_modelAS.json'],
                       weightfile=['./intermediate/enc_weightsAS.hdf5',
                                   './intermediate/dec_weightsAS.hdf5'])

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

        outdir = args.resultdir
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        outdir = outdir + '/' + args.testDataType
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        outdir = outdir + '/staticAE'
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        if args.exp == 'emb':
            print('plotting embedding not implemented!')

        if args.exp == 'lp':
            lp.expstaticLP(None,
                           graphs[-args.timelength:],
                           embedding,
                           1,
                           outdir + '/',
                           'l' + str(args.timelength) + '_emb' + str(dim_emb) + '_samples' + str(sample),
                           n_sample_nodes=graphs[i].number_of_nodes()
                           )

    elif args.testDataType == 'enron':
        print("datatype:", args.testDataType)
        embedding = AE(d=dim_emb,
                       beta=5,
                       nu1=1e-6,
                       nu2=1e-6,
                       K=3,
                       n_units=[500, 300, ],
                       n_iter=epochs,
                       xeta=1e-8,
                       n_batch=20,
                       modelfile=['./intermediate/enc_modelAS.json',
                                  './intermediate/dec_modelAS.json'],
                       weightfile=['./intermediate/enc_weightsAS.hdf5',
                                   './intermediate/dec_weightsAS.hdf5'])

        files = [file for file in os.listdir('./test_data/enron') if '.gpickle' in file if 'month' in file]
        length = len(files)
        graphsall = []

        for i in range(length):
            G = nx.read_gpickle('./test_data/enron/month_' + str(i + 1) + '_graph.gpickle')
            graphsall.append(G)

        outdir = args.resultdir
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        outdir = outdir + '/' + args.testDataType
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        outdir = outdir + '/staticAE'
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        if args.exp == 'emb':
            print('plotting embedding not implemented!')

        if args.exp == 'lp':
            sample = graphsall[0].number_of_nodes()
            graphs = graphsall[-args.timelength:]
            lp.expstaticLP(None,
                           graphs,
                           embedding,
                           1,
                           outdir + '/',
                           'l' + str(args.timelength) + '_emb' + str(dim_emb) + '_samples' + str(sample),
                           n_sample_nodes=sample
                           )
