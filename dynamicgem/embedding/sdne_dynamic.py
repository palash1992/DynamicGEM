disp_avlbl = True
import os
if os.name == 'posix' and 'DISPLAY' not in os.environ:
    disp_avlbl = False
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import scipy.io as sio

import sys
sys.path.append('./')

from .dynamic_graph_embedding import DynamicGraphEmbedding
from dynamicgem.utils import graph_util, plot_util
from dynamicgem.visualization import plot_dynamic_sbm_embedding
from dynamicgem.graph_generation import dynamic_SBM_graph
from .sdne_utils import *

from keras.layers import Input, Dense, Lambda, merge, Subtract
from keras.models import Model, model_from_json
import keras.regularizers as Reg
from keras.optimizers import SGD, Adam
from keras import backend as KBack

from theano.printing import debugprint as dbprint, pprint
import pdb
from argparse import ArgumentParser
import time

MAX_CHUNK_SIZE = 50000

class SDNE(DynamicGraphEmbedding):

    def __init__(self, d, beta, alpha, nu1, nu2, K, n_units, rho, n_iter, n_iter_subs, xeta, n_batch, modelfile=None,
                 weightfile=None, node_frac=1, n_walks_per_node=5, len_rw=2):
        ''' Initialize the SDNE class

        Args:
            d: dimension of the embedding
            beta: penalty parameter in matrix B of 2nd order objective
            alpha: weighing hyperparameter for 1st order objective
            nu1: L1-reg hyperparameter
            nu2: L2-reg hyperparameter
            K: number of hidden layers in encoder/decoder
            n_units: vector of length K-1 containing #units in hidden layers of encoder/decoder, not including the units in the embedding layer
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
        '''
        super().__init__(d)
        self._method_name = 'sdne' # embedding method name
        self._d = d
        self._Y = None # embedding
        self._Ys = None # embeddings
        self._beta = beta
        self._alpha = alpha
        self._nu1 = nu1
        self._nu2 = nu2
        self._K = K
        self._n_units = n_units
        self._actfn = 'relu' # We use relu instead of sigmoid from the paper, to avoid vanishing gradients and allow correct layer deepening
        self._rho = rho
        self._n_iter = n_iter
        self._n_iter_subs = n_iter_subs
        self._xeta = xeta
        self._n_batch = n_batch
        self._modelfile = modelfile
        self._weightfile = weightfile
        self._node_frac = node_frac
        self._n_walks_per_node = n_walks_per_node
        self._len_rw = len_rw
        self._num_iter = n_iter # max number of iterations during sgd (variable)
        # self._node_num is number of nodes: initialized later in learn_embedding()
        # self._encoder is the vertex->embedding model
        # self._decoder is the embedding->vertex model
        # self._autocoder is the vertex->(vertex,embedding) model
        # self._model is the SDNE model to be trained (uses self._autoencoder internally)


    def get_method_name(self):
        return self._method_name

    def get_method_summary(self):
        return '%s_%d' % (self._method_name, self._d)

    def learn_embedding(self, graph, prevStepInfo=True, loadfilesuffix=None, savefilesuffix=None, subsample=False):
        if subsample:
            S = graph_util.randwalk_DiGraph_to_adj(graph, node_frac=self._node_frac, n_walks_per_node=self._n_walks_per_node, len_rw=self._len_rw)
        else:
            S = graph_util.transform_DiGraph_to_adj(graph)
        if not np.allclose(S.T, S):
            print("SDNE only works for symmetric graphs! Making the graph symmetric")
        S = (S + S.T)/2					# enforce S is symmetric
        S -= np.diag(np.diag(S))		# enforce diagonal = 0
        self._node_num = S.shape[0]
        n_edges = np.count_nonzero(S)	# Double counting symmetric edges deliberately to maintain autoencoder symmetry
        # Create matrix B
        B = np.ones(S.shape)
        B[S != 0] = self._beta

        # compute degree of each node
        deg = np.sum(S!=0, 1)



        # Generate encoder, decoder and autoencoder
        if(prevStepInfo):
            self._num_iter = self._n_iter_subs
            # Load previous encoder, decoder and weights
            if loadfilesuffix is None:
                encoder_prev = loadmodel(self._modelfile[0])
                decoder_prev = loadmodel(self._modelfile[1])
                loadweights(encoder_prev, self._weightfile[0])
                loadweights(decoder_prev, self._weightfile[1])
            else:
                encoder_prev = loadmodel('encoder_model_'+loadfilesuffix+'.json')
                decoder_prev = loadmodel('decoder_model_'+loadfilesuffix+'.json')
                loadweights(encoder_prev, 'encoder_weights_'+loadfilesuffix+'.hdf5')
                loadweights(decoder_prev, 'decoder_weights_'+loadfilesuffix+'.hdf5')
            # Size consistency is being assumed for encoder and decoder, and won't be checked for!

            # Check graph for size changes from previous step
            prev_node_num = encoder_prev.layers[0].input_shape[1]
            # Note: We only assume node addition. No nodes should be deleted from the graph at any time step.
            if(self._node_num > prev_node_num):
                n_units_prev = [encoder_prev.layers[i].output_shape[1] for i in range(len(encoder_prev.layers))]
                # Get new number of units in each layer
                n_units = get_new_model_shape(n_units_prev, self._node_num, self._rho, deepen=False)
                self._K = len(n_units) - 1
                self._n_units = n_units[1:-1]
                self._encoder = get_encoder(self._node_num, n_units[-1], self._K, self._n_units, self._nu1, self._nu2, self._actfn)
                self._decoder = get_decoder(self._node_num, n_units[-1], self._K, self._n_units, self._nu1, self._nu2, self._actfn)
                self._autoencoder = get_autoencoder(self._encoder, self._decoder)
                # First do widening for encoder layers
                [w1, b1] = [None, None]
                for i in range(0, len(n_units_prev) - 1):
                    if(n_units[i] > n_units_prev[i]):
                        if i == 0:
                            w2, b2 = encoder_prev.layers[i+1].get_weights()
                            final_w1, final_b1, w1 = widernet(None, None, w2, n_units[i], n_units_prev[i])
                            b1 = b2
                        else:
                            w2, b2 = encoder_prev.layers[i+1].get_weights()
                            final_w1, final_b1, w1 = widernet(w1, b1, w2, n_units[i], n_units_prev[i])
                            b1 = b2
                            self._encoder.layers[i].set_weights([final_w1, final_b1])
                    else:
                        if i > 0:
                            self._encoder.layers[i].set_weights([w1, b1])
                        [w1, b1] = encoder_prev.layers[i+1].get_weights()
                # Next do deepening for encoder layers
                last_layer = len(n_units_prev) - 2
                extra_depth = len(n_units) - len(n_units_prev)
                [temp_w, temp_b] = [w1, b1]
                while(extra_depth > 0):
                    w1, temp_w, b1, temp_b = deepernet(n_units[last_layer], n_units[last_layer+1], n_units_prev[-1], temp_w, temp_b)
                    self._encoder.layers[last_layer+1].set_weights([w1, b1])
                    last_layer += 1
                    extra_depth -= 1
                self._encoder.layers[last_layer+1].set_weights([temp_w, temp_b])
                # Next do deepening for decoder layers
                n_units.reverse()
                n_units_prev.reverse()
                last_layer = 0
                extra_depth = len(n_units) - len(n_units_prev)
                [temp_w, temp_b] = decoder_prev.layers[last_layer+1].get_weights()
                while(extra_depth > 0):
                    w1, temp_w, b1, temp_b = deepernet(n_units[last_layer], n_units[last_layer+1], n_units_prev[1], temp_w, temp_b)
                    self._decoder.layers[last_layer+1].set_weights([w1, b1])
                    last_layer += 1
                    extra_depth -= 1
                # Next do widening for decoder layers
                extra_depth = len(n_units) - len(n_units_prev)
                [w1, b1] = [temp_w, temp_b]
                for i in range(last_layer + 1, len(n_units) - 1):
                    if(n_units[i] > n_units_prev[i - extra_depth]):
                        w2, b2 = decoder_prev.layers[i+1 - extra_depth].get_weights()
                        final_w1, final_b1, w1 = widernet(w1, b1, w2, n_units[i], n_units_prev[i - extra_depth])
                        b1 = b2
                        self._decoder.layers[i].set_weights([final_w1, final_b1])
                    else:
                        self._decoder.layers[i].set_weights([w1, b1])
                        w1, b1 = decoder_prev.layers[i+1 - extra_depth].get_weights()
                if(n_units[len(n_units) - 1] > n_units_prev[len(n_units) - 1 - extra_depth]):
                    final_w1, final_b1, w1 = widernet(w1, b1, None, n_units[len(n_units) - 1], n_units_prev[len(n_units) - 1 - extra_depth])
                    self._decoder.layers[len(n_units) - 1].set_weights([final_w1, final_b1])
                else:
                    self._decoder.layers[len(n_units) - 1].set_weights([w1, b1])
            else:
                # If no new nodes, then just initialize with previous models
                self._encoder = encoder_prev
                self._decoder = decoder_prev
                self._autoencoder = get_autoencoder(self._encoder, self._decoder)
        else:
            self._num_iter = self._n_iter
            # If cannot use previous step information, initialize new models
            self._encoder = get_encoder(self._node_num, self._d, self._K, self._n_units, self._nu1, self._nu2, self._actfn)
            self._decoder = get_decoder(self._node_num, self._d, self._K, self._n_units, self._nu1, self._nu2, self._actfn)
            self._autoencoder = get_autoencoder(self._encoder, self._decoder)

        # Initialize self._model
        # Input
        x_in = Input(shape=(2*self._node_num,), name='x_in')
        x1 = Lambda(lambda x: x[:,0:self._node_num], output_shape=(self._node_num,))(x_in)
        x2 = Lambda(lambda x: x[:,self._node_num:2*self._node_num], output_shape=(self._node_num,))(x_in)
        # Process inputs
        [x_hat1, y1] = self._autoencoder(x1)
        [x_hat2, y2] = self._autoencoder(x2)
        # Outputs
        x_diff1 = Subtract()([x_hat1, x1])
        # x_diff1 = merge([x_hat1, x1], mode=lambda (a,b): a - b, output_shape=lambda L: L[1])
        # x_diff2 = merge([x_hat2, x2], mode=lambda (a,b): a - b, output_shape=lambda L: L[1])
        x_diff2 = Subtract()([x_hat2, x2])
        # y_diff = merge([y2, y1], mode=lambda (a,b): a - b, output_shape=lambda L: L[1])
        y_diff = Subtract()([y2, y1])

        # Objectives
        def weighted_mse_x(y_true, y_pred):
            ''' Hack: This fn doesn't accept additional arguments. We use y_true to pass them.
                y_pred: Contains x_hat - x
                y_true: Contains [b, deg]
            '''
            return KBack.sum(KBack.square(y_pred * y_true[:,0:self._node_num]), axis=-1)/y_true[:,self._node_num]
        def weighted_mse_y(y_true, y_pred):
            ''' Hack: This fn doesn't accept additional arguments. We use y_true to pass them.
                y_pred: Contains y2 - y1
                y_true: Contains s12
            '''
            min_batch_size = y_true.shape[0]
            return KBack.sum(KBack.square(y_pred), axis=-1).reshape([min_batch_size, 1]) * y_true

        # Model
        self._model = Model(input=x_in, output=[x_diff1, x_diff2, y_diff])
        sgd = SGD(lr=self._xeta, decay=1e-5, momentum=0.99, nesterov=True)
        # adam = Adam(lr=self._xeta, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        self._model.compile(optimizer=sgd, loss=[weighted_mse_x, weighted_mse_x, weighted_mse_y], loss_weights=[1, 1, self._alpha])
        # self._model.compile(optimizer=adam, loss=[weighted_mse_x, weighted_mse_x, weighted_mse_y], loss_weights=[1, 1, self._alpha])

        # Structure data in the correct format for the SDNE model
        # InData format: [x1, x2]
        # OutData format: [b1, b2, s12, deg1, deg2]
        data_chunk_size = MAX_CHUNK_SIZE
        InData = np.zeros((data_chunk_size, 2*self._node_num))
        OutData = np.zeros((data_chunk_size, 2*self._node_num + 3))
        for epoch_num in range(self._num_iter):
            print('EPOCH %d/%d' % (epoch_num, self._num_iter))
            e = 0
            k = 0
            for i in range(self._node_num):
                for j in range(self._node_num):
                    if(S[i,j] != 0):
                        temp = np.append(S[i,:], S[j,:])
                        InData[k,:] = temp
                        temp = np.append(np.append(np.append(np.append(B[i,:], B[j,:]), S[i,j]), deg[i]), deg[j])
                        OutData[k,:] = temp
                        e += 1
                        k += 1
                        if k == data_chunk_size:
                            self._model.fit(InData, [ np.append(OutData[:,0:self._node_num], np.reshape(OutData[:,2*self._node_num+1], [data_chunk_size, 1]), axis=1),
                                np.append(OutData[:,self._node_num:2*self._node_num], np.reshape(OutData[:,2*self._node_num+2], [data_chunk_size, 1]), axis=1),
                                OutData[:,2*self._node_num] ], nb_epoch=1, batch_size=self._n_batch, shuffle=True, verbose=1)
                            k = 0
            if k > 0:
                self._model.fit(InData[:k, :], [ np.append(OutData[:k,0:self._node_num], np.reshape(OutData[:k,2*self._node_num+1], [k, 1]), axis=1),
                    np.append(OutData[:k,self._node_num:2*self._node_num], np.reshape(OutData[:k,2*self._node_num+2], [k, 1]), axis=1),
                    OutData[:k,2*self._node_num] ], nb_epoch=1, batch_size=self._n_batch, shuffle=True, verbose=1)


        # Get embedding for all points
        _, self._Y = self._autoencoder.predict(S, batch_size=self._n_batch)

        # Save the autoencoder and its weights
        if(self._weightfile is not None):
            saveweights(self._encoder, self._weightfile[0])
            saveweights(self._decoder, self._weightfile[1])
        if(self._modelfile is not None):
            savemodel(self._encoder, self._modelfile[0])
            savemodel(self._decoder, self._modelfile[1])
        if(savefilesuffix is not None):
            saveweights(self._encoder, 'encoder_weights_'+savefilesuffix+'.hdf5')
            saveweights(self._decoder, 'decoder_weights_'+savefilesuffix+'.hdf5')
            savemodel(self._encoder, 'encoder_model_'+savefilesuffix+'.json')
            savemodel(self._decoder, 'decoder_model_'+savefilesuffix+'.json')
            # Save the embedding
            np.savetxt('embedding_'+savefilesuffix+'.txt', self._Y)

        return self._Y

    def learn_embeddings(self, graphs, prevStepInfo = False, loadsuffixinfo=None, savesuffixinfo=None, subsample=False):
        if loadsuffixinfo is None:
            if savesuffixinfo is None:
                Y = self.learn_embedding(graphs[0], prevStepInfo, subsample=subsample)
            else:
                Y = self.learn_embedding(graphs[0], prevStepInfo, savefilesuffix=savesuffixinfo+'_'+str(1), subsample=subsample)
        else:
            if savesuffixinfo is None:
                Y = self.learn_embedding(graphs[0], prevStepInfo, loadfilesuffix=loadsuffixinfo+'_'+str(1), subsample=subsample)
            else:
                Y = self.learn_embedding(graphs[0], prevStepInfo, loadfilesuffix=loadsuffixinfo+'_'+str(1), savefilesuffix=savesuffixinfo+'_'+str(1), subsample=subsample)
        self._Ys = [np.copy(Y)]
        for i in range(1, len(graphs)):
            print('Processing graph {0}:'.format(i+1))
            if loadsuffixinfo is None:
                if savesuffixinfo is None:
                    Y = self.learn_embedding(graphs[i], True, subsample=subsample)
                else:
                    Y = self.learn_embedding(graphs[i], True, savefilesuffix=savesuffixinfo+'_'+str(i+1), subsample=subsample)
            else:
                if savesuffixinfo is None:
                    Y = self.learn_embedding(graphs[i], True, loadfilesuffix=loadsuffixinfo+'_'+str(i+1), subsample=subsample)
                else:
                    Y = self.learn_embedding(graphs[i], True, loadfilesuffix=loadsuffixinfo+'_'+str(i+1), savefilesuffix=savesuffixinfo+'_'+str(i+1), subsample=subsample)
            self._Ys.append(np.copy(Y))
        return self._Ys

    def get_embedding(self, filesuffix):
        return self._Y if filesuffix is None else np.loadtxt('embedding_'+filesuffix+'.txt')

    def get_embeddings(self):
        return self._Ys

    def get_edge_weight(self, i, j, embed=None, filesuffix=None):
        if embed is None:
            if filesuffix is None:
                embed = self._Y
            else:
                embed = np.loadtxt('embedding_'+filesuffix+'.txt')
        if i == j:
            return 0
        else:
            S_hat = self.get_reconst_from_embed(embed[(i,j),:], filesuffix)
            return (S_hat[i,j] + S_hat[j,i])/2

    def get_reconstructed_adj(self, embed=None, filesuffix=None):
        if embed is None:
            if filesuffix is None:
                embed = self._Y
            else:
                embed = np.loadtxt('embedding_'+filesuffix+'.txt')
        S_hat = self.get_reconst_from_embed(embed, filesuffix)
        return graphify(S_hat)

    def get_reconst_from_embed(self, embed, filesuffix=None):
        if filesuffix is None:
            return self._decoder.predict(embed, batch_size=self._n_batch)
        else:
            try:
                decoder = model_from_json(open('decoder_model_'+filesuffix+'.json').read())
            except:
                print('Error reading file: {0}. Cannot load previous model'.format('decoder_model_'+filesuffix+'.json'))
                exit()
            try:
                decoder.load_weights('decoder_weights_'+filesuffix+'.hdf5')
            except:
                print('Error reading file: {0}. Cannot load previous weights'.format('decoder_weights_'+filesuffix+'.hdf5'))
                exit()
            return decoder.predict(embed, batch_size=self._n_batch)

if __name__ == '__main__':
    parser = ArgumentParser(description='Learns node embeddings for a sequence of graph snapshots')
    parser.add_argument('-t', '--testDataType', help='Type of data to test the code')
    if len(sys.argv) < 2:
        sys.argv = [sys.argv[0], '-h']
    args = parser.parse_args()

    if args.testDataType == 'sbm_rp':
        node_num = 10000
        community_num = 500
        node_change_num = 100
        length = 2
        dynamic_sbm_series = dynamic_SBM_graph.get_random_perturbation_series(node_num, community_num, length, node_change_num)
        dynamic_embedding = SDNE(d=100, beta=5, alpha=1e-5, nu1=1e-6, nu2=1e-6, K=3, n_units=[500, 300,], rho=0.3, n_iter=30, n_iter_subs=5, xeta=0.01, n_batch=500, modelfile=['./intermediate/enc_model.json', './intermediate/dec_model.json'], weightfile=['./intermediate/enc_weights.hdf5', './intermediate/dec_weights.hdf5'], node_frac=1, n_walks_per_node=10, len_rw=2)
        dynamic_embedding.learn_embeddings([g[0] for g in dynamic_sbm_series], False, subsample=False)
        plot_dynamic_sbm_embedding.plot_dynamic_sbm_embedding(dynamic_embedding.get_embeddings(), dynamic_sbm_series)
        plt.savefig('result/visualization_sdne_rp.png')
        plt.show()
    elif args.testDataType == 'sbm_cd':
        node_num = 10000
        community_num = 500
        node_change_num = 100
        length = 2
        dynamic_sbm_series = dynamic_SBM_graph.get_community_diminish_series(node_num, community_num, length, 1, node_change_num)
        dynamic_embedding = SDNE(d=100, beta=5, alpha=1e-5, nu1=1e-6, nu2=1e-6, K=3, n_units=[500, 300,], rho=0.3, n_iter=30, n_iter_subs=5, xeta=0.01, n_batch=500, modelfile=['./intermediate/enc_model.json', './intermediate/dec_model.json'], weightfile=['./intermediate/enc_weights.hdf5', './intermediate/dec_weights.hdf5'], node_frac=1, n_walks_per_node=10, len_rw=2)
        dynamic_embedding.learn_embeddings([g[0] for g in dynamic_sbm_series], False, subsample=False)
        plot_dynamic_sbm_embedding.plot_dynamic_sbm_embedding(dynamic_embedding.get_embeddings(), dynamic_sbm_series)
        plt.savefig('result/visualization_sdne_cd.png')
        plt.show()
    else:
        dynamic_graph_series = graph_util.loadRealGraphSeries('data/real/hep-th/month_', 1, 5)
        dynamic_embedding = SDNE(d=100, beta=2, alpha=1e-6, nu1=1e-5, nu2=1e-5, K=3, n_units=[400, 250,], rho=0.3, n_iter=100, n_iter_subs=30, xeta=0.001, n_batch=500, modelfile=['./intermediate/enc_model.json', './intermediate/dec_model.json'], weightfile=['./intermediate/enc_weights.hdf5', './intermediate/dec_weights.hdf5'])
        dynamic_embedding.learn_embeddings(dynamic_graph_series, False)