import matplotlib.pyplot as plt
from time import time
import networkx as nx
try: import cPickle as pickle
except: import pickle
import numpy as np

#import helper libraries
from dynamicgem.utils      import graph_util, plot_util, dataprep_util
from dynamicgem.evaluation import visualize_embedding as viz
from dynamicgem.evaluation import evaluate_graph_reconstruction as gr
from dynamicgem.graph_generation import dynamic_SBM_graph as sbm

#import the methods
from dynamicgem.embedding.ae_static    import AE
from dynamicgem.embedding.dynamicTriad import dynamicTriad
from dynamicgem.embedding.TIMERS       import TIMERS
from dynamicgem.embedding.dynAE        import DynAE
from dynamicgem.embedding.dynRNN       import DynRNN
from dynamicgem.embedding.dynAERNN     import DynAERNN


# Parameters for Stochastic block model graph
# Todal of 1000 nodes
node_num           = 1000
# Test with two communities
community_num      = 2
# At each iteration migrate 10 nodes from one community to the another
node_change_num    = 10
# Length of total time steps the graph will dynamically change
length             = 10
# output directory for result
outdir = './output'
testDataType = 'sbm_cd'
#Generate the dynamic graph
dynamic_sbm_series = sbm.get_community_diminish_series_v2(node_num, 
                                                          community_num, 
                                                          length, 
                                                          1, #comminity ID to perturb
                                                          node_change_num)
graphs     = [g[0] for g in dynamic_sbm_series]

node_colors_arr = [None] * node_colors.shape[0]
for idx in range(node_colors.shape[0]):
    node_colors_arr[idx] = np.where(node_colors[idx, :].toarray() == 1)[1][0]

models  = []
# parameters for the dynamic embedding
# dimension of the embedding
dim_emb = 128

# Load the models you want to run
models.append(AE(d          = dim_emb, 
                 beta       = 5, 
                 nu1        = 1e-6, 
                 nu2        = 1e-6,
                 K          = 3, 
                 n_units    = [500, 300, ],
                 n_iter     = epochs, 
                 xeta       = 1e-4,
                 n_batch    = 100,
                 modelfile  = ['./intermediate/enc_modelsbm.json',
                             './intermediate/dec_modelsbm.json'],
                 weightfile = ['./intermediate/enc_weightssbm.hdf5',
                             './intermediate/dec_weightssbm.hdf5']))

datafile  = dataprep_util.prep_input_dynTriad(graphs, length, testDataType)
models.append(dynamicTriad(niters     = 20,
                 starttime  = 0,
                 datafile   = datafile,
                 batchsize  = 1000,
                 nsteps     = length,
                 embdim     = dim_emb,
                 stepsize   = 1,
                 stepstride = 1,
                 outdir     = outdir,
                 cachefn    = '/tmp/'+ testDataType,
                 lr         = 0.1,
                 beta       = [0.1,0.1],
                 negdup     = 1,
                 datasetmod = 'core.dataset.adjlist',
                 trainmod   = 'dynamicgem.dynamictriad.core.algorithm.dynamic_triad',
                 pretrain_size = length,
                 sampling_args = {},
                 validation = 'link_reconstruction',
                 datatype   = testDataType,
                 scale      = 1,
                 classifier = 'lr',
                 debug      = False,
                 test       = 'link_predict',
                 repeat     = 1,
                 resultdir  = outdir,
                 testDataType = testDataType,
                 clname       = 'lr',
                 node_num     = node_num ))

datafile  = dataprep_util.prep_input_TIMERS(graphs, length, testDataType) 
models.append(TIMERS(K        = dim_emb, 
                 Theta         = 0.5, 
                 datafile      = datafile,
                 length        =  length,
                 nodemigration = node_change_num,
                 resultdir     = output,
                 datatype      = testDataType))

models.append(DynAE(d           = dim_emb,
                 beta           = 5,
                 n_prev_graphs  = 2,
                 nu1            = 1e-6,
                 nu2            = 1e-6,
                 n_units        = [500, 300,],
                 rho            = 0.3,
                 n_iter         = 250,
                 xeta           = 1e-4,
                 n_batch        = 100,
                 modelfile      = ['./intermediate/enc_model_dynAE.json', 
                                   './intermediate/dec_model_dynAE.json'],
                 weightfile     = ['./intermediate/enc_weights_dynAE.hdf5', 
                                   './intermediate/dec_weights_dynAE.hdf5'],
                 savefilesuffix = "testing" ))

models.append(DynRNN(d        = dim_emb,
                beta           = 5,
                n_prev_graphs  = 2,
                nu1            = 1e-6,
                nu2            = 1e-6,
                n_enc_units    = [500,300],
                n_dec_units    = [500,300],
                rho            = 0.3,
                n_iter         = 250,
                xeta           = 1e-3,
                n_batch        = 100,
                modelfile      = ['./intermediate/enc_model_dynRNN.json', 
                                  './intermediate/dec_model_dynRNN.json'],
                weightfile     = ['./intermediate/enc_weights_dynRNN.hdf5', 
                                  './intermediate/dec_weights_dynRNN.hdf5'],
                savefilesuffix = "testing"  ))

models.append(DynAERNN(d   = dim_emb,
            beta           = 5,
            n_prev_graphs  = 2,
            nu1            = 1e-6,
            nu2            = 1e-6,
            n_aeunits      = [500, 300],
            n_lstmunits    = [500,dim_emb],
            rho            = 0.3,
            n_iter         = 250,
            xeta           = 1e-3,
            n_batch        = 100,
            modelfile      = ['./intermediate/enc_model_dynAERNN.json', 
                              './intermediate/dec_model_dynAERNN.json'],
            weightfile     = ['./intermediate/enc_weights_dynAERNN.hdf5', 
                              './intermediate/dec_weights_dynAERNN.hdf5'],
            savefilesuffix = "testing"
        ))

# For each model, learn the embedding and evaluate on graph reconstruction and visualization
for embedding in models:
    print ('Num nodes: %d, num edges: %d' % (G.number_of_nodes(), G.number_of_edges()))
    t1 = time()
    # Learn embedding - accepts a networkx graph or file with edge list
    Y, t = embedding.learn_embedding(graph=G, edge_f=None, is_weighted=True, no_python=True)
    print (embedding._method_name+':\n\tTraining time: %f' % (time() - t1))
    # Evaluate on graph reconstruction
    MAP, prec_curv, err, err_baseline = gr.evaluateStaticGraphReconstruction(G, embedding, Y, None)
    #---------------------------------------------------------------------------------
    print(("\tMAP: {} \t precision curve: {}\n\n\n\n"+'-'*100).format(MAP,prec_curv[:5]))
    #---------------------------------------------------------------------------------
    # Visualize
    viz.plot_embedding2D(embedding.get_embedding(), di_graph=G, node_colors=node_colors_arr)
    plt.show()
    plt.clf()
