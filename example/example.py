import matplotlib.pyplot as plt
from time import time
import networkx as nx
import pickle
import numpy as np
import os

# import helper libraries
from dynamicgem.utils import graph_util, plot_util, dataprep_util
from dynamicgem.evaluation import visualize_embedding as viz
from dynamicgem.visualization import plot_dynamic_sbm_embedding
from dynamicgem.evaluation import evaluate_graph_reconstruction as gr
from dynamicgem.graph_generation import dynamic_SBM_graph as sbm

# import the methods
from dynamicgem.embedding.ae_static import AE
from dynamicgem.embedding.dynamicTriad import dynamicTriad
from dynamicgem.embedding.TIMERS import TIMERS
from dynamicgem.embedding.dynAE import DynAE
from dynamicgem.embedding.dynRNN import DynRNN
from dynamicgem.embedding.dynAERNN import DynAERNN

# Parameters for Stochastic block model graph
# Todal of 1000 nodes
node_num = 1000
# Test with two communities
community_num = 2
# At each iteration migrate 10 nodes from one community to the another
node_change_num = 10
# Length of total time steps the graph will dynamically change
length = 7
# output directory for result
outdir = './output'
intr = './intermediate'
if not os.path.exists(outdir):
    os.mkdir(outdir)
if not os.path.exists(intr):
    os.mkdir(intr)
testDataType = 'sbm_cd'
# Generate the dynamic graph
dynamic_sbm_series = list(sbm.get_community_diminish_series_v2(node_num,
                                                               community_num,
                                                               length,
                                                               1,  # comminity ID to perturb
                                                               node_change_num))
graphs = [g[0] for g in dynamic_sbm_series]
# parameters for the dynamic embedding
# dimension of the embedding
dim_emb = 128
lookback = 2

# AE Static
embedding = AE(d=dim_emb,
               beta=5,
               nu1=1e-6,
               nu2=1e-6,
               K=3,
               n_units=[500, 300, ],
               n_iter=200,
               xeta=1e-4,
               n_batch=100,
               modelfile=['./intermediate/enc_modelsbm.json',
                          './intermediate/dec_modelsbm.json'],
               weightfile=['./intermediate/enc_weightssbm.hdf5',
                           './intermediate/dec_weightssbm.hdf5'])
embs = []
t1 = time()
# ae static
for temp_var in range(length):
    emb, _ = embedding.learn_embeddings(graphs[temp_var])
    embs.append(emb)
print(embedding._method_name + ':\n\tTraining time: %f' % (time() - t1))

viz.plot_static_sbm_embedding(embs[-4:], dynamic_sbm_series[-4:])

# TIMERS
datafile = dataprep_util.prep_input_TIMERS(graphs, length, testDataType)
embedding = TIMERS(K=dim_emb,
                   Theta=0.5,
                   datafile=datafile,
                   length=length,
                   nodemigration=node_change_num,
                   resultdir=outdir,
                   datatype=testDataType)
if not os.path.exists(outdir):
    os.mkdir(outdir)
outdir_tmp = outdir + '/sbm_cd'
if not os.path.exists(outdir_tmp):
    os.mkdir(outdir_tmp)
if not os.path.exists(outdir_tmp + '/incremental'):
    os.mkdir(outdir_tmp + '/incrementalSVD')
if not os.path.exists(outdir_tmp + '/rerunSVD'):
    os.mkdir(outdir_tmp + '/rerunSVD')
if not os.path.exists(outdir_tmp + '/optimalSVD'):
    os.mkdir(outdir_tmp + '/optimalSVD')

t1 = time()
embedding.learn_embedding()
embedding.get_embedding(outdir_tmp, 'optimalSVD')
print(embedding._method_name + ':\n\tTraining time: %f' % (time() - t1))
embedding.plotresults(dynamic_sbm_series)

# dynAE
embedding = DynAE(d=dim_emb,
                  beta=5,
                  n_prev_graphs=lookback,
                  nu1=1e-6,
                  nu2=1e-6,
                  n_units=[500, 300, ],
                  rho=0.3,
                  n_iter=250,
                  xeta=1e-4,
                  n_batch=100,
                  modelfile=['./intermediate/enc_model_dynAE.json',
                             './intermediate/dec_model_dynAE.json'],
                  weightfile=['./intermediate/enc_weights_dynAE.hdf5',
                              './intermediate/dec_weights_dynAE.hdf5'],
                  savefilesuffix="testing")
embs = []
t1 = time()
for temp_var in range(lookback + 1, length + 1):
    emb, _ = embedding.learn_embeddings(graphs[:temp_var])
    embs.append(emb)
print(embedding._method_name + ':\n\tTraining time: %f' % (time() - t1))
plt.figure()
plt.clf()
plot_dynamic_sbm_embedding.plot_dynamic_sbm_embedding_v2(embs[-5:-1], dynamic_sbm_series[-5:])
plt.show()

# dynRNN
embedding = DynRNN(d=dim_emb,
                   beta=5,
                   n_prev_graphs=lookback,
                   nu1=1e-6,
                   nu2=1e-6,
                   n_enc_units=[500, 300],
                   n_dec_units=[500, 300],
                   rho=0.3,
                   n_iter=250,
                   xeta=1e-3,
                   n_batch=100,
                   modelfile=['./intermediate/enc_model_dynRNN.json',
                              './intermediate/dec_model_dynRNN.json'],
                   weightfile=['./intermediate/enc_weights_dynRNN.hdf5',
                               './intermediate/dec_weights_dynRNN.hdf5'],
                   savefilesuffix="testing")
embs = []
t1 = time()
for temp_var in range(lookback + 1, length + 1):
    emb, _ = embedding.learn_embeddings(graphs[:temp_var])
    embs.append(emb)
print(embedding._method_name + ':\n\tTraining time: %f' % (time() - t1))
plt.figure()
plt.clf()
plot_dynamic_sbm_embedding.plot_dynamic_sbm_embedding_v2(embs[-5:-1], dynamic_sbm_series[-5:])
plt.show()

# dynAERNN
embedding = DynAERNN(d=dim_emb,
                     beta=5,
                     n_prev_graphs=lookback,
                     nu1=1e-6,
                     nu2=1e-6,
                     n_aeunits=[500, 300],
                     n_lstmunits=[500, dim_emb],
                     rho=0.3,
                     n_iter=250,
                     xeta=1e-3,
                     n_batch=100,
                     modelfile=['./intermediate/enc_model_dynAERNN.json',
                                './intermediate/dec_model_dynAERNN.json'],
                     weightfile=['./intermediate/enc_weights_dynAERNN.hdf5',
                                 './intermediate/dec_weights_dynAERNN.hdf5'],
                     savefilesuffix="testing")

embs = []
t1 = time()
for temp_var in range(lookback + 1, length + 1):
    emb, _ = embedding.learn_embeddings(graphs[:temp_var])
    embs.append(emb)
print(embedding._method_name + ':\n\tTraining time: %f' % (time() - t1))
plt.figure()
plt.clf()
plot_dynamic_sbm_embedding.plot_dynamic_sbm_embedding_v2(embs[-5:-1], dynamic_sbm_series[-5:])
plt.show()

# dynamicTriad
datafile = dataprep_util.prep_input_dynTriad(graphs, length, testDataType)
embedding = dynamicTriad(niters=20,
                         starttime=0,
                         datafile=datafile,
                         batchsize=1000,
                         nsteps=length,
                         embdim=dim_emb,
                         stepsize=1,
                         stepstride=1,
                         outdir=outdir,
                         cachefn='/tmp/' + testDataType,
                         lr=0.1,
                         beta=[0.1, 0.1],
                         negdup=1,
                         datasetmod='core.dataset.adjlist',
                         trainmod='dynamicgem.dynamictriad.core.algorithm.dynamic_triad',
                         pretrain_size=length,
                         sampling_args={},
                         validation='link_reconstruction',
                         datatype=testDataType,
                         scale=1,
                         classifier='lr',
                         debug=False,
                         test='link_predict',
                         repeat=1,
                         resultdir=outdir,
                         testDataType=testDataType,
                         clname='lr',
                         node_num=node_num)
t1 = time()
embedding.learn_embedding()
print(embedding._method_name + ':\n\tTraining time: %f' % (time() - t1))
embedding.get_embedding()
embedding.plotresults(dynamic_sbm_series)
