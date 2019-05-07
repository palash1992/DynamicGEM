import os
from dynamicgem.evaluation import visualize_embedding as viz
from dynamicgem.embedding.ae_static import AE
from dynamicgem.graph_generation import dynamic_SBM_graph as sbm
from time import time

def main():
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
    dim_emb = 8
    lookback = 2

    # AE Static
    embedding = AE(d=dim_emb,
                   beta=5,
                   nu1=1e-6,
                   nu2=1e-6,
                   K=3,
                   n_units=[500, 300],
                   n_iter=5,
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


if __name__ == '__main__':
    main()
