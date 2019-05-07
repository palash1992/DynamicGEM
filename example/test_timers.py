import os
from dynamicgem.utils import dataprep_util
from dynamicgem.embedding.TIMERS import TIMERS
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
    if not os.path.exists(outdir_tmp + '/incrementalSVD'):
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


if __name__ == '__main__':
    main()
