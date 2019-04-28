import networkx as nx
import os
import matplotlib.pyplot as plt
import sys
import numpy as np

sys.path.append('./')
from dynamicgem.graph_generation import SBM_graph
from dynamicgem.graph_generation import dynamic_SBM_graph

outdir = '../data'


def prep_input_dynTriad(graphs, length, dataname):
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    dirname = outdir + '/' + dataname

    if not os.path.exists(dirname):
        os.mkdir(dirname)

    dirname = dirname + '/' + 'dynTriad'

    if not os.path.exists(dirname):
        os.mkdir(dirname)

    for i in range(length):
        G = graphs[i]

        outName = dirname + '/' + str(i)

        with open(outName, "w") as text_file:

            for i, adj in enumerate(nx.generate_adjlist(G)):

                text_file.write(str(i))
                for nodes in adj.split(" "):
                    weight = 1.0
                    text_file.write(" ")
                    text_file.write(str(nodes))
                    text_file.write(" ")
                    text_file.write(str(weight))
                text_file.write("\n")
    return os.path.abspath(dirname)


def prep_input_TIMERS(graphs, length, dataname):
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    dirname = outdir + '/' + dataname

    if not os.path.exists(dirname):
        os.mkdir(dirname)

    dirname = dirname + '/' + 'TIMERS'

    if not os.path.exists(dirname):
        os.mkdir(dirname)

    for i in range(length):
        G = graphs[i]

        outName = dirname + '/' + str(i)
        nx.write_adjlist(G, outName)

    return os.path.abspath(dirname)


def getemb_dynTriad(dir, length, K):
    allembs = []
    for i in range(length):
        fname = dir + '/' + str(i) + '.out'
        with open(fname, "r") as text_file:
            data = text_file.read().split('\n')
            # print(len(data))
            embs = np.zeros([len(data) - 1, K])
            for k, line in enumerate(data[:-1]):
                # print(line)
                vals = line.split(' ')
                node = int(vals[0])
                for j, emb in enumerate(vals[1:]):
                    embs[node][j] = float(emb)
            # print(embs, np.shape(embs))
            allembs.append(embs)
    return allembs


def getemb_TIMERS(dir, length, K, method):
    allembs = []

    fnameU = dir + '/0_U.txt'
    with open(fnameU, "r") as text_file:
        data = text_file.read().split('\n')
        # print(len(data))
        embsU = np.zeros([len(data) - 1, K])
        for k, line in enumerate(data[:-1]):
            # print(line)
            vals = line.split(' ')[1:]
            for j, embU in enumerate(vals):
                embsU[k][j] = float(embU)

    fnameV = dir + '/0_V.txt'
    with open(fnameV, "r") as text_file:
        data = text_file.read().split('\n')
        # print(len(data))
        embsV = np.zeros([len(data) - 1, K])
        for k, line in enumerate(data[:-1]):
            # print(line)
            vals = line.split(' ')[1:]
            for j, embV in enumerate(vals):
                embsV[k][j] = float(embV)

    allembs.append(np.concatenate([embsU, embsV], axis=1))

    for i in range(1, length):
        fnameU = dir + '/' + method + '/' + str(i) + '_U.txt'
        with open(fnameU, "r") as text_file:
            data = text_file.read().split('\n')
            # print(len(data))
            embsU = np.zeros([len(data) - 1, K])
            for k, line in enumerate(data[:-1]):
                # print(line)
                vals = line.split(' ')[1:]
                for j, embU in enumerate(vals):
                    embsU[k][j] = float(embU)

        fnameV = dir + '/' + method + '/' + str(i) + '_V.txt'
        with open(fnameV, "r") as text_file:
            data = text_file.read().split('\n')
            # print(len(data))
            embsV = np.zeros([len(data) - 1, K])
            for k, line in enumerate(data[:-1]):
                # print(line)
                vals = line.split(' ')[1:]
                for j, embV in enumerate(vals):
                    embsV[k][j] = float(embV)
        allembs.append(np.concatenate([embsU, embsV], axis=-1))

    return allembs


def get_graph_academic(filepath):
    files = os.listdir(filepath)
    length = len(files)
    graphs = []

    with open(filepath + '/' + str(length - 1), 'r') as fh:
        lines = fh.read().splitlines()

    for i in range(length):
        G = nx.DiGraph()
        with open(filepath + '/' + str(i), 'r') as fh:
            lines = fh.read().splitlines()

            node_dict = {}
            for i, line in enumerate(lines):
                node = int(line.split(' ')[0])
                node_dict[node] = int(i)

            for line in lines:
                data = line.split(' ')
                if len(data) > 2:
                    u = int(data[0])
                    data = data[1:]
                    for i in range(0, len(data), 2):
                        v = int(data[i])
                        weight = float(data[i + 1])
                        G.add_edge(node_dict[u], node_dict[v], weight=weight)
                else:
                    u = int(data[0])
                    G.add_node(node_dict[u])
        graphs.append(G)

    return graphs, length


if __name__ == "__main__":
    # length=5
    # dynamic_sbm_series = dynamic_SBM_graph.get_community_diminish_series_v2(1000, 
    #                                                                         2, 
    #                                                                         length, 
    #                                                                         1, 
    #                                                                         10)
    # graphs     = [g[0] for g in dynamic_sbm_series]
    # dirname=prep_input_TIMERS(graphs,length,'sbm_cd')
    # print(dirname)

    # embs=getemb_TIMERS('./output/sbm_cd', 5,128,'incrementalSVD')
    # print(embs, np.shape(embs))
    graphs, length = get_graph_academic('./test_data/academic/adjlist')
    print(length)
    for i in range(length):
        print(i, "Nodes", len(graphs[i].nodes()), "Edges:", len(graphs[i].edges()))
