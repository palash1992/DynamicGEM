import pickle
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import operator
import sys
sys.path.append('./')
from dynamicgem.utils import graph_util
import random


class SBMGraph(object):
    def __init__(self, node_num, community_num, community_id=1, nodes_to_purturb=5,inblock_prob=0.2, crossblock_prob=0.01, community_size=None):
        self._node_num = node_num
        self._community_num = community_num
        self._community_id = community_id
        self._nodes_to_purturb=nodes_to_purturb
        self._graph = None
        self._chngnodes = None
        self.set_mtx_B(inblock_prob, crossblock_prob)
        self.sample_node_community(community_size)

    def set_mtx_B(self, inblock_prob=0.1, crossblock_prob=0.01):
        self._B = np.ones((self._community_num, self._community_num)) * crossblock_prob
        for i in range(self._community_num):
            self._B[i, i] = inblock_prob
        return self._B

    def set_mtx_B_v2(self, inblock_prob=0.1, crossblock_prob=0.01):
        self._B = np.ones((self._community_num, self._community_num)) * crossblock_prob
        for i in range(self._community_num):
            self._B[i, i] = inblock_prob
        return self._B


    def sample_node_community(self, community_size=None):
        if community_size is None:
            community_size = np.random.multinomial(self._node_num, [1.0 / self._community_num] * self._community_num)
        print("community_size", community_size)    

        self._node_community = []
        assert(len(community_size) == self._community_num)
        for i, size in enumerate(community_size):
            self._node_community += [i] * size

    def sample_graph(self):
        self._graph = nx.DiGraph()
        # add nodes
        self._graph.add_nodes_from(range(self._node_num))
        # add edges
        for i in range(self._node_num):
            for j in range(i):
                prob = self._B[self._node_community[i], self._node_community[j]]
                if np.random.uniform() <= prob:
                    self._graph.add_edge(i, j)
                    self._graph.add_edge(j, i)
        return self._graph

    def sample_graph_v2(self, inblock_prob=0.1):
        self._graph = nx.DiGraph()
        # add nodes
        self._graph.add_nodes_from(range(self._node_num))
        # add edges
        nodes2change=self._nodes_to_purturb
        print(nodes2change)
        chngnodes = []
        cnt=0
        cFlag=False
        for i in range(self._node_num):
            
            for j in range(i):
                prob = self._B[self._node_community[i], self._node_community[j]]

                if self._node_community[i] != self._node_community[j] and cnt<nodes2change and self._node_community[i]==self._community_id:
                    prob = inblock_prob*2
                    cFlag=True
                    
                if np.random.uniform() <= prob:
                    self._graph.add_edge(i, j)
                    self._graph.add_edge(j, i)
            if cFlag:
                cFlag=False 
                chngnodes.append(i)
                cnt+=1

        chngnodes=np.unique(chngnodes)
        print(chngnodes)
        self._chngnodes=chngnodes
        return self._graph, self._chngnodes

    def sample_graph_v3(self, inblock_prob=0.1):
        self._graph = nx.DiGraph()
        # add nodes
        self._graph.add_nodes_from(range(self._node_num))
        # add edges
        nodes2change=self._nodes_to_purturb

        # G_cen = nx.degree_centrality(self._graph) 
        # G_cen = sorted(G_cen.items(), key=operator.itemgetter(1),reverse = False)
        # chngnodes=[]
        # count = 0
        # i     = 0
        # while count<self._nodes_to_purturb:
        #     if self._node_community[G_cen[i][0]]==self._community_id:
        #         chngnodes.append(G_cen[i][0])
        #         count+=1
        #     i+=1

        nodes= [i for i in range(self._node_num) if self._node_community[i]==self._community_id]

        chngnodes =random.sample(nodes, self._nodes_to_purturb)
        chngedges = {}
        for i in range(self._node_num):
            
            for j in range(i):
                prob = self._B[self._node_community[i], self._node_community[j]]

                if self._node_community[i] != self._node_community[j] and (i in chngnodes or j in chngnodes):
                    if i in chngnodes:
                        if i not in chngedges.keys():
                            chngedges[i]=1
                        else:
                            chngedges[i]+=1
                        if chngedges[i]<31:    
                            self._graph.add_edge(i, j)
                            self._graph.add_edge(j, i) 
                        else:
                            if np.random.uniform() <= prob:
                                self._graph.add_edge(i, j)
                                self._graph.add_edge(j, i)       
                    else:
                        if j not in chngedges.keys():
                            chngedges[j]=1
                        else:
                            chngedges[j]+=1
                        if chngedges[j]<31:    
                            self._graph.add_edge(i, j)
                            self._graph.add_edge(j, i)
                        else:
                            if np.random.uniform() <= prob:
                                self._graph.add_edge(i, j)
                                self._graph.add_edge(j, i)
                else:    
                    if np.random.uniform() <= prob:
                        self._graph.add_edge(i, j)
                        self._graph.add_edge(j, i)

        # print(chngnodes)
        self._chngnodes=chngnodes
        return self._graph, self._chngnodes 

    def sample_graph_motiv(self, inblock_prob=0.1):
        self._graph = nx.DiGraph()
        # add nodes
        self._graph.add_nodes_from(range(self._node_num))
        # add edges
        nodes2change=self._nodes_to_purturb

        # G_cen = nx.degree_centrality(self._graph) 
        # G_cen = sorted(G_cen.items(), key=operator.itemgetter(1),reverse = False)
        # chngnodes=[]
        # count = 0
        # i     = 0
        # while count<self._nodes_to_purturb:
        #     if self._node_community[G_cen[i][0]]==self._community_id:
        #         chngnodes.append(G_cen[i][0])
        #         count+=1
        #     i+=1

        nodes= [i for i in range(self._node_num) if self._node_community[i]==self._community_id]

        chngnodes =random.sample(nodes, self._nodes_to_purturb)
        chngedges = {}
        for i in range(self._node_num):
            
            for j in range(i):
                prob = self._B[self._node_community[i], self._node_community[j]]
                if np.random.uniform() <= prob:
                        self._graph.add_edge(i, j)
                        self._graph.add_edge(j, i)

        pos=nx.spring_layout(self._graph)                
        color=['#4B0082','#FFD700']
        # plt.figure()
        plt.subplot(221)
        nodes_draw=nx.draw_networkx_nodes(self._graph,pos,node_size=60,node_color=[color[self._node_community[p]] for p in self._graph.nodes()])
        nx.draw_networkx_edges(self._graph,pos,arrows=False,width=0.2,alpha=0.5,edge_color='#6B6B6B')
        nodes_draw.set_edgecolor('w') 
        plt.title("(a)",fontsize=10) 

        for i in range(self._node_num):
        
            for j in range(i):
                if self._node_community[i] != self._node_community[j] and (i in chngnodes or j in chngnodes):
                    if i in chngnodes:
                        if i not in chngedges.keys():
                            chngedges[i]=1
                        else:
                            chngedges[i]+=1
                        if chngedges[i]<6:    
                            self._graph.add_edge(i, j)
                            self._graph.add_edge(j, i) 
                    else:
                        if j not in chngedges.keys():
                            chngedges[j]=1
                        else:
                            chngedges[j]+=1
                        if chngedges[j]<6:    
                            self._graph.add_edge(i, j)
                            self._graph.add_edge(j, i)

        # print(chngnodes)
        self._chngnodes=chngnodes
        nodes_draw=nx.draw_networkx_nodes(self._graph, 
                                      pos, 
                                      nodelist=chngnodes, 
                                      node_color='r', 
                                      node_size=80, 
                                      with_labels=False)
        nodes_draw.set_edgecolor('k')
        self._pos=pos
        return self._graph, self._chngnodes        


if __name__ == '__main__':
    # settings = [(128, 3), (256, 3), (512, 4), (1024, 5)]
    settings = [(100,2)]
    color=['r','g']
    # for (node_num, community_num) in settings:
    my_graph = SBMGraph(1000, 2,1,10)
    my_graph.sample_graph_v3()
    nx.write_graphml(my_graph._graph, "./graphs/SBM.graphml")

    # G=my_graph.sample_graph()
    # pos=nx.spring_layout(G)
    # G_cen = nx.degree_centrality(G) 
    # for i in G_cen.keys():
    #     G_cen[i]=str("{0:.3f}".format(G_cen[i])) 
    # plt.figure(1)
    # plt.subplot(121)
    # nx.draw_networkx_nodes(G,pos,node_size=800,node_color=[color[my_graph._node_community[p]] for p in G.nodes()])
    # nx.draw_networkx_edges(G,pos,arrows=False,width=1.0,alpha=0.5)
    # nx.draw_networkx_labels(G,pos,G_cen,font_size=8)
    # plt.subplot(122)
    # nx.draw_networkx(G,pos,with_labels=True,arrows=False,node_color=[color[my_graph._node_community[p]] for p in G.nodes()])

    # my_graph.sample_graph_v3()
    # print(my_graph._chngnodes)
    # pos=nx.spring_layout(my_graph._graph)
    # G_cen= nx.degree_centrality(my_graph._graph)
    # for i in G_cen.keys():
    #     G_cen[i]=str("{0:.3f}".format(G_cen[i])) 
    # plt.figure(1) 
    # plt.subplot(121)
    # nx.draw_networkx_nodes(my_graph._graph,pos,node_size=800,node_color=[color[my_graph._node_community[p]] for p in my_graph._graph.nodes()])
    # nx.draw_networkx_edges(my_graph._graph,pos,arrows=False,width=1.0,alpha=0.5)
    # nx.draw_networkx_labels(my_graph._graph,pos,G_cen,font_size=8)
    # plt.subplot(122)
    # nx.draw_networkx(my_graph._graph,pos,with_labels=True,arrows=False,node_color=[color[my_graph._node_community[p]] for p in my_graph._graph.nodes()])


    # plt.show()
        # save graph
        # file_name = "data/synthetic/static_SBM/SBM_%d_%d_graph.gpickle" % (node_num, community_num)
        # nx.write_gpickle(my_graph._graph, file_name)
        # file_name = "data/synthetic/static_SBM/SBM_%d_%d_node.pkl" % (node_num, community_num)
        # with open(file_name, 'wb') as fp:
        #     pickle.dump(my_graph._node_community, fp)

    # test load
    # file_name = "data/synthetic/static_SBM/SBM_%d_%d_graph.gpickle" % (1024, 5)
    # G = nx.read_gpickle(file_name)
    # plt.matshow(graph_util.transform_DiGraph_to_adj(G))
    # plt.show()