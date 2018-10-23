import networkx as nx
import os
import numpy as np

files=[file for file in os.listdir('./test_data/AS/as-733') if '.gpickle' in file]
length=len(files)
degree=[]
nodes=[]
edges=[]
for i in range(length):
    G=nx.read_gpickle('./test_data/AS/as-733/month_'+str(i+1)+'_graph.gpickle')
    print("Nodes:Edges= ", len(G.nodes()) ,':',len(G.edges()))
    nodes.append(len(G.nodes()))
    edges.append(len(G.edges()))
print("nodes, max:min=",np.max(nodes),':',np.min(nodes))    
print("edges, max:min=",np.max(edges),':',np.min(edges))
print(length)
    # print("Edges: ", len(G.edges())	)
    # print("degree:", G.degree())
    # degree.append(G.degree())