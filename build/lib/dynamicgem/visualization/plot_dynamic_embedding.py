disp_avlbl = True
from os import environ
if 'DISPLAY' not in environ:
	disp_avlbl = False
	import matplotlib
	matplotlib.use('Agg')
import matplotlib.pyplot as plt

import networkx as nx
import numpy as np
from sklearn.manifold import TSNE
import sys
sys.path.append('./')
import pdb

from utils import graph_util
from utils import plot_util

def plot_single_step(node_pos, graph, node_labels):
	if node_labels:
		node_colors = plot_util.get_node_color(node_labels)
	else:
		node_colors = None
	node_num, embedding_dimension = node_pos.shape
	pos = {}
	for i in xrange(node_num):
		pos[i] = node_pos[i, :]

	# draw nodes
	nx.draw_networkx_nodes(graph, pos, nodelist=range(node_num), node_color=node_colors, node_size=20, with_labels=False)
	# draw all edges
	nx.draw_networkx_edges(graph, pos, width=0.1, arrows=False, alpha=0.8)
	
def plot_dynamic_embedding(nodes_pos_list, graph_series, t_steps, node_labels=None):
	length = len(t_steps)
	node_num, dimension = nodes_pos_list[0].shape

	if dimension > 2:
		print "Embedding dimension greater than 2, use tSNE to reduce it to 2"
		model = TSNE(n_components=2, random_state=42)
		nodes_pos_list = [model.fit_transform(X) for X in nodes_pos_list]

	for i in range(len(t_steps)):
		ax = plt.subplot(1, length, i+1)
		# ax.set_xlim([-50, 50])
		# ax.set_ylim([-50, 50])
		plot_single_step(nodes_pos_list[t_steps[i]], graph_series[t_steps[i]], node_labels)