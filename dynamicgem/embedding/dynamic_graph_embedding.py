from abc import ABCMeta


class DynamicGraphEmbedding:
	__metaclass__ = ABCMeta

	def __init__(self, d):
		"""Initialize the Dynamic Graph Embedding class

		Args:
			d: dimension of embedding
		"""
		pass

	def get_method_name(self):
		""" Returns the name for the embedding method

		Return:
			The name of embedding
		"""
		return ''

	def get_method_summary(self):
		""" Returns the summary for the embedding include method name and paramater setting

		Return:
			A summary string of the method
		"""
		return ''

	def learn_embeddings(self, graphs):
		"""Learning the graph embedding from the adjcency matrix.

		Args:
			graphs: the graphs to embed in networkx DiGraph format
		"""
		pass

	def get_embeddings(self):
		""" Returns the learnt embeddings

		Return:
			A list of numpy arrays of size #nodes * d
		"""
		pass

	def get_edge_weight(self, i, j):
		"""Compute the weight for edge between node i and node j

		Args:
			i, j: two node id in the graph for embedding
		Returns:
			A single number represent the weight of edge between node i and node j

		"""
		pass

	def get_edge_weight_at_t(self, i, j, t):
		"""Compute the weight for edge between node i and node j at timestep t

		Args:
			i, j: two node id in the graph for embedding
			t: timestep
		Returns:
			Estimated weight of edge between node i and node j at timestep t

		"""
		pass

	def get_reconstructed_adjs(self):
		"""Construct the graphs from the learnt embeddings

		Returns:
			List of numpy arrays containing the reconstructed graphs.
		"""
		pass

	def predict_next_embedding(self):
		"""Predict embedding at next time step

		Returns:
			Numpy array containing the next time step's embedding.
		"""
		pass

	def predict_next_graph_adj(self):
		"""Predict graph adjacency at next time step

		Returns:
			Numpy array containing the next time step's adjacency.
		"""
		pass
