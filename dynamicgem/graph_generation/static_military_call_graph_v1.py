import cPickle as pickle
import numpy as np
import networkx as nx
import pdb

# from graph_generation import SBM_graph
import SBM_graph

class StaticMilitaryGraph(object):
	def __init__(self, n_soldiers, n_commanders, n_commander_comms, n_family_mem_per_person, inblock_prob, crossblock_prob, commander_to_soldier_p, to_family_p):
		''' Initialize the StaticMilitaryGraph

		Args:
			n_soldiers: number of soldiers
			n_commanders: number of commanders
			n_commander_comms: number of commander communities
			n_family_mem_per_person: number of family members per person
			inblock_prob: list of inblock probability for soldiers, commanders and family members
			crossblock_prob: list of crossblock probability for soldiers, commanders and family members
			commander_to_soldier_p: probability of call from commander to a soldier in his platoon
			to_family_p: probability of call from a person to his family member
		'''
		self._n_soldiers = n_soldiers
		self._n_commanders = n_commanders
		self._n_commander_comms = n_commander_comms
		self._n_family_mem_per_person = n_family_mem_per_person
		self._inblock_soldier_p = inblock_prob[0]
		self._inblock_commander_p = inblock_prob[1]
		self._inblock_family_p = inblock_prob[2]
		self._crossblock_soldier_p = crossblock_prob[0]
		self._crossblock_commander_p = crossblock_prob[1]
		self._crossblock_family_p = crossblock_prob[2]
		self._commander_to_soldier_p = commander_to_soldier_p
		self._to_family_p = to_family_p
		self._n_military = self._n_soldiers + self._n_commanders
		self._n_family_members = self._n_military*self._n_family_mem_per_person
		

	def sample_graph(self):
		''' Sample a static military graph

		'''
		soldier_subgraph = SBM_graph.SBMGraph(self._n_soldiers, self._n_commanders, inblock_prob=self._inblock_soldier_p, crossblock_prob=self._crossblock_soldier_p)
		commander_subgraph = SBM_graph.SBMGraph(self._n_commanders, self._n_commander_comms, inblock_prob=self._inblock_commander_p, crossblock_prob=self._crossblock_commander_p)
		family_subgraph = SBM_graph.SBMGraph(self._n_family_members, self._n_military, inblock_prob=self._inblock_family_p, crossblock_prob=self._crossblock_family_p)

		soldier_subgraph.sample_graph()
		commander_subgraph.sample_graph()
		family_subgraph.sample_graph()
		union_graph = nx.disjoint_union(commander_subgraph._graph, soldier_subgraph._graph)
		union_graph = nx.disjoint_union(union_graph, family_subgraph._graph)
		assert(union_graph.number_of_nodes()==self._n_family_members+self._n_military)
		for soldier_idx in range(self._n_soldiers):
			commander_idx  = soldier_subgraph._node_community[soldier_idx]
			if np.random.uniform() <= self._commander_to_soldier_p:
				union_graph.add_edge(commander_idx, self._n_commanders+soldier_idx)
				union_graph.add_edge(self._n_commanders+soldier_idx, commander_idx)
		for family_idx in range(self._n_family_members):
			military_idx  = family_subgraph._node_community[family_idx]
			if np.random.uniform() <= self._to_family_p:
				union_graph.add_edge(military_idx, self._n_military+family_idx)
				union_graph.add_edge(self._n_military+family_idx, military_idx)
		self._graph = union_graph
		self._n_nodes = union_graph.number_of_nodes()

		
		
if __name__ == '__main__':
	militaryGraph = StaticMilitaryGraph(1000, 100, 10, 1, [0.1]*3, [0.01]*3, 0.1, 0.1)
	militaryGraph.sample_graph()
	print(militaryGraph._graph.number_of_edges())
	pdb.set_trace()
