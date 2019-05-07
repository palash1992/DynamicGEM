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
        self._n_nodes = self._n_military + self._n_family_members
        self._graph = None
        self.set_mtx_B()
        self.sample_node_communities()

    def construct_mtx_B_component(self, n_communities, inblock_prob=0.1, crossblock_prob=0.01):
        B_component = np.ones((n_communities, n_communities)) * crossblock_prob
        for i in range(n_communities):
            B_component[i, i] = inblock_prob
        return B_component

    def set_mtx_B(self):
        self._B_c = self.construct_mtx_B_component(self._n_commander_comms, self._inblock_commander_p, self._crossblock_commander_p)
        self._B_s = self.construct_mtx_B_component(self._n_commanders, self._inblock_soldier_p, self._crossblock_soldier_p)
        self._B_f = self.construct_mtx_B_component(self._n_military, self._inblock_family_p, self._crossblock_family_p)
        self._B = np.ones((self._n_nodes, self._n_nodes))*self._commander_to_soldier_p
        n1 = self._B_c.shape[0]
        n2 = self._B_s.shape[0]
        n3 = self._B_f.shape[0]
        n = n1 + n2 + n3
        self._B[:n1, :n1] = self._B_c
        self._B[n1:n1+n2, n1:n1+n2] = self._B_s
        self._B[n1+n2:n, n1+n2:n] = self._B_f
        for i in range(n1+n2, n):
            for j in range(n1+n2):
                self._B[i, j] = self._to_family_p
                self._B[j, i] = self._to_family_p
        return self._B

    def sample_node_communities(self):
        comm_size_c = np.random.multinomial(self._n_commanders, [1.0 / self._n_commander_comms] * self._n_commander_comms)
        comm_size_s = np.random.multinomial(self._n_soldiers, [1.0 / self._n_commanders] * self._n_commanders)
        comm_size_f = np.random.multinomial(self._n_family_members, [1.0 / self._n_military] * self._n_military)
        self._node_community = []
        for i, size in enumerate(comm_size_c):
            self._node_community += [i] * size
        for i, size in enumerate(comm_size_s):
            self._node_community += [i+self._n_commander_comms] * size
        for i, size in enumerate(comm_size_f):
            self._node_community += [i+self._n_commander_comms+self._n_commanders] * size

    def sample_graph(self):
        ''' Sample a static military graph

        '''
        self._graph = nx.DiGraph()
        # add nodes
        self._graph.add_nodes_from(range(self._n_nodes))
        # add edges
        for i in range(self._n_nodes):
            for j in range(i):
                prob = self._B[self._node_community[i], self._node_community[j]]
                if np.random.uniform() <= prob:
                    self._graph.add_edge(i, j)
                    self._graph.add_edge(j, i)
        return self._graph

        
        
if __name__ == '__main__':
    militaryGraph = StaticMilitaryGraph(1000, 100, 10, 1, [0.1]*3, [0.01]*3, 0.1, 0.01)
    militaryGraph.sample_graph()
    print(militaryGraph._graph.number_of_edges())
    pdb.set_trace()
