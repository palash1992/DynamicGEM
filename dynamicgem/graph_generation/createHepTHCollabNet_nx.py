import networkx as nx
import os
import re
import pdb
import pickle

import sys
sys.path.append('./')

DATA_DIR = 'data/real/hep-th/abs'
folder_names = os.listdir(DATA_DIR)

author_namesD = {}
author_id = 0

G = nx.DiGraph()
prevMonth = '01'
prevFolder = '1992'
file_sno = 1
for curr_folder in sorted(folder_names):	# Go through all the years
	file_names = sorted(os.listdir(DATA_DIR + '/' + curr_folder))
	for curr_file in file_names:	# Go through all the papers in one year
		month = curr_file[2:4]
		if month != prevMonth:
			f_name = "data/real/hep-th/graphs/month_" + str(file_sno) + "_graph.gpickle"
			nx.write_gpickle(G, f_name)
			file_sno += 1

			print(prevFolder + '_' + prevMonth + ': Number of nodes = ' + str(G.number_of_nodes()))
			print(prevFolder + '_' + prevMonth + ': Number of edges = ' + str(G.number_of_edges()))
			prevMonth = month
			prevFolder = curr_folder

		with open(DATA_DIR + '/' + curr_folder + '/' + curr_file) as f:	# Open each paper
			for line in f:	# Read lines until you get the line containing author names
				line = line.strip()
				if line.startswith('Authors'):
					author_names = line.split(':')[1]
					author_namesL = re.split('\(.*\)|\(.*$|[0-9,()]| and |Jr\.', author_names) # Get list of authors
					author_namesL = [author_namesL[i].strip() for i in range(len(author_namesL))]
					author_namesL = filter(None, author_namesL)#[a for a in author_namesL  if a]
					for author in author_namesL:	# Add the author in the dictionary and corresponding node in the graph
						if author not in author_namesD:
							author_namesD[author] = author_id
							G.add_node(author_id)
							author_id += 1
					for i in range(len(author_namesL)): # Connect all the authors in the current paper
						for j in range(i+1, len(author_namesL)):
							author_i = author_namesD[author_namesL[i]]
							author_j = author_namesD[author_namesL[j]]
							assert(author_i != author_j)
							if G.has_edge(author_i, author_j):
								weight = G.get_edge_data(author_i, author_j)['weight']
								weight += 1
								G.add_edge(author_i, author_j, weight=weight)
								G.add_edge(author_j, author_i, weight=weight)
							else:
								G.add_edge(author_i, author_j, weight=1)
								G.add_edge(author_j, author_i, weight=1)
					break

f_name = "data/real/hep-th/graphs/month_" + str(file_sno) + "_graph.gpickle"
nx.write_gpickle(G, f_name)
pickle.dump(author_namesD, open("data/real/hep-th/graphs/authorsD.pickle", "wb"))
print(curr_folder + '_' + month + ': Number of nodes = ' + str(G.number_of_nodes()))
print(curr_folder + '_' + month + ': Number of edges = ' + str(G.number_of_edges())	)



