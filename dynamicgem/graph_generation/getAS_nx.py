import networkx as nx
import numpy as np
import os
DATA_DIR = 'as-733'
fnames = sorted(os.listdir(DATA_DIR))

routersD = {}
routerId = 0
file_sno = 1
for curr_file in fnames:
	with open(DATA_DIR+ '/' + curr_file) as f:
		G = nx.DiGraph()
		G.add_nodes_from(range(7716))
		for line in f:
			line = line.strip()
			if line.startswith('#'):
				continue
			v_i, v_j = line.split('\t')
			if v_i not in routersD:
				routersD[v_i] =  routerId
				routerId += 1
			if v_j not in routersD:
				routersD[v_j] =  routerId
				routerId += 1
			# G.add_node(routersD[v_i])
			# G.add_node(routersD[v_j])
			G.add_edge(routersD[v_i], routersD[v_j], weight=1)
		fname_s = 'as-graphs/month_' + str(file_sno) + '_graph.gpickle'
		print('File %d/%d' % (file_sno, len(fnames)))
		file_sno += 1
		nx.write_gpickle(G, fname_s)


