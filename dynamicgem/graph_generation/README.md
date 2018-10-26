# Graph Generation

## Static SBM graph
Each generated synthetic SBM graph is saved as two files:

1. [prefix]\_[#nodes]\_[#communities]_graph.gpickle: the graph saved in networkx format.
2. [prefix]\_[#nodes]\_[#communities]_node.pkl: the communities id each node has as a python list of length #nodes.

## Dynamic SBM graph
### Dynamic graph types
Currently we produce two different series of dynamic SBM graphs:

1. Random node perturbation: for each time step, we randomly resample the community for a few nodes and afterwards resample all the edges associated with the nodes who change their community.
2. Diminishing community: we choose one community and gradually change its nodes into other communities. The edges of the changed nodes are resampled according to the their new community.

### Storage
Each series of dynamic SBM graphs are saved as several files:

1. [prefix]\_[t]\_graph.gpickle: the graph at time step t.
2. [prefix]\_[t]\_node.pkl: a pickled python dictionary node_infos where `node_infos['community']` contains a python list of node community and `node_infos['perturbation']` contains the nodes changed before time step t.
