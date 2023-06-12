import os
import dgl
from dgl import save_graphs, load_graphs
from dgl.data.utils import makedirs, save_info, load_info

graphPath = os.path.join(os.curdir, 'graphs', 'graphsV1.bin')
graphs, labelDict = load_graphs(graphPath)
# print(len(list(graphs)))
# print(labelDict)

bg = dgl.batch(graphs)
print(bg)
print(bg.batch_size)
print(bg.batch_num_nodes())
print(bg.batch_num_edges())