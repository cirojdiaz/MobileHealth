import numpy as np
import math
import torch
import dgl
from dgl.data.utils import save_graphs
import networkx as nx
import matplotlib.pyplot as plt
import os

# X - 2D numpy array, training data matrix
# y - 1D numpy array, labels
def throughData(X, y):
    # load data
    data = np.load(csv, allow_pickle=True).item()

    x = list(data.items()) #x is a list of tuples
    keysToExtract = ['DIASBP_R', 'PULSE_R', 'SYSBP_R', 'HEIGHT_R', 'WEIGHT_R', 'CREATININE_R', 'HDL_R', 'LDL_R', 'TOTCHOL_R', 'RDAYSFROMINDEX', 'PostCond']

    flag = False # flag to check if there is a nan value
    
    allGraphs = [] # list of all graphs
    node_batches = [] # <----- list of post_batched node indices belonging to each graph
    postCondList = [] # graph label 

    running_node_sum = 0

    print(x)
    # loop through the list of patients (list of tuples)
    for i, e in enumerate(x): 
        numNodes = 0  # number of nodes

        nodeFeatureList = [] # node feature list, list of list 
        # edgeFeatureList = [] # edge feature list, list of list
        
        print(f'Patient {i}')
        if(len(e[1]) > 1): # ignore patients with only one visit
            for j, k in enumerate(e[1]): # loop through all the visits for one patient
                extractValues = [] # list of values to extract from the dictionary
                edge = [] # list of edge values
                post = [] # list of post cond values for a visit
                for key in keysToExtract: # loop through the keys to extract
                    value = k[key] # extract the value
                    # if(key == 'RDAYSFROMINDEX'): # if the key is RDAYSFROMINDEX, convert the value to a float
                    #     edge.append(float(value))
                    #     continue
                    if(key == 'PostCond'): # if the key is PostCond, append the value to the postCondList
                        post.append(int(value))
                        continue
                    elif math.isnan(value): # if the value is nan, skip this visit
                        flag = True
                        break
                    else: 
                        extractValues.append(float(value)) # append values
                if flag == True: # if there is a nan value, skip this visit
                    flag = False
                    continue
                else: # append list of values to the node feature list
                    # edgeFeatureList.append(edge)
                    nodeFeatureList.append(extractValues)
                    numNodes += 1
                    
            # compare post cond 
            if(post.count(1) > post.count(0)):
                # if there are more 1s than 0s, append 1 to the postCondList
                # postCondList.append(torch.tensor([1], dtype=torch.float32))
                postCondList.append([1])
                # postCondList.append(1)
            else:
                # postCondList.append(torch.tensor([0], dtype=torch.float32))
                postCondList.append([0])
                # postCondList.append(0)
            
            # create graph and append to allGraphs
            if(numNodes > 1): # another check to make sure there is more than one node
                # edgeFeatureList.pop(0) # remove the first edge feature since there are less edges than nodes, the first visit has an edge feature which is not needed 
                # g = createGraph(edgeFeatureList, nodeFeatureList, numNodes)
                g = createGraph(nodeFeatureList, numNodes)
                allGraphs.append(g)
                node_batches.append(list(range(running_node_sum, running_node_sum + numNodes))) # <----- keep track of what node ids nodes belonging to this graph will have in batched graph
                running_node_sum += numNodes # <-----
                # plotGraph(g) # plot graph for visualization purposes
        else:
            continue

    labelTensor = torch.tensor(postCondList, dtype = torch.int32) # convert postCondList to tensor
    
    filePath =  os.path.join(os.getcwd(), 'graphs', 'graphsV2.bin') 
    graphLabels = {"gLabel": labelTensor}
    
    saveGraph(filePath, allGraphs, graphLabels)

    return dgl.batch(allGraphs), node_batches, labelTensor # <-----


# graph function
# def createGraph(edgeFeature, nodeFeature, numNodes):
def createGraph(nodeFeature, numNodes):
    source = list(range(numNodes - 1))  # source nodes
    destination = list(range(1, numNodes))  # destination nodes
    
    
    nodeFeatureTensor = torch.tensor(nodeFeature)
    # edgeFeatureTensor = torch.tensor(edgeFeature)

    # print("Number of edge features:", edgeFeatureTensor.shape)
    print("Number of edges:", numNodes - 1)
    
    # edgeFeatureTensor = torch.cat((edgeFeatureTensor, edgeFeatureTensor), 0) # duplicate edge features (for reverse edges)
    
    # print("Number of edge features after doubling:", edgeFeatureTensor.shape)
    
    g = dgl.graph((source, destination)) # create graph
    g = dgl.add_reverse_edges(g) # add reverse edges
    g.ndata['feat'] = nodeFeatureTensor # add node features
    # g.edata['feat'] = edgeFeatureTensor # add edge features
    return g

# plotting graphs for visualization purpose 
def plotGraph(graph):
    G = dgl.to_networkx(graph) # convert to networkx graph
    
    # graphing options
    options = {
    'node_color': 'red',
    'node_size': 20,
    'width': 1,
    }

    nx.draw(G, **options)
    plt.show()

# save graph
def saveGraph(filePath, graphs, labels):
    print("Saving graph...")
    save_graphs(filePath, graphs, labels)


def main():
    csv = "duke_imputed_normalized_dict.npy"
    throughData(csv)

main()
