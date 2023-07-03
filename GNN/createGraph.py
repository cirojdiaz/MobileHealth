import numpy as np
import math
import torch
import dgl
from dgl.data.utils import save_graphs
import networkx as nx
import matplotlib.pyplot as plt
import os
import pandas as pd

def throughData(csv):
    """
    Creates a batch of bidirectional graphs from a csv file of patient data
   
    param: csv file of patient data
    return: batch of graphs, list of nodes, list of labels
    output: saves batch of graphs to .bin file
    """
    subjDict = create_subj_dict(csv) 
    # subjDict is a dictionary
    # subjDict[patient] is a list 
    # subjDict[patient][visit] is a pandas series
    
    keysToExtract = ['DIASBP_R', 'PULSE_R', 'SYSBP_R', 'bmicalculated', 'CREATININE_R', 'HDL_R', 'LDL_R', 'TOTCHOL_R', 'RDAYSFROMINDEX', 'PostCond']

    flag = False # flag to check if there is a nan value
    
    allGraphs = [] # list of all graphs
    postCondList = [] # graph label 

    running_node_sum = 0 # <----- keep track of node ids in batched graph
    
    node_batches = [] # <----- list of post_batched node indices belonging to each graph
    
    for patient in subjDict:
        numNodes = 0  # number of nodes
        
        nodeFeatureList = [] # node feature list, list of list 
        # edgeFeatureList = [] # edge feature list, list of list

        print(f'Patient {patient}')
        if(len(subjDict[patient]) > 1):
            for visit in range(len(subjDict[patient])): # each visit is a dictionary
                extractValues = [] # list of vital sign values to extract 
                # edge = [] # list of edge values
                post = [] # list of post cond values for a visit
                for key in keysToExtract:
                    value = subjDict[patient][visit][key]
                    
                    if(key == 'PostCond'): # if the key is PostCond, append the value to the postCondList
                        post.append(int(value))
                        continue
                    if math.isnan(value): # if the value is nan, skip this visit
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
            # create graph and append to allGraphs
            if(numNodes > 1): # another check to make sure there is more than one node
                if(post.count(1) > post.count(0)):
                    # if there are more 1s than 0s, append 1 to the postCondList
                    # postCondList.append(torch.tensor([1], dtype=torch.float32))
                    postCondList.append([1])
                    # postCondList.append(1)
                else:
                    # postCondList.append(torch.tensor([0], dtype=torch.float32))
                    postCondList.append([0])
                    # postCondList.append(0)
                
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
    
    filePath =  os.path.join(os.getcwd(), 'graphs', 'graphsTest.bin') 
    graphLabels = {"gLabel": labelTensor}
    
    saveGraph(filePath, allGraphs, graphLabels)
    return dgl.batch(allGraphs), node_batches, labelTensor # <-----

# graph function
# def createGraph(edgeFeature, nodeFeature, numNodes):
def createGraph(nodeFeature, numNodes):
    """
    Creates a bidirectional graph from a list of node features
    
    param: list of node features, number of nodes (int)
    output: bidirectional graph
    """
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
    """
    Plots a graph
    
    param: graph
    output: plot of graph
    """
    G = dgl.to_networkx(graph) # convert to networkx graph
    
    # graphing options
    options = {
    'node_color': 'red',
    'node_size': 20,
    'width': 1,
    }

    # plot graph
    nx.draw(G, **options)
    plt.show()

# save graph
def saveGraph(filePath, graphs, labels):
    """
    Saves graph(s) and associated labels to a file
    
    param: file path (str), list of graphs, list of labels 
    output: binary file containing graphs and labels
    """
    
    print("Saving graph...")
    save_graphs(filePath, graphs, labels)
                
def create_subj_dict(df):
    subjects = dict()
    for index, row in df.iterrows():
        if row["RSUBJID"] not in subjects.keys():
            subjects[row["RSUBJID"]] = list()

        subjects[row["RSUBJID"]].append(row.drop(columns="RSUBJID"))
    
    return subjects
'''
def main():
    df = pd.read_csv('duke_vital_model_imputed.csv')
    throughData(df)

main()
'''