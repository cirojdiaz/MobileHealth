from dgl.nn import GraphConv
import os
os.environ['DGLBACKEND'] = 'pytorch'
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.data
import dgl.function as fn
import numpy as np


class GCNLAYER(nn.Module):
    def __init__(self, node_dims, msg_dim, output_dims,dropout_rate):
        super(GCNLAYER, self).__init__()

        # create weights for message generation
        self.W_msg_1 = nn.Linear(node_dims, msg_dim)
        self.W_msg_2 = nn.Linear(msg_dim, msg_dim)
        self.W_msg_3 = nn.Linear(msg_dim, msg_dim)

        # create weights for combining neighbourhood information with current embedding
        self.W_combine_neighbours_1 = nn.Linear(msg_dim + node_dims, output_dims)
        self.W_combine_neighbours_2 = nn.Linear(output_dims, output_dims)

        # create weights for attention calculations
        self.W_att_1 = nn.Linear(2 * node_dims, 1)
        self.W_att_2 = nn.Linear(2 * node_dims, 1)
        self.W_att_3 = nn.Linear(2 * node_dims, 1)

        # create dropout layer 
        self.dropout = nn.Dropout(dropout_rate)

        # layer normalization across feature dimension
        self.layer_norm = nn.LayerNorm(node_dims)

        '''
        # initialize weight matrices
        nn.init.xavier_normal_(self.W_combine_neighbours_1.weight)
        nn.init.xavier_normal_(self.W_combine_neighbours_2.weight)
        nn.init.xavier_normal_(self.W_msg_1.weight)
        nn.init.xavier_normal_(self.W_msg_2.weight)
        nn.init.xavier_normal_(self.W_msg_3.weight)
        nn.init.xavier_normal_(self.W_att_1.weight)
        nn.init.xavier_normal_(self.W_att_2.weight)
        nn.init.xavier_normal_(self.W_att_3.weight)
        '''

        # Kaiming He weight matrices, optimal for ReLU activation functions
        nn.init.kaiming_normal_(self.W_combine_neighbours_1.weight)
        nn.init.kaiming_normal_(self.W_combine_neighbours_2.weight)
        nn.init.kaiming_normal_(self.W_msg_1.weight)
        nn.init.kaiming_normal_(self.W_msg_2.weight)
        nn.init.kaiming_normal_(self.W_msg_3.weight)
        nn.init.kaiming_normal_(self.W_att_1.weight)
        nn.init.kaiming_normal_(self.W_att_2.weight)
        nn.init.kaiming_normal_(self.W_att_3.weight)       

    '''
    def print_debug_info(self, tensor, name):
        """Utility function for printing debug info about a tensor."""
        print(f"{name}: min={tensor.min()}, max={tensor.max()}")
        if torch.isnan(tensor).any():
            print(f"NaNs found in {name}")
        if (tensor == 0).any():
            print(f"Zeros found in {name}")
    '''

    # define message generation function
    def message_func(self, edges):
        return {'m': edges.data['avg_a'] * F.relu(self.W_msg_3(F.relu(self.W_msg_2(F.relu(self.W_msg_1(edges.src['h']))))))}
    
    #define attention calculation functions
    def att_func_1(self, edges):
        return {'a1': torch.exp(F.relu(self.W_att_1(torch.cat([edges.src['h'], edges.dst['h']], 1))))}
    def att_func_2(self, edges):
        return {'a2': torch.exp(F.relu(self.W_att_2(torch.cat([edges.src['h'], edges.dst['h']], 1))))}
    def att_func_3(self, edges):
        return {'a3': torch.exp(F.relu(self.W_att_3(torch.cat([edges.src['h'], edges.dst['h']], 1))))}
    
    # Calculate the average of all three attentions
    def avg_att(self, edges):
        # Initialize avg_a to zero matrix of same dimensions as a1
        #avg_a = torch.zeros_like(edges.data['a1'])
        '''
        non_zero_mask = (edges.dst['att1'] + edges.dst['att2'] + edges.dst['att3']) != 0
        avg_a[non_zero_mask] = (edges.data['a1'][non_zero_mask] / (edges.dst['att1'][non_zero_mask]) + 
                                edges.data['a2'][non_zero_mask] / (edges.dst['att2'][non_zero_mask]) + 
                                edges.data['a3'][non_zero_mask] / (edges.dst['att3'][non_zero_mask])) / 3
        return {'avg_a': avg_a}     
        '''
        return {'avg_a': (edges.data['a1'] / edges.dst['att1'] + edges.data['a2'] / edges.dst['att2'] + edges.data['a3'] / edges.dst['att3']) / 3}

    # define forward pass
    def forward(self, g, node_features):
        # so that changes don't remain in graph object
        with g.local_scope():
            
            # store features (after dropout and layer normalization in graph)
            g.ndata['h'] = self.dropout(self.layer_norm(node_features))

            # calculate exponentiated attentions and store in edges
            g.apply_edges(self.att_func_1)
            g.apply_edges(self.att_func_2)
            g.apply_edges(self.att_func_3)

            # sum up exponentiated attentions accross all in-edges for each node
            g.update_all(lambda x: {'a1_att': x.data['a1']}, fn.sum('a1_att', 'att1'))
            g.update_all(lambda x: {'a2_att': x.data['a2']}, fn.sum('a2_att', 'att2'))
            g.update_all(lambda x: {'a3_att': x.data['a3']}, fn.sum('a3_att', 'att3'))

            # calculate final attentions
            g.apply_edges(self.avg_att)

            # Handle NaN values by replacing them with zeroes 
            #g.edata['avg_a'] = torch.where(torch.isnan(g.edata['avg_a']), torch.zeros_like(g.edata['avg_a']), g.edata['avg_a'])
            # Initialize h_neight to a zero matrix of same size of node_features
            #g.ndata['h_neigh'] = torch.zeros_like(node_features)

            # aggregate neighbourhood information
            g.update_all(self.message_func, fn.mean('m', 'h_neigh'))

            #combine neighbourhood information with current embedding
            g.ndata['h'] = self.W_combine_neighbours_2(F.relu(self.W_combine_neighbours_1(torch.cat([g.ndata['h'], g.ndata['h_neigh']], 1))))
            
            return g.ndata['h']


# custom average pooling layer to average all nodes within the same graph
class AvgPoolingLayer(nn.Module):
    def __init__(self):
        super(AvgPoolingLayer, self).__init__()

    # define pooling proccess - average all rows that belong to the same "batch" (i.e the same patient)
    def forward(self, feats, node_batches):  
        #print(feats.shape)
        pooled_vals = torch.empty(0, feats.shape[1])
        for batch in node_batches:
            pooled_vals = torch.cat([pooled_vals, torch.mean(feats[batch], dim=0).reshape(1, -1)], dim=0)
        return pooled_vals

# define model
class GCN(nn.Module):
    def __init__(self, in_feats, msg_dims, h_feats, dropout_rate, num_classes=1):
        super(GCN, self).__init__()
        # create graph layers
        self.conv_1 = GCNLAYER(in_feats, msg_dims, h_feats,dropout_rate)
        self.conv_2 = GCNLAYER(h_feats, msg_dims, h_feats,dropout_rate)

        # create avg_pool layer
        self.avg_pool = AvgPoolingLayer()

        #create dense layers
        self.dense_1 = nn.Linear(h_feats, h_feats)
        self.dense_2 = nn.Linear(h_feats, num_classes)
    

    def forward(self, g, node_feats, node_batches):
        # pass input features through graph layers
        h = self.conv_1(g, node_feats)
        h = F.relu(h)
        h = self.conv_2(g, h)
        h = F.relu(h)

        # pool rows belonging to same patient to get patient embeddings
        h = self.avg_pool(h, node_batches)

        # Upsample h and corresponding labels for it(Need to differentiate using backprop), use a smote function to create synthetic samples so that it only happens in the train set


        # pass patient embeddings through dense layers
        h = self.dense_1(h)
        h = F.relu(h)
        h = self.dense_2(h)
        #h = F.relu(h)
        
        # apply activation function and return logits (predictions)
        #h = F.softmax(h, dim=1)
        # Since BCEWithLogitsLoss combines a sigmoid layer and BCELoss, no need to add a sigmoid layer in the forward pass
        return h