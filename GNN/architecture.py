from dgl.nn import GraphConv
import os
os.environ['DGLBACKEND'] = 'pytorch'
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.data
import dgl.function as fn

class GCNLAYER(nn.Module):
    def __init__(self, node_dims, msg_dim, output_dims):
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
        self.dropout = nn.Dropout(0.2)

        # initialize weight matrices
        nn.init.xavier_normal_(self.W_apply_2.weight)
        nn.init.xavier_normal_(self.W_apply.weight)
        nn.init.xavier_normal_(self.W_msg.weight)
        nn.init.xavier_normal_(self.W_msg_2.weight)
        nn.init.xavier_normal_(self.W_msg_3.weight)
        nn.init.xavier_normal_(self.W_att_1.weight)
        nn.init.xavier_normal_(self.W_att_2.weight)
        nn.init.xavier_normal_(self.W_att_3.weight)

    # define message generation function
    def message_func(self, edges):
        return {'m': edges.data['avg_a'] * F.relu(self.W_msg_3(F.relu(self.W_msg_2(F.relu(self.W_msg_1(torch.cat(edges.src['h'])))))))}
    
    #define attention calculation functions
    def att_func_1(self, edges):
        return {'a1': torch.exp(F.relu(self.W_att_1(torch.cat([edges.src['h'], edges.dst['h']], 1))))}
    def att_func_2(self, edges):
        return {'a2': torch.exp(F.relu(self.W_att_2(torch.cat([edges.src['h'], edges.dst['h']], 1))))}
    def att_func_3(self, edges):
        return {'a3': torch.exp(F.relu(self.W_att_3(torch.cat([edges.src['h'], edges.dst['h']], 1))))}
    
    # Calculate the average of all three attentions
    def avg_att(self, edges):
        return {'avg_a': (edges.data['a1'] / edges.dst['att1'] + edges.data['a2'] / edges.dst['att2'] + edges.data['a3'] / edges.dst['att3']) / 3}

    # define forward pass
    def forward(self, g, node_features):
        # so that changes don't remain in graph object
        with g.local_scope():
            
            # store features (after dropout in graph)
            g.ndata['h'] = self.dropout(node_features)

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
        pooled_vals = list()
        for batch in node_batches:
            pooled_vals.append(torch.mean(feats[batch], dim=0))
        return torch.tensor(pooled_vals)

# define model
class GCN(nn.Module):
    def __init__(self, in_feats, msg_dims, h_feats, num_classes=2):
        super(GCN, self).__init__()
        # create graph layers
        self.conv_1 = GCNLAYER(in_feats, msg_dims, h_feats)
        self.conv_2 = GCNLAYER(h_feats, msg_dims, h_feats)

        # create avg_pool layer
        self.avg_pool = AvgPoolingLayer()

        #create dense layers
        self.dense_1 = nn.Linear(h_feats, h_feats)
        self.dense_2 = nn.Linear(h_feats, num_classes)
    

    def forward(self, g, node_feats, node_batches):
        # pass input features through graph layers
        h = self.conv_1(g, node_feats)
        h = F.relu(h)
        h = self.conv2(g, h)
        h = F.relu(h)

        # pool rows belonging to same patient to get patient embeddings
        h = self.avg_pool(h, node_batches)

        # pass patient embeddings through dense layers
        h = self.dense_1(g, h)
        h = F.relu(h)
        h = self.dense_2(g, h)
        h = F.relu(h)

        # apply activation function and return logits (predictions)
        h = F.softmax(h, dim=1)

        return h