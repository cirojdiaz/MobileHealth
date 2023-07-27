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

# define model
class RNN(nn.Module):
    def __init__(self, in_feats, hidden_state_size, out_feats):
        super(RNN, self).__init__()
        # hidden state
        self.hidden_state = torch.rand(hidden_state_size).float()

        # next state function
        self.nsf_dense1 = nn.Linear(hidden_state_size + in_feats, hidden_state_size)
        self.nsf_dense2 = nn.Linear(hidden_state_size, hidden_state_size)

        # decoding layers
        self.decode1 = nn.Linear(hidden_state_size, hidden_state_size)
        self.decode2 = nn.Linear(hidden_state_size, out_feats)

        # nn.init.kaiming_normal_(self.nsf_dense1.weight)
        # nn.init.kaiming_normal_(self.nsf_dense2.weight)
        # nn.init.kaiming_normal_(self.decode1.weight)
        # nn.init.kaiming_normal_(self.decode2.weight) 


    def forward(self, features):

        # initalize matrix of initial hidden states
        self.hs_matrix = self.hidden_state.repeat(len(features), 1)

        # create dictionary such that x'th key stores list of patients that have exaclty x of hopsital visits
        subject_lens = dict()
        for i in range(len(features)):
            if len(features[i]) not in subject_lens.keys():
                subject_lens[len(features[i])] = list()
            subject_lens[len(features[i])].append(i)
        
        # modify dictionary so x'th key stores list of patients with at least x hospital visits
        max_len = max(subject_lens.keys())
        subjects_so_far = list()
        for i in range(max_len, 0, -1):
            if i in subject_lens.keys():
                subjects_so_far += subject_lens[i]
            subject_lens[i] = subjects_so_far.copy()

        # feed forward - sequentially update the hidden state using the next hospital visit
        for i in range(1, max_len + 1):

            features_tensor = torch.stack([features[index][i - 1] for index in subject_lens[i]])

            # input to next state function
            hs_w_entries = torch.cat([features_tensor, self.hs_matrix[subject_lens[i]]], axis=1)

            # to avoid inplace operations - pytorch doesn't allow these
            hs_matrix_copy = self.hs_matrix[subject_lens[i]].clone()
            
            # Pass through nsf (next state function) dense1
            hs_matrix_copy = self.nsf_dense1(hs_w_entries)
            hs_matrix_copy = F.relu(hs_matrix_copy)

            # Pass through nsf (next state function) dense2
            hs_matrix_copy = self.nsf_dense2(hs_matrix_copy)
            hs_matrix_copy = F.relu(hs_matrix_copy)
            
            # copy updated states into matrix of hidden states
            self.hs_matrix.index_copy(0, torch.tensor(subject_lens[i]), hs_matrix_copy)

        # translate final hidden states into predictions
        logits = self.decode1(self.hs_matrix)
        logits = F.relu(logits)
        logits = self.decode2(logits)
        logits = torch.sigmoid(logits)

        return logits