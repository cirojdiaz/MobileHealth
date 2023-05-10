import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.data
from sklearn import metrics
from sklearn.utils import class_weight
import numpy as np
import dgl.function as fn
from torch.nn import KLDivLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau

class GCNELAYER(nn.Module):
    def __init__(self, node_dims, edge_dims, output_dims):
        super(GCNELAYER, self).__init__()
        self.W_msg = nn.Linear(node_dims + edge_dims, output_dims)
        self.W_msg_2 = nn.Linear(output_dims, output_dims)
        self.W_apply = nn.Linear(output_dims + node_dims, output_dims)
    def message_func(self, edges):
        return {'m': F.relu(self.W_msg_2(F.relu(self.W_msg(torch.cat([edges.src['h'], edges.data['h']], 1)))))}
    def forward(self, g, node_features, edge_features):
        with g.local_scope():
            g.ndata['h'] = node_features
            g.edata['h'] = edge_features
            g.update_all(self.message_func, fn.mean('m', 'h_neigh'))
            g.ndata['h'] = self.W_apply(torch.cat([g.ndata['h'], g.ndata['h_neigh']], 1))
            return g.ndata['h']

class MyANN(nn.Module):
    def __init__(self, input_dim, h_feats):
        super(MyANN, self).__init__()
        self.fc1 = nn.Linear(input_dim, h_feats)
        self.fc2 = nn.Linear(h_feats, h_feats)
        self.fc3 = nn.Linear(h_feats, 1)
        # self.fc1 = nn.linear(h_feats, output_dim)

    def forward(self, X):
        h = self.fc1(X)
        h = F.relu(h)
        # print(h.shape)
        h = self.fc2(h)
        h = F.relu(h)
        # print(h.shape)
        h = self.fc3(h)
        # print(h.shape)
        h = torch.sigmoid(h)
        return h

def train(X_train, y_train, model, X_test=None, y_test=None, validate=True, init_lr=0.01):
    model.double()
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)

    best_val_acc = 0
    best_test_acc = 0

    for e in range(4500):
        # Forward
        logits_train = model(X_train)
        logits_test = model(X_test)

        # Compute prediction
        threshold = 0.5

        pred_train = torch.zeros_like(logits_train)
        pred_train[logits_train > threshold] = 1
        pred_train = pred_train.view(-1)

        pred_test = torch.zeros_like(logits_test)
        pred_test[logits_test > threshold] = 1
        pred_test = pred_test.view(-1)

        # Compute loss
        loss = F.binary_cross_entropy(logits_train.view(-1), y_train)

        # Compute accuracy on training/validation/test
        train_acc = (pred_train == y_train).float().mean()
        train_bal_acc = metrics.balanced_accuracy_score(y_train, pred_train)
        train_recall = metrics.recall_score(y_train, pred_train)
        train_precision = metrics.precision_score(y_train, pred_train, zero_division=0)



        if validate:
            val_acc = (pred_test == y_test).float().mean()
            fpr, tpr, thresholds = metrics.roc_curve(y_test, pred_test, pos_label=1)
            val_auc = metrics.auc(fpr, tpr)
            bal_acc = metrics.balanced_accuracy_score(y_test, pred_test)
            val_recall = metrics.recall_score(y_test, pred_test)
            val_precision = metrics.precision_score(y_test, pred_test, zero_division=0)

        # test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        # Save the best validation accuracy and the corresponding test accuracy.
        if validate and best_val_acc < val_acc:
            best_val_acc = val_acc
            best_val_auc = val_auc
            best_bal_acc = bal_acc
            # best_test_acc = test_acc

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        # if e % 5 == 0:
        #     print(
        #         "In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f})".format(
        #             e, loss, val_acc, best_val_acc
        #         )
        #     )
        if validate and e % 5 == 0:
            print(
                "In epoch {}, loss: {:.3f}, train_acc: {:.3f}, train_bal_acc: {:.3f}, train_recall: {:.3f}, train_precision: {:.3f}, val acc: {:.3f}, bal acc: {:.3f}, val recall: {:.3f}, val precision: {:.3f}".format(
                    e, loss, train_acc, train_bal_acc, train_recall, train_precision, val_acc, bal_acc, val_recall, val_precision
                )
            )

        elif e % 5 == 0:
            print(
                "In epoch {}, loss: {:.3f}, train_acc: {:.3f}, train_bal_acc: {:.3f}".format(
                    e, loss, train_acc, train_bal_acc
                )
            )