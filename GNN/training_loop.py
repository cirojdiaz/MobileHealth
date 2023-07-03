from dgl.nn import GraphConv
import os
os.environ['DGLBACKEND'] = 'pytorch'
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_optimizer as optim
import dgl
import dgl.data
from sklearn import metrics
from sklearn.utils import class_weight
import numpy as np
import dgl.function as fn
from torch_optimizer import Lookahead
from torch.nn import KLDivLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier



# Look into optimal ESP 
def train(g, node_batches, model, labels, train_mask, val_mask=[], test_mask=[], validate=True, test=False, init_lr=0.01, stoch=True, num_batches=20, early_stopping=True, early_stopping_patience=10, early_stopping_warmup=0, max_epochs=4950, lamb_beta1=0.9, lamb_beta2=0.999, lamb_eps=1e-6, lamb_wd=0.01, lookahead_k=5, lookahead_alpha=0.5, l1_lambda=0.01):
    # ensure that validation and/or test masks contain at least 1 node if validation and/or testing is enabled
    if test and len(test_mask) == 0:
        print("test_mask must contain at least 1 node if testing is enabled")
        raise
    if validate and len(val_mask) == 0:
        print("val_mask must contain at least 1 node if validation is enabled")
        raise
    
    # set model params to be doubles for compatibility
    # model.double()

    # convert train and val masks to numpy arrays for consistency
    train_mask = np.array(train_mask)
    val_mask = np.array(val_mask)
    test_mask = np.array(test_mask)
    
    # create optimizer and scheduler
    #optimizer = torch.optim.Adam(model.parameters(), lr=init_lr,weight_decay=0.0001)
    optimizer = optim.Lamb(
        model.parameters(),
        lr=init_lr,
        betas=(lamb_beta1, lamb_beta2),
        eps=lamb_eps,
        weight_decay=lamb_wd
    )

    # Wrapper optimizer mantaining fast weights and slow weights updated every k steps
    optimizer = Lookahead(optimizer, k=lookahead_k, alpha=lookahead_alpha)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, verbose=True)

    # set best scores
    best_val_acc = 0
    best_test_acc = 0
    best_val_bal_acc = 0
    best_scores = {}

    # extract node and edge features from graph object
    features = g.ndata["feat"]
    # compute class weights
    #class_weights_weighted = torch.tensor([1, 1])
    # kept out for now because compute_class_weights seems to be behaving funny
    # Computing weights for balancing dataset
    class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels[train_mask].squeeze().numpy())
    # Index the weight of the positive class and convert to tensor
    class_weights = torch.tensor(class_weights[1], dtype=torch.float)  # class_weights=torch.tensor(class_weights, dtype=torch.double)
    #class_weights_multiplier = torch.tensor([1, 1])
    #class_weights_weighted = class_weights * class_weights_multiplier
    
    prev_val_loss = 0
    prev_bal_val_acc = 0
    bad_epoch_cnt = 0

    #logit_multiplier = torch.tensor([1, 1])

    for e in range(max_epochs):
        # iterate through batches
        logits = model(g, features, node_batches)

        #new_logits = logits * logit_multiplier
    
        #pred = new_logits.argmax(1)
        pred = (torch.sigmoid(logits) > 0.5).float()
        # Pass class weights through the pos_weight argument to handle class imbalances
        loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)

        # Inputs: logits, target:labels
        loss = loss_function(logits[train_mask].squeeze(), labels[train_mask].squeeze().float())

        # L1 regularization
        l1_lambda = 0.01
        for p in model.parameters():
            if p.dim() > 1:
                loss += l1_lambda * p.abs().sum()

        # Compute loss
        # Note that you should only compute the losses of the nodes in the current training set batch. <- don't worry about this note
        #print(class_weights_weighted.float())
        #print(labels[train_mask].reshape(-1).long().shape)
        #loss = F.cross_entropy(input=logits[train_mask], target=labels[train_mask].reshape(-1).long(), weight=class_weights_weighted.float())
        # Look into this, check alternatives 
        # Clear gradients of optimized variables before backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute accuracy on entire training/validation/test
        train_loss = loss_function(logits[train_mask].squeeze().float(), labels[train_mask].squeeze().float())
        #train_loss = F.cross_entropy(logits[train_mask], labels[train_mask].reshape(-1).long(), weight=class_weights_weighted.float())
        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        train_bal_acc = metrics.balanced_accuracy_score(labels[train_mask], pred[train_mask])
        train_recall = metrics.recall_score(labels[train_mask], pred[train_mask])
        train_precision = metrics.precision_score(labels[train_mask], pred[train_mask], zero_division=0)
        train_f1 = metrics.f1_score(labels[train_mask], pred[train_mask])
        train_auc = metrics.roc_auc_score(labels[train_mask], pred[train_mask])

        if validate:
            val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
            fpr, tpr, thresholds = metrics.roc_curve(labels[val_mask], pred[val_mask], pos_label=1)
            val_auc = metrics.auc(fpr, tpr)
            bal_acc = metrics.balanced_accuracy_score(labels[val_mask], pred[val_mask])
            val_recall = metrics.recall_score(labels[val_mask], pred[val_mask])
            val_precision = metrics.precision_score(labels[val_mask], pred[val_mask], zero_division=0)
            #val_loss = F.cross_entropy(logits[val_mask], labels[val_mask].reshape(-1).long(), weight=class_weights_weighted.float())
            val_loss = loss_function(logits[val_mask].squeeze().float(), labels[val_mask].squeeze().float())
            val_f1 = metrics.f1_score(labels[val_mask], pred[val_mask])
            val_auc = metrics.roc_auc_score(labels[val_mask], pred[val_mask])

            # Learning rate scheduling
            scheduler.step(val_loss.item())
            # early stopping, we will not worry about this for now
            if e == 0:
                prev_bal_val_acc = bal_acc
            if val_acc > prev_bal_val_acc and e >= early_stopping_warmup:
                bad_epoch_cnt += 1
            else:
                bad_epoch_cnt = 0
            prev_bal_val_acc = bal_acc
            if bad_epoch_cnt > early_stopping_patience:
                break
            
        # Save the best validation accuracy and the corresponding test accuracy.
            if best_val_acc < bal_acc and e > 0:
                best_val_acc = val_acc
                best_val_auc = val_auc
                best_val_bal_acc = bal_acc
                if test:
                    best_test_acc = test_acc
                best_scores = {'epoch': e,
                                'train_loss': train_loss.item(),
                                'train_acc': train_acc.item(),  
                                'train_bal_acc': train_bal_acc,
                                'train_recall': train_recall,
                                'train_precision': train_precision,
                                'train_f1': train_f1,
                                'train_auc': train_auc,
                                'val_loss': val_loss.item(),
                                'val_acc': val_acc.item(), 
                                'val_bal_acc': bal_acc,
                                'val_recall': val_recall,
                                'val_precision': val_precision,
                                'val_f1': val_f1,
                                'val_auc': val_auc}

        if test:
            test_acc = (pred[test_mask] == labels[test_mask]).float().mean()
            fpr, tpr, thresholds = metrics.roc_curve(labels[test_mask], pred[test_mask], pos_label=1)
            test_auc = metrics.auc(fpr, tpr)
            test_bal_acc = metrics.balanced_accuracy_score(labels[test_mask], pred[test_mask])
            test_recall = metrics.recall_score(labels[test_mask], pred[test_mask])
            test_precision = metrics.precision_score(labels[test_mask], pred[test_mask], zero_division=0)
            #test_loss = F.cross_entropy(logits[test_mask], labels[test_mask].reshape(-1).long(), weight=class_weights_weighted.reshape(-1).float())
            test_loss = loss_function(logits[test_mask].squeeze().float(), labels[test_mask].squeeze().float())
            test_f1 = metrics.f1_score(labels[test_mask], pred[test_mask])
            test_auc = metrics.roc_auc_score(labels[test_mask], pred[test_mask])


        # print out results every 5 epochs
        '''
        if e % 5 == 0:
            print(
                "In epoch {}, loss: {:.3f}, train_acc: {:.3f}, train_bal_acc: {:.3f}, train_recall: {:.3f}, train_precision: {:.3f}".format(
                    e, train_loss, train_acc, train_bal_acc, train_recall, train_precision
                ), end=""
            )
            if validate:
                print(
                    " val loss: {:.3f}, val acc: {:.3f}, bal acc: {:.3f}, val recall: {:.3f}, val precision: {:.3f}".format(
                        val_loss, val_acc, bal_acc, val_recall, val_precision
                    ), end=""
                )
            if test:
                print("")
                print(
                    " test loss: {:.3f}, test acc: {:.3f}, test bal acc: {:.3f}, test recall: {:.3f}, test precision: {:.3f}".format(
                        test_loss, test_acc, test_bal_acc, test_recall, test_precision
                    ), end=""
                )
            print("")
        '''
    
    print("Best scores in iteration: ", best_scores)
    return best_val_bal_acc