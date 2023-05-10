import torch
import torch.utils.data
import torch.optim as optim
import sys
import math
import numpy as np
import logging
import nni
import argparse
import set_random_seed
import pandas as pd
import numpy as np
from sklearn.metrics import balanced_accuracy_score

logger = logging.getLogger('func_approx_NNI')

def map_act_func(af_name):
    if af_name == "ReLU":
        act_func = torch.nn.ReLU()
    elif af_name == "LeakyReLU":
        act_func = torch.nn.LeakyReLU()
    elif af_name == "Sigmoid":
        act_func = torch.nn.Sigmoid()
    elif af_name == "Tanh":
        act_func = torch.nn.Tanh()
    elif af_name == "Softplus":
        act_func = torch.nn.Softplus()
    else:
        sys.exit("Invalid activation function")
    return act_func

def map_optimizer(opt_name, net_params, lr):
    if opt_name == "SGD":
        opt = optim.SGD(net_params, lr=lr)
    elif opt_name == "Adam":
        opt = optim.Adam(net_params, lr=lr)
    else:
        sys.exit("Invalid optimizer")
    return opt

def map_loss_func(loss_name):
    if loss_name == "binary_cross":
        loss_func = torch.nn.BCELoss()
    return loss_func

class FuncApproxNet (torch.nn.Module):
    def __init__(self, params):
        super(FuncApproxNet, self).__init__()
        self.hidden_size_1 = params['hidden_size_1']
        self.hidden_size_2 = params['hidden_size_2']
        self.hidden_size_3 = params['hidden_size_3']
        act_func = map_act_func(params['act_func'])
        self.fc1 = self._fc_block(1, self.hidden_size_1, act_func)
        last_layer_size = self.hidden_size_1
        if self.hidden_size_2 > 0:
            self.fc2 = self._fc_block(self.hidden_size_1, self.hidden_size_2, act_func)
            last_layer_size = self.hidden_size_2
        if self.hidden_size_3 > 0:
            self.fc3 = self._fc_block(self.hidden_size_2, self.hidden_size_3, act_func)
            last_layer_size = self.hidden_size_3
        self.out = self._fc_block(last_layer_size, 1, torch.nn.Sigmoid)

    def forward(self, x):
        x = self.fc1(x)
        if self.hidden_size_2:
            x = self.fc2(x)
        if self.hidden_size_3:
            x = self.fc3(x)
        x = self.out(x)
        return x

    def _fc_block(self, in_c, out_c, act_func):
        block = torch.nn.Sequential(
            torch.nn.Linear(in_c, out_c),
            act_func
        )
        return block

# class FuncApproxDataset(torch.utils.data.Dataset):
#     def __init__(self, func_name, is_train):
#         self.size = 1000 if is_train else 100
#         self.is_train = is_train
#         if func_name == 'quadratic':
#             self.func = self._quadratic
#         elif func_name == 'oscillator':
#             self.func = self._oscillator
#         self._prep_oscillator()

#     def __getitem__(self, index):
#         if self.is_train:
#             data = torch.rand(1, 1)
#         else:
#             data = torch.empty(1, 1, dtype=torch.float)
#             val = (index % self.size)/float(self.size)
#             data.fill_(val)  # make uniform spacing for consistent evaluation
#         label = self.func(data)
#         return {'x': data, 'y_exp': label}

#     def __len__(self):
#         return self.size

#     def _prep_oscillator(self):
#         self.osc_x = np.arange(11) * 0.1
#         self.osc_y = np.zeros_like(self.osc_x)
#         y = self.osc_x[0]
#         for i in range(len(self.osc_x)):
#             y = -(y * y - 0.5)
#             self.osc_y[i] = y

#     def _quadratic(self, x):
#         return x*x

#     def _oscillator(self, x):
#         y = torch.from_numpy(np.interp(x, self.osc_x, self.osc_y))
#         y = y.float()
#         return y

def main(params):
    if not validate_params(params): # for invalid param combinations, report the worst possible result
        nni.report_final_result(0.0)
        return

    train_df = pd.read_csv("train_ds_resampled.csv")
    test_df = pd.read_csv("test_ds_resampled.csv")

    y_train = train_df["PostCond"]
    y_test = test_df["PostCond"]

    train_ds = train_df.drop(labels=["PostCond"], axis=1).to_numpy()
    train_ds = np.concatenate([train_ds, y_train], axis=1)
    eval_ds = test_df.drop(labels=["PostCond"], axis=1).to_numpy()
    eval_ds = np.concatenate([eval_ds, y_test], axis=1)

    net = FuncApproxNet(params)
    optimizer = map_optimizer(params['optimizer'], net.parameters(), params['learning_rate'])
    loss_func = map_loss_func(params['loss'])

    epochs = 1500
    last_results = []
    for epoch in range(epochs):
        # log evaluation results every 5 epochs
        if epoch % 5 == 4:
            rms_error = 0
            max_error = 0
            net.eval()
            with torch.no_grad():
                eval_x = eval_ds[:, :-1]
                eval_y = eval_ds[:, -1].flatten()
                logits = net(eval_x)
                pred = torch.zeros_like(logits)
                pred[logits > 0.5] = 1
                pred = pred.flatten()
                eval_auc = balanced_accuracy_score(pred, eval_y)

                train_x = train_ds[:, :-1]
                train_y = train_ds[: -1].flatten()
                logits = net(train_x)
                pred = torch.zeros_like(logits)
                pred[logits > 0.5] = 1
                train_auc = balanced_accuracy_score(pred, train_y)
            eval_metric = eval_auc
            train_metric = train_auc
            nni.report_intermediate_result(eval_metric)
            print("epoch ", str(epoch), " | train metric: ", str(train_metric), " | eval metric : ", str(eval_metric))

        # do training
        net.train()
        data = train_ds[:, :-1]
        label = train_ds[:, -1]
        output = net(data)
        loss = loss_func(output, label)
        loss.backward()
        optimizer.step()

        if epoch >= epochs-25:
            last_results.append(eval_metric)
    nni.report_final_result(min(last_results)) # use min of last results since results fluctuates a lot sometimes

def generate_default_params():
    '''
    Generate default parameters for mnist network.
    '''
    params = {
        'hidden_size_1': 16,
        'hidden_size_2': 16,
        'hidden_size_3': 16,
        'act_func': 'ReLU',
        'learning_rate': 0.005,
        'optimizer': 'Adam',
        'loss': 'SmoothL1Loss'}
    return params

def validate_params(params):
    if params['hidden_size_2'] == 0 and params['hidden_size_3'] != 0:
        return False
    return True

if __name__ == '__main__':
    set_random_seed.set_random_seed()
    parser = argparse.ArgumentParser(description='Script to train a function approximator')
    # parser.add_argument('func_name', type=str, help='function name')
    # args = parser.parse_args()
    try:
        # get parameters form tuner
        updated_params = nni.get_next_parameter()
        logger.debug(updated_params)
        # run a NNI session
        params = generate_default_params()
        params.update(updated_params)
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise