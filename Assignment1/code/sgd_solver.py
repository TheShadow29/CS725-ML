from __future__ import print_function
from __future__ import division
import pandas as pd
import numpy as np


class one_train_info():
    def __init__(self, param_list):
        self.id1 = param_list[0]
        self.output = param_list[-1]
        self.all_other_params = param_list[1:-1]
        return

    def show_param(self, param):
        if param == 'Id':
            return self.id1
        elif param == 'Output':
            return self.output
        else:
            return self.all_other_params[param]


class reader(object):
    def __init__(self, fname):
        data_reader = pd.read_csv(fname, sep=',')
        self.all_info_list = list()
        for i in range(data_reader.last_valid_index() + 1):
            new_train_info = one_train_info(data_reader.loc[i])
            self.all_info_list.append(new_train_info)
        return


class model(object):
    def __init__(self, lamb, p, phi):
        self.lamb = lamb
        self.p = p
        self.phi = np.append(phi, 1)
        # Reminder to not forget the bias
        self.weights = np.zeros(self.phi.shape)

    # def calculate_loss(self):


if __name__ == '__main__':
    fname = '../data/train.csv'
    all_data = reader(fname)
    # Regression with p-norm regularization
    # loss = ||y - y'||^2 + lambda * ||w||^p
    # Can try to use k-fold cross validation
