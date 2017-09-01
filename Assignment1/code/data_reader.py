from __future__ import print_function
from __future__ import division
import pandas as pd


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


if __name__ == '__main__':
    fname = '../data/train.csv'
    all_data = reader(fname)
    # Regression options:
    # -- SVR
    # -- Ridge loss
    # -- Hinge loss [Lasso]
    # -- zero norm loss
