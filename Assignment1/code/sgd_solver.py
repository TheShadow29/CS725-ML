from __future__ import print_function
from __future__ import division
import pandas as pd
import numpy as np


class one_train_info():
    def __init__(self, param_list):
        self.id1 = param_list[0]
        self.output = param_list[-1]
        # Note to self : not taking date and time into consideration
        self.all_other_params = param_list[2:-1]
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


class phi_holder(object):
    def __init__(self, _phi, _id_list):
        self.phi = _phi
        self.id_list = _id_list


class model(object):
    # Might also need an id_list to ind map
    def __init__(self, _lamb, _p, all_data):
        self.lamb = _lamb
        self.p = _p
        observed_y = list()
        id_list = list()
        # +1 is for the bias term
        phi_matrix = np.zeros((len(all_data.all_info_list),
                              len(all_data.all_info_list[0].all_other_params) + 1))
        for ind, d_info in enumerate(all_data.all_info_list):
            # d_info is of the type one_train_info
            id_list.append(d_info.id1)
            observed_y.append(d_info.output)
            phi_matrix[ind, :] = np.append(d_info.all_other_params, 1)

        self.id_np_list = np.array(id_list)
        self.y_obs = np.array(observed_y)
        self.Phi_h = phi_holder(phi_matrix, id_list)
        # Reminder to not forget the bias
        self.weights = np.zeros(self.Phi_h.phi.shape[1])
        self.total_ind = self.Phi_h.phi.shape[0]

        self.lr = 1

    def complete_loss(self):
        norm2_error = 0
        for i in range(self.total_ind):
            norm2_error += self.norm2_error_one_ind(i)**2

        penalty = self.lamb * np.linalg.norm(self.weights, ord=self.p)
        return norm2_error + penalty

    def norm2_error_one_ind(self, ind):
        # note that it is not norm2 squared [Need to be careful]
        # currently accepting index, may need to change if it should accept the ID
        return np.abs(np.dot(self.Phi_h.phi[ind, :], self.weights) - self.y_obs[ind])

    def one_weight_update(self, x_ind):
        new_weights = self.weights - self.lr * self.gradient_one_point(x_ind)
        return new_weights

    def gradient_one_point(self, x_ind):
        grad = np.zeros(self.weights.shape)
        for i in range(grad.shape):
            grad[i] = 2 * norm2_error_one_ind(x_ind) +

    def penalty_grad(self, w_ind):



if __name__ == '__main__':
    fname = '../data/train.csv'
    all_data = reader(fname)
    # Regression with p-norm regularization
    # loss = ||y - y'||^2 + lambda * ||w||^p
    # Can try to use k-fold cross validation
    # to check if the sgd code is indeed working
    # do a sanity check for the case p=2
    lamb = 1
    p = 2
    sgd_model = model(lamb, p, all_data)
