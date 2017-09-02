from __future__ import print_function
from __future__ import division
import pandas as pd
import numpy as np
import pdb
from sgd_template import get_feature_matrix, get_output
# class one_train_info():
#     def __init__(self, param_list):
#         self.id1 = param_list[0]
#         self.output = param_list[-1]
#         # Note to self : not taking date and time into consideration
#         self.all_other_params = param_list[2:-1]
#         return

#     def show_param(self, param):
#         if param == 'Id':
#             return self.id1
#         elif param == 'Output':
#             return self.output
#         else:
#             return self.all_other_params[param]


# class one_test_info():
#     def __init__(self, param_list):
#         self.id1 = param_list[0]
#         self.all_other_params = param_list[2:]


# class reader(object):
#     def __init__(self, fname):
#         data_reader = pd.read_csv(fname, sep=',')
#         self.all_info_list = list()
#         is_test = 'test' in fname
#         if is_test:
#             for i in range(data_reader.last_valid_index() + 1):
#                 new_test_info = one_test_info(data_reader.loc[i], is_test)
#                 self.all_info_list.append(new_train_info)
#         else:
#             for i in range(data_reader.last_valid_index() + 1):
#                 new_train_info = one_train_info(data_reader.loc[i], is_test)
#                 self.all_info_list.append(new_train_info)
#         return


# class phi_holder(object):
#     def __init__(self, _phi, _id_list):
#         self.phi = _phi
#         self.id_list = _id_list


class model(object):
    # Might also need an id_list to ind map
    def __init__(self, feature_matrix, output_vec, _lamb, _p):
        self.lamb = _lamb
        self.p = _p

        self.id_np_list = np.arange(feature_matrix.shape[0])
        self.y_obs = np.array(output_vec)
        # self.Phi_h = phi_holder(phi_matrix, self.id_np_list)
        # Reminder to not forget the bias
        self.phi_matrix = feature_matrix
        self.weights = np.zeros(self.phi_matrix.shape[1])
        self.total_ind = self.phi_matrix.shape[0]

        self.lr = 1e-6

    def complete_loss(self):
        norm2_error = 0
        for i in range(self.total_ind):
            # norm2_error += (self.norm_error_one_ind(i))**2
            # l1 = np.dot(self.phi_matrix[i, :], self.weights) - self.y_obs[i]
            # pdb.set_trace()
            # print(norm2_error)
            l1 = self.norm_error_one_ind(i)
            norm2_error += l1 ** 2

        penalty = self.lamb * np.sum(np.power(np.abs(self.weights), self.p))
        # print(penalty)
        return norm2_error + penalty

    def norm_error_one_ind(self, ind):
        # note that it is not norm2 squared [Need to be careful]
        # currently accepting index, may need to change if it should accept the ID
        return np.dot(self.phi_matrix[ind, :], self.weights) - self.y_obs[ind]

    def one_weight_update(self, x_ind):
        new_weights = self.weights - self.lr * self.gradient_one_point(x_ind)
        return new_weights

    def gradient_one_point(self, x_ind):
        grad = np.zeros(self.weights.shape)
        # pdb.set_trace()
        for i in range(grad.shape[0]):
            # need x_ind for norm2_error_one_ind and w_ind for penaly_grad
            grad[i] = 2 * self.norm_error_one_ind(x_ind) * self.phi_matrix[x_ind, i] + \
                      self.penalty_grad(i)
        return grad

    def penalty_grad(self, w_ind):
        wj = self.weights[w_ind]
        return self.lamb * self.p * np.abs(wj)**(self.p-1) * np.sign(wj)

    def sgd(self):
        # Stochastic gradient descent
        # Assuming no shuffling for now
        for i in range(self.total_ind):
            self.weights = self.one_weight_update(i)
        return

    def p2_solver(self):
        tmp_mat = np.dot(self.phi_matrix.T, self.phi_matrix)
        tmp_mat += self.lamb * np.identity(self.phi_matrix.shape[1])
        tmp_mat2 = np.linalg.inv(tmp_mat)
        tmp_mat3 = np.dot(self.phi_matrix.T, self.y_obs)
        return np.dot(tmp_mat2, tmp_mat3)


if __name__ == '__main__':
    fname = '../data/train.csv'
    # fname = '../data/test_features.csv'
    feature_matrix = get_feature_matrix(fname)
    out_vec = get_output(fname)
    lambda_reg = 1
    p = 2
    # weight_vector = get_weight_vector(feature_matrix, out_vec, lambda_reg, p)
    # Regression with p-norm regularization
    # loss = ||y - y'||^2 + lambda * ||w||^p
    # Can try to use k-fold cross validation
    # to check if the sgd code is indeed working
    # do a sanity check for the case p=2
    # sgd_model = model(lamb, p, all_data)
    sgd_model = model(feature_matrix, out_vec, lambda_reg, p)
    # sgd_model.sgd()
