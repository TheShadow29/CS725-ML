from __future__ import print_function
from __future__ import division
import pandas as pd
import numpy as np
import pdb
from sgd_template import get_feature_matrix, get_output
import matplotlib.pyplot as plt
import re
import warnings
warnings.filterwarnings("error")
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

        self.lr = 1e-7

    # def reset_vars(self, feature_matrix, output_vec, )

    def complete_loss(self):
        norm2_error = 0
        for i in range(self.total_ind):
            # norm2_error += (self.norm_error_one_ind(i))**2
            # l1 = np.dot(self.phi_matrix[i, :], self.weights) - self.y_obs[i]
            # pdb.set_trace()
            # print(norm2_error)
            l1 = self.norm_error_one_ind(i)
            norm2_error += l1 ** 2
        norm2_error /= self.total_ind
        # do not consider bias term in penalty
        penalty = self.lamb * np.sum(np.power(np.abs(self.weights[:-1]), self.p))
        # print(penalty)
        # print(norm2_error, penalty, self.lr)
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
        g1 = 2 * self.norm_error_one_ind(x_ind)
        for i in range(grad.shape[0]):
            # need x_ind for norm2_error_one_ind and w_ind for penaly_grad
            try:
                g_tmp = g1 * self.phi_matrix[x_ind, i]
                g2 = self.penalty_grad(i)
                grad[i] = g_tmp + g2
            except Exception as e:
                print(g1, x_ind, i)
                print(self.weights)
                raise e
        return grad

    def penalty_grad(self, w_ind):
        wj = self.weights[w_ind]
        return self.lamb * self.p * np.abs(wj)**(self.p-1) * np.sign(wj)

    def sgd(self, nit=5, batch=1e2):
        # Stochastic gradient descent
        # Assuming no shuffling for now

        assert type(batch) is int
        curr_loss = self.complete_loss()
        prev_loss = curr_loss + 10000
        for it in range(nit):
            # it = 0
            # while (prev_loss - curr_loss > 100):
            # it += 1
            for i in range(self.total_ind):

                self.weights = self.one_weight_update(i)
                # curr_loss = self.complete_loss()
                if (i % batch == 0):
                    # print('i=', i)
                    # if True:
                    prev_loss = curr_loss
                    curr_loss = self.complete_loss()
                    print('i=', i, curr_loss, self.lr)
                    if (curr_loss < prev_loss - 10):
                        # print('Inc LR')
                        self.lr = 1.1 * self.lr
                    elif (curr_loss > prev_loss + 10):
                        # print('Dec LR')
                        self.lr = self.lr / 2

            print('loss', curr_loss, 'Iter', it)

            # if (curr_loss < prev_loss):
            #     self.lr = 2 * self.lr
            # else:
            #     self.lr = self.lr / 2
        return

    def p2_solver(self):
        tmp_mat = np.dot(self.phi_matrix.T, self.phi_matrix)
        tmp_mat += self.lamb * np.identity(self.phi_matrix.shape[1])
        tmp_mat2 = np.linalg.inv(tmp_mat)
        tmp_mat3 = np.dot(self.phi_matrix.T, self.y_obs)
        # self.weights = np.dot(tmp_mat2, tmp_mat3)
        return np.dot(tmp_mat2, tmp_mat3)

    def p2_direct_solve(self):
        self.weights = self.p2_solver()
        return

    def valid_accuracy(self, valid_phi, valid_output):
        tot_valid_ind = valid_phi.shape[0]
        err = 0
        for ind in range(tot_valid_ind):
            y_pred = np.dot(valid_phi[ind, :], self.weights)
            tmp_err = valid_output[ind] - y_pred
            err += tmp_err**2
        rmse = (err / tot_valid_ind)**(0.5)
        print(rmse)
        return rmse

    def test_printer(self, test_id_list, test_phi, out_fname):
        str_format = '{},{}\n'
        str_to_write = str_format.format('Id', 'Output')
        for ind, tid in enumerate(test_id_list):
            y_pred = np.dot(test_phi[ind, :], self.weights)
            str_to_write += str_format.format(tid, str(y_pred[0]))
        with open('../eval/' + out_fname, 'w') as f:
            f.write(str_to_write)

def plotter():
    dat_reader = pd.read_csv('../data/train.csv', sep=',')

    x_ax = np.array(pat.findall('\n'.join(dat_reader['date'])), dtype='int')
    pdb.set_trace()
    # for x in x_ax:
    #     x_ax_new.append(int(x.split(' ')[-1].split(':')[0]))

    plt.plot(x_ax[:,1], dat_reader['Output'])
    plt.show()


if __name__ == '__main__':

    pat = re.compile(r'(\d*)-(\d*)-(\d*)\s(\d*):(\d*):(\d*)')
    plotter()
    pdb.set_trace()
    fname = '../data/train.csv'
    # fname = '../data/test_features.csv'
    feature_matrix = get_feature_matrix(fname)
    out_vec = get_output(fname)
    lambda_reg = 1
    p = 1.5
    # weight_vector = get_weight_vector(feature_matrix, out_vec, lambda_reg, p)
    # Regression with p-norm regularization
    # loss = ||y - y'||^2 + lambda * ||w||^p
    # Can try to use k-fold cross validation
    # to check if the sgd code is indeed working
    # do a sanity check for the case p=2
    # sgd_model = model(lamb, p, all_data)
    tot_points = feature_matrix.shape[0]
    tr_pts = int(tot_points * 0.8)
    sgd_model = model(feature_matrix[:tr_pts], out_vec[:tr_pts], lambda_reg, p)
    sgd_model.sgd(10, 1000)
    # sgd_model.complete_loss()
    # test_feature_mat = get_feature_matrix('../data/test_features.csv')
    # sgd_model.p2_direct_solve()
    sgd_model.valid_accuracy(feature_matrix[tr_pts:], out_vec[tr_pts:])

    # test_fname = '../data/test_features.csv'
    # test_data_reader = pd.read_csv(test_fname, sep=',')
    # test_id_list = test_data_reader['Id']
    # test_feature_mat = get_feature_matrix(test_fname)
    # out_fname = 'test_eval_l_' + str(lambda_reg) + '_p_' + str(p) + '.csv'
    # sgd_model.test_printer(test_id_list, test_feature_mat, out_fname)
