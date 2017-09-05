from __future__ import print_function
from __future__ import division
# import pandas as pd
import numpy as np
# import pdb
# from sgd_template import get_feature_matrix, get_output
# import matplotlib.pyplot as plt
# import re
import pickle


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
        self.weights = np.zeros((self.phi_matrix.shape[1], 1))
        self.total_ind = self.phi_matrix.shape[0]

        self.lr = 1e-12
        self.init_lr = 1e-12
        self.dec_lr_count = 1
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
        if self.p == 2:
            return self.p2_direct_solve()
        if self.p == 1:
            self.p1_solver()
            return
        assert type(batch) is int
        curr_loss = self.complete_loss()
        prev_loss = curr_loss + 10000
        # old_weights = self.weights
        # self.dec_lr_count = 1
        same_lr = 0
        # iter_arr = np.arange(nit)
        # for it in range(nit):
        it = 0
        curr_weights = self.weights
        prev_weights = curr_weights + 10

        while not np.allclose(prev_weights, curr_weights) or it > nit:

            # while (prev_loss - curr_loss > 100):
            it += 1
            ind_arr = np.arange(self.total_ind)
            if it > 0:
                np.random.shuffle(ind_arr)
            # for i in range(self.total_ind):
            for i_n, i in enumerate(ind_arr):
                self.weights = self.one_weight_update(i)
                # curr_loss = self.complete_loss()
                if (i_n % batch == 0):
                    # print('i=', i)
                    # if True:
                    prev_loss = curr_loss
                    curr_loss = self.complete_loss()
                    print('i=', i_n, curr_loss, self.lr)
                    prev_weights = curr_weights
                    curr_weights = self.weights
                    if (curr_loss < prev_loss - 10):
                        print('Inc LR')
                        # old_weights = self.weights
                        self.lr = 1.1 * self.lr
                        self.init_lr = max(self.init_lr, self.lr)
                    elif (curr_loss > prev_loss + 10):
                        print('Dec LR')
                        self.dec_lr_count += 1
                        # self.weights = old_weights
                        self.lr = self.init_lr / self.dec_lr_count
                        same_lr = 0
                    elif curr_loss > prev_loss:
                        same_lr += 1
                        if same_lr == 2:
                            same_lr = 0
                            self.dec_lr_count += 1
                            self.lr = self.init_lr / self.dec_lr_count
            print('loss', curr_loss, 'Iter', it)

            # if (curr_loss < prev_loss):
            #     self.lr = 2 * self.lr
            # else:
            #     self.lr = self.lr / 2
        return

    # def ista(self):
    def p2_solver(self):
        tmp_mat = np.dot(self.phi_matrix.T, self.phi_matrix)
        tmp_mat += self.lamb * np.identity(self.phi_matrix.shape[1])
        tmp_mat2 = np.linalg.inv(tmp_mat)
        tmp_mat3 = np.dot(self.phi_matrix.T, self.y_obs)
        # self.weights = np.dot(tmp_mat2, tmp_mat3)
        return np.dot(tmp_mat2, tmp_mat3)

    def p2_direct_solve(self):
        # pdb.set_trace()
        self.weights = self.p2_solver()
        # pdb.set_trace()
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
        with open('../eval/' + out_fname[:-4] + '.pkl', 'wb') as g:
            pickle.dump(self.weights, g)
        return

    def soft_function(self, y_arr, lamb):
        out_vec = np.zeros(y_arr.shape)
        for ind, y in enumerate(y_arr):
            if y >= lamb:
                out_vec[ind] = y - lamb
            elif y <= -lamb:
                out_vec[ind] = y + lamb
            else:
                out_vec[ind] = 0
        return out_vec

    def p1_solver(self):
        # for p=1, algorithm employed is ISTA
        # Iterative Soft Thresholding Algorithm
        k = 0
        # eps = 1e-10
        eig_values, eig_vecs = np.linalg.eig(np.dot(self.phi_matrix.T, self.phi_matrix))
        alpha = 1.1 * np.max(eig_values)
        curr_weights = self.weights
        prev_weights = curr_weights + 10
        # while np.linalg.norm(prev_weights - curr_weights) > eps:
        while not np.allclose(prev_weights, curr_weights):
            # print('Prev_w', prev_weights)
            # print('Curr_w', curr_weights)
            k += 1
            prev_weights = curr_weights
            tmp1 = self.weights
            tmp2 = np.dot(self.phi_matrix.T, self.y_obs - np.dot(self.phi_matrix, self.weights))
            # pdb.set_trace()
            tmp3 = tmp1 + tmp2 / alpha
            curr_weights = self.soft_function(tmp3, self.lamb / (2 * alpha))
            self.weights = curr_weights
            print('Iter', k, 'loss', self.complete_loss())
        return

# def plotter():
#     dat_reader = pd.read_csv('../data/train.csv', sep=',')

#     x_ax = np.array(pat.findall('\n'.join(dat_reader['date'])), dtype='int')
#     pdb.set_trace()
#     # for x in x_ax:
#     #     x_ax_new.append(int(x.split(' ')[-1].split(':')[0]))

#     plt.plot(x_ax[:, 3], dat_reader['Output'])
#     plt.show()


# if __name__ == '__main__':

#     # pat = re.compile(r'(\d*)-(\d*)-(\d*)\s(\d*):(\d*):(\d*)')
#     # plotter()
#     # pdb.set_trace()
#     fname = '../data/train.csv'
#     # fname = '../data/test_features.csv'
#     feature_matrix = get_feature_matrix(fname)
#     out_vec = get_output(fname)
#     lambda_reg = 1
#     p = 1
#     # weight_vector = get_weight_vector(feature_matrix, out_vec, lambda_reg, p)
#     # Regression with p-norm regularization
#     # loss = ||y - y'||^2 + lambda * ||w||^p
#     # Can try to use k-fold cross validation
#     # to check if the sgd code is indeed working
#     # do a sanity check for the case p=2
#     # sgd_model = model(lamb, p, all_data)
#     tot_points = feature_matrix.shape[0]
#     tr_pts = int(tot_points * 0.9)
#     sgd_model = model(feature_matrix[:tr_pts], out_vec[:tr_pts], lambda_reg, p)
#     sgd_model.p2_direct_solve()
#     # sgd_model.sgd(1000, 1000)
#     sgd_model.p1_solver()
#     print(sgd_model.complete_loss())  #
#     # test_feature_mat = get_feature_matrix('../data/test_features.csv')
#     # sgd_model.p2_direct_solve()
#     sgd_model.valid_accuracy(feature_matrix[tr_pts:], out_vec[tr_pts:])
#     # sgd_model.valid_accuracy(feature_matrix, out_vec)
#     test_fname = '../data/test_features.csv'
#     test_data_reader = pd.read_csv(test_fname, sep=',')
#     test_id_list = test_data_reader['Id']
#     test_feature_mat = get_feature_matrix(test_fname)
#     out_fname = 'test_eval_l_' + str(lambda_reg) + '_p_' + str(p) + '_tr_' + str(tr_pts) + '.csv'
#     sgd_model.test_printer(test_id_list, test_feature_mat, out_fname)
