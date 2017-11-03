from __future__ import print_function
from __future__ import division
import numpy as np
import pandas as pd
import pdb
import timeit
import warnings
warnings.filterwarnings('error')


class data_loader(object):
    def __init__(self, fname):
        self.fname = fname
        self.csv_data = pd.read_csv(self.fname, sep=',')
        self.curr_ind = 0
        self.last_loc = self.csv_data.last_valid_index()
        self.norm_data_params()

    def norm_data_params(self):
        data = np.array(self.csv_data)
        self.mean = np.mean(data, axis=0)
        self.mean[-1] = 0
        self.std = np.std(data, axis=0)
        self.std[-1] = 1
        return

    def normalize(self, data):
        return (data - self.mean)/self.std
        # new_dat[:, -1]
        # new_dat_x = (data[:-1] - self.mean)/self.std
        # return np.append(new_dat_x, data[-1])

    def next_item(self, num=1, columns='all'):
        # if columns == 'all':
        if self.curr_ind + num < self.last_loc:
            # pdb.set_trace()
            # Note : .loc[a:b] gives a and b inclusive
            out_dat = self.csv_data.loc[self.curr_ind:self.curr_ind + num - 1]
            self.curr_ind += num
        else:
            out_dat = self.csv_data.loc[self.curr_ind:]
            self.curr_ind = self.last_loc
        norm_d = self.normalize(np.array(out_dat))
        return norm_d
        # return np.array(out_dat)

    def reset_curr_ind(self, new_ind=0):
        self.curr_ind = new_ind
        return


def batch_to_one_hot(vec, one_hot_vec_len):
    one_hot_vec = np.zeros((vec.shape[0], one_hot_vec_len))
    # pdb.set_trace()
    for ac_ind, ac_d in enumerate(vec):
        one_hot_vec[ac_ind, ac_d] = 1
    return one_hot_vec


def cross_entropy(actual_data, pred_data):
    # pred data will be of shape batch_size x 10
    one_hot_vec_len = pred_data.shape[1]
    # batch_s = pred_data.shape[0]
    out1 = np.log(pred_data)
    # actual_data_one_hot = np.zeros((batch_s, one_hot_vec_len))
    # # actual_data_one_hot[:, actual_data] = 1
    # for ac_ind, ac_d in enumerate(actual_data):
    #     actual_data_one_hot[ac_ind, ac_d] = 1
    actual_data_one_hot = batch_to_one_hot(actual_data, one_hot_vec_len)
    out2 = -np.multiply(actual_data_one_hot, out1)
    out3 = np.sum(out2, axis=1)
    # grad_out = np.divide(actual_data_one_hot, pred_data)
    return out3


# def pre_process(inp_data):

# class node(object):
#     def __init__(self, gradient_back=0, curr_val=0):
#         self.curr_val = curr_val
#         self.gradient_back = gradient_back
#         return

#     def reset_grad(self):
#         self.gradient_back = 0

#     def reset_curr_val(self):
#         self.curr_val = 0

#     def reset_node(self):
#         self.gradient_back = 0
#         self.curr_val = 0
def sigmoid(a):
    # Assume a to be vector
    tmp_a = 1 + np.exp(-a)
    return 1./tmp_a


def sigmoid_grad(a):
    tmp_a = sigmoid(a)
    return np.multiply(tmp_a, (1 - tmp_a))


def relu(a):
    out = np.zeros(a.shape)
    out[a > 0] = a[a > 0]
    return out
    # if a > 0:
    #     return a
    # else:
    #     return np.zeros(a.shape)


def relu_grad(a):
    out = np.zeros(a.shape)
    # for i, a_i in enumerate(a):
    #     if a_i > 0:
    #         out[i] = 1
    out[a > 0] = 1
    return out


def softmax(x):
    # """Compute softmax values for each sets of scores in x."""
    # pdb.set_trace()
    try:
        # x0 = np.max(x)
        # x1 = x - x0
        e_x = np.exp(x - np.max(x))
    except Warning as e:
        pdb.set_trace()
        pass

    return e_x / e_x.sum()


def softmax_batch(x):
    n = x.shape[1]
    x11 = np.max(x, axis=1)
    e_x = np.exp(x - np.array([x11]*n).T)
    return e_x / np.sum(e_x, axis=1)[:, None]

# def softmax(inp):
#     # Assume inp is an array
#     # pdb.set_trace()
#     new_inp = inp - inp.mean()
#     n_t = np.exp(new_inp)
#     z_t = np.sum(n_t)
#     # outp = np.zeros(inp.shape)
#     # for ind, i in enumerate(inp):
#     #     new_inp = inp - i
#     #     # pdb.set_trace()
#     #     outp[ind] = 1./np.sum(np.exp(new_inp))

#     return n_t / z_t


def softmax_grad(q, l):
    # q = softmax(l)
    rows = q.shape[0]
    cols = l.shape[0]
    out_grad = np.zeros((rows, cols))
    # pdb.set_trace()
    for i in range(rows):
        for j in range(cols):
            if i != j:
                out_grad[i, j] = -q[i] * q[j]
            else:
                out_grad[i, j] = q[i]*(1 - q[i])
    return out_grad


class lin_layer(object):
    def __init__(self, inp_nodes, out_nodes, batch_size, act='lin'):
        # self.node_list = list()
        # for _ in range(num_nodes):
        #     new_node = node()
        #     self.node_list.append(new_node)
        # self.tot_nodes = len(self.node_list)
        self.bs = batch_size
        self.inp_nodes = inp_nodes
        self.out_nodes = out_nodes
        self.W = np.random.random((self.out_nodes, self.inp_nodes))/10
        # self.W = np.zeros((self.out_nodes, self.inp_nodes))
        # self.W = np.ones((self.out_nodes, self.inp_nodes))
        self.b = np.zeros(self.out_nodes)
        self.activation = act
        self.W_grad = np.zeros(self.W.shape)
        self.W_grad_tmp = np.zeros(self.W.shape)
        # self.W_grad_list = list()
        self.b_grad = np.zeros(self.b.shape)
        self.b_grad_tmp = np.zeros(self.b.shape)
        # self.b_grad_list = list()
        self.a = np.zeros(self.b.shape)
        self.a_mat = np.zeros((self.bs, self.b.shape[0]))
        self.act_grad = np.zeros(self.a.shape)
        self.act_grad_mat = np.zeros((self.bs, self.a.shape[0]))
        self.a_list = [np.zeros(self.a.shape) for i in range(self.bs)]
        self.h = np.zeros(self.b.shape)
        self.h_mat = np.zeros((self.bs, self.b.shape[0]))
        # self.h_list = list()
        self.h_list = [np.zeros(self.h.shape) for i in range(self.bs)]
        self.lr = 1e-2
        self.lr_it = 1
        self.gamma = 0.9
        self.vw_prev = 0
        self.vb_prev = 0
        self.forward_count = 0
        self.thresh = 0.01
        self.lambda_reg = 1e-7
        # pdb.set_trace()
        return

    def weights_l2(self):
        return np.linalg.norm(self.W, 'fro') + np.linalg.norm(self.b)
        # norm = 0
        # for n in self.node_list:
        #     norm += n.curr_val**2
        # return np.sqrt(norm)

    def forward(self, inp):
        self.a = np.dot(self.W, inp) + self.b
        # self.a_list.append(self.a)
        # self.a_list[self.forward_count] = self.a
        if self.activation == 'lin':
            self.h = self.a
            self.act_grad = np.ones(self.a.shape)
        if self.activation == 'sigmoid':
            self.h = sigmoid(self.a)
            self.act_grad = sigmoid_grad(self.a)
        if self.activation == 'relu':
            self.h = relu(self.a)
            self.act_grad = relu_grad(self.a)
        # self.h_list[self.forward_count] = self.h
        # self.h_list.append(self.h)
        # if self.activation == 'softmax':
        #     self.h = softmax(self.a)
        # self.forward_count += 1
        # if self.forward_count == self.bs:
            # self.forward_count = 0
        # pdb.set_trace()
        return self.h

    def forward_batch(self, inp):
        # self.a_mat = np.dot(self.W, inp) + self.b
        # pdb.set_trace()
        b_mat = np.array([self.b] * self.bs)
        # self.a_mat = np.dot(inp, self.W.T) + self.b
        self.a_mat = np.dot(inp, self.W.T) + b_mat
        if self.activation == 'lin':
            self.h_mat = self.a_mat
            self.act_grad_mat = np.ones(self.a.shape)
        if self.activation == 'sigmoid':
            self.h_mat = sigmoid(self.a_mat)
            self.act_grad_mat = sigmoid_grad(self.a)
        if self.activation == 'relu':
            self.h_mat = relu(self.a_mat)
            self.act_grad_mat = relu_grad(self.a_mat)
        # pdb.set_trace()
        return self.h_mat

    def backward(self, g, h_prev, ind):
        # g is the backprop grad from the layer above
        # g is assumed to be a vector
        # pdb.set_trace()
        if self.activation == 'lin':
            act_grad = np.ones(self.a_list[ind].shape)
        if self.activation == 'sigmoid':
            act_grad = sigmoid_grad(self.a_list[ind])
        if self.activation == 'relu':
            # print('back relu')
            act_grad = relu_grad(self.a_list[ind])
        # if self.activation == 'softmax':
        #     act_grad = softmax_grad(self.a)
        # pdb.set_trace()
        g_new = np.multiply(g, act_grad)
        b_gradt = g_new
        W_gradt = np.dot(g_new[:, None], h_prev[None, :]) + 2 * 1e-7 * self.W
        # self.W_grad_list.append(W_gradt)
        # self.b_grad_list.append(b_gradt)
        self.W_grad_tmp += W_gradt
        self.b_grad_tmp += b_gradt
        g_out = np.dot(g_new, self.W)
        g_out = np.squeeze(g_out)
        return g_out

    def back(self, g_in, h_prev):
        if self.activation == 'lin':
            act_grad = np.ones(self.a.shape)
        if self.activation == 'sigmoid':
            act_grad = sigmoid_grad(self.a)
        if self.activation == 'relu':
            act_grad = relu_grad(self.a)
        # pdb.set_trace()
        g_new = np.multiply(g_in, act_grad)
        b_gradt = g_new
        W_gradt = np.dot(g_new[:, None], h_prev[None, :]) + 2 * 1e-7 * self.W
        self.W_grad_tmp += W_gradt
        self.b_grad_tmp += b_gradt
        g_out = np.dot(g_new, self.W)
        g_out = np.squeeze(g_out)
        # pdb.set_trace()
        return g_out

    def back_b(self, g, h_prev):
        if self.activation == 'lin':
            act_grad_mat = np.ones(self.a_mat.shape)
        if self.activation == 'sigmoid':
            act_grad_mat = sigmoid_grad(self.a_mat)
        if self.activation == 'relu':
            act_grad_mat = relu_grad(self.a_mat)

        # pdb.set_trace()
        # g_new = np.multiply(g, self.act_grad_mat)
        g_new = np.multiply(g, act_grad_mat)
        b_gradt_mat = g_new
        # W_gradt_mat = np.dot(g_new.T, h_prev) + 2 * 100 * self.W
        # W_gradt_mat = np.dot(g_new.T, h_prev) + 2 * 100 * self.W
        W_gradt_mat = np.dot(g_new.T, h_prev) + 2 * self.lambda_reg * self.W
        # W_grad_mat = np.zeros(self.W_grad)
        # pdb.set_trace()
        self.b_grad_tmp = np.sum(b_gradt_mat, axis=0)
        self.W_grad_tmp = W_gradt_mat
        if (np.log10(np.abs(self.W_grad_tmp)) > 10).any():
            pdb.set_trace()
        g_out = np.dot(g_new, self.W)
        # g_out = np.squeeze(g_out)
        # pdb.set_trace()
        return g_out

    def update_weights(self, optim='sgd'):
        curr_batch_size = self.bs
        self.W_grad = self.W_grad_tmp / curr_batch_size
        # self.W_grad[self.W_grad > self.thresh] = self.thresh
        self.b_grad = self.b_grad_tmp / curr_batch_size

        if optim == 'sgd':
            # pdb.set_trace()
            self.W = self.W - self.lr * self.W_grad / self.lr_it
            self.b = self.b - self.lr * self.b_grad / self.lr_it
            # self.lr_it += 1
            self.clear_lists()
            return

        elif optim == 'mom':
            self.vw_cur = self.gamma * self.vw_prev + self.lr * self.W_grad / self.lr_it
            self.W = self.W - self.vw_cur
            self.vb_cur = self.gamma * self.vb_prev + self.lr * self.b_grad / self.lr_it
            self.b = self.b - self.vb_cur
            self.vw_prev = self.vw_cur
            self.vb_prev = self.vb_cur
            # self.lr_it += 1     #
            # pdb.set_trace()
            self.clear_lists()
            return

    def clear_lists(self):
        self.W_grad_tmp[...] = 0
        self.b_grad_tmp[...] = 0
        # del self.W_grad_list[:]
        # del self.b_grad_list[:]
        # del self.h_list[:]
        return


class neural_network(object):
    def __init__(self, inp_nodes, batch_size):
        self.outp_nodes = 10
        self.batch_size = batch_size
        self.list_layers = list()
        fc1 = lin_layer(inp_nodes, 100, batch_size, 'relu')
        self.list_layers.append(fc1)
        fc2 = lin_layer(100, 100, batch_size, 'relu')
        self.list_layers.append(fc2)
        fc3 = lin_layer(100, self.outp_nodes, batch_size, 'lin')
        self.list_layers.append(fc3)
        self.tot_layers = len(self.list_layers)
        self.y_pred_list = np.zeros((self.batch_size, self.outp_nodes))
        self.eps = 1e-5
        self.all_z = np.zeros(self.outp_nodes)
        self.lr = 0
        self.lambda_reg = 0
        return

    def set_lr(self, lr):
        for l in self.list_layers:
            l.lr = lr
        self.lr = lr
        return

    def set_lamb(self, lamb):
        for l in self.list_layers:
            l.lambda_reg = lamb
        self.lambda_reg = lamb
        return

    def weights_l2(self):
        norm = 0
        for l in self.list_layers:
            norm += l.weights_l2() ** 2
        return np.sqrt(norm)

    def train_nicely(self, xtd, ytd):

        for i in range(self.batch_size):
            # self.y_pred_list[i, :] = self.feed_forward(xtd[i])
            y_pred_tmp = self.feed_forward(xtd[i])
            # y_pred_tmp[y_pred_tmp < self.eps] = 1e-5
            g_init = np.zeros(self.outp_nodes)
            g_init[ytd[i]] = 1
            g_init = - g_init + y_pred_tmp
            # pdb.set_trace()
            self.bp(g_init, xtd[i])
            self.y_pred_list[i] = y_pred_tmp
        # self.y_pred_list[self.y_pred_list < self.eps] = 1e-5
        return self.y_pred_list

    def train_better(self, xtd, ytd):
        y_pred_tmp = self.feed_forward_batch(xtd)
        # y_pred_tmp[y_pred_tmp < self.eps] = 1e-5
        g_init = batch_to_one_hot(ytd, self.outp_nodes)
        g_init = -g_init + y_pred_tmp
        self.back_batch(g_init, xtd)
        return y_pred_tmp

    def back_batch(self, g_init, inp):
        g_prev = g_init
        for l in range(self.tot_layers-1, -1, -1):
            h_prev_mat = self.list_layers[l-1].h_mat
            if l == 0:
                h_prev_mat = inp
            g_prev = self.list_layers[l].back_b(g_prev, h_prev_mat)

    def feed_forward_batch(self, inp):
        # pdb.set_trace()
        out = self.list_layers[0].forward_batch(inp)
        out = self.list_layers[1].forward_batch(out)
        out = self.list_layers[2].forward_batch(out)
        # if np.isinf(out).any():
        # pdb.set_trace()
        out = softmax_batch(out)
        return out

    def test_nicely(self, xtd):
        # pdb.set_trace()
        y_test_list = list()
        for i in range(xtd.shape[0]):
            # self.y_pred_list[i, :] = self.feed_forward(xtd[i])
            y_pred_tmp = self.feed_forward(xtd[i])
            y_pred_tmp[y_pred_tmp < self.eps] = 1e-5
            y_test_list.append(y_pred_tmp)
            # g_init = np.zeros(self.outp_nodes)
            # g_init[ytd[i]] = 1
            # g_init = - g_init + y_pred_tmp
            # pdb.set_trace()
            # self.bp(g_init, xtd[i])
        # self.y_pred_list[self.y_pred_list < self.eps] = 1e-5

        return np.argmax(np.array(y_test_list), axis=1)

    def bp(self, g_init, inp):
        g_prev = g_init
        for l in range(self.tot_layers-1, -1, -1):
            h_prev = self.list_layers[l-1].h
            if l == 0:
                h_prev = inp
            g_prev = self.list_layers[l].back(g_prev, h_prev)

    def feed_forward(self, inp):
        # pdb.set_trace()
        out = self.list_layers[0].forward(inp)
        out = self.list_layers[1].forward(out)
        out = self.list_layers[2].forward(out)
        # if np.isinf(out).any():
        # pdb.set_trace()
        out = softmax(out)
        return out

    def get_output(self, inp):
        rows = inp.shape[0]
        out_list = list()
        for r in range(rows):
            out = self.feed_forward(inp[r, :])
            out_list.append(out)
        return np.array(out_list)

    def back_prop(self, g_init_list, inp_list):
        for ind, g_ in enumerate(g_init_list):
            # pdb.set_trace()
            self.back_prop_one_g(g_, inp_list[ind], ind)
        return

    def back_prop_one_g(self, g_init, inp, ind):
        g_prev = g_init
        for i in range(self.tot_layers-1, -1, -1):
            # pdb.set_trace()
            h_prev = self.list_layers[i-1].h_list[ind]
            if i == 0:
                h_prev = inp
            g_next = self.list_layers[i].backward(g_prev, h_prev, ind)
            g_prev = g_next

        return

    def update_weights(self, optim='sgd'):
        for l in self.list_layers:
            l.update_weights(optim)


class model(object):
    def __init__(self, nn, train_data, test_data, batch_size, loss_fn='cel2', optimizer='sgd'):
        self.nn = nn
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_data = train_data
        self.test_data = test_data
        self.train_frac = 0.8
        self.num_train_data_tot = self.train_data.last_loc + 1
        self.num_train_data = int(self.num_train_data_tot * self.train_frac)
        self.batch_size = batch_size
        # self.val_data_num = self.num_train_data_tot - self.num_train_data
        self.val_data_num = 500
        self.val_start_id = 0
        self.y_pred_list = [0 for i in range(self.val_data_num)]
        self.y_val_data_list = [0 for i in range(self.val_data_num)]

        return

    def train_net(self, num_epoch=30):
        # pdb.se
        for epoch in range(num_epoch):
            for it in np.arange(0, self.num_train_data_tot, self.batch_size):
                curr_loss = self.train_iter2()
                if it % 100000 == 0:
                    print(it / 100000)
                #     val_acc = self.validation_net()
                #     print(it / 10000, curr_loss, val_acc)
                #     print(self.train_data.curr_ind)
            if True:
                val_acc = self.validation_net()
                print(epoch, curr_loss, val_acc)
                print(self.train_data.curr_ind)
                # print(val_acc)

            self.train_data.reset_curr_ind(0)
        return

    def train_iter1(self):
        # pdb.set_trace()
        train_data_new = self.train_data.next_item(num=self.batch_size)
        # pdb.set_trace()
        x_train_data = train_data_new[:, :-1]
        y_train_data = train_data_new[:, -1].astype(np.int)
        y_pred_data = self.nn.get_output(x_train_data)
        # pdb.set_trace()
        curr_loss = self.loss(y_train_data, y_pred_data)
        g_init_list = self.g_init_list(y_train_data, y_pred_data)
        # pdb.set_trace()
        self.nn.back_prop(g_init_list, x_train_data)
        # self.nn.update_weights(optim='sgd')
        self.nn.update_weights(optim='mom')
        # print(curr_loss, g_init_list)
        # print(curr_loss)
        return curr_loss

    def train_iter2(self):
        train_data_new = self.train_data.next_item(num=self.batch_size)
        x_train_data = train_data_new[:, :-1]
        y_train_data = train_data_new[:, -1].astype(np.int)
        # y_pred_data1 = self.nn.train_nicely(x_train_data, y_train_data)
        # pdb.set_trace()
        y_pred_data = self.nn.train_better(x_train_data, y_train_data)
        # pdb.set_trace()
        curr_loss = self.loss(y_train_data, y_pred_data)
        self.nn.update_weights(optim='mom')
        # self.nn.update_weights(optim='sgd')
        return curr_loss

    def test_net(self):
        self.test_data.reset_curr_ind()
        test_data_new = self.test_data.next_item(num=self.test_data.last_loc + 1)
        x_test_data = test_data_new[:]
        y_pred_data = self.nn.test_nicely(x_test_data)
        fname = 'test_out.txt'
        str_out = ''
        str_out_format = '{},{}\n'
        str_out += str_out_format.format('id', 'predicted_class')
        with open(fname, 'w') as f:
            for ind, yp in enumerate(y_pred_data):
                str_out += str_out_format.format(ind, yp)
            f.write(str_out)
        # pdb.set_trace()

    def validation_net(self):
        num_corr = 0
        tot_num = 0
        tr_curr_ind = self.train_data.curr_ind
        print('val')
        self.train_data.reset_curr_ind(self.num_train_data)
        # for it in np.arange(self.num_train_data, self.num_train_data_tot):
        val_data_new = self.train_data.next_item(num=self.val_data_num)
        x_val_data = val_data_new[:, :-1]
        y_val_data = val_data_new[:, -1].astype(int)
        y_pred_dist = self.nn.get_output(x_val_data)
        # pdb.set_trace()
        y_pred = np.argmax(y_pred_dist, axis=1)
        # for it in np.arange(self.num_train_data, self.num_train_data + val_data_num):
        for it in range(self.val_data_num):
            # pdb.set_trace()
            if y_pred[it] == y_val_data[it]:
                num_corr += 1
            tot_num += 1
            self.y_pred_list.append(y_pred)
            self.y_val_data_list.append(y_val_data[0])
        assert tot_num == self.val_data_num
        self.train_data.reset_curr_ind(tr_curr_ind)
        # pdb.set_trace()
        return num_corr / tot_num

    def loss(self, actual_data, pred_data):
        ce_loss = cross_entropy(actual_data, pred_data)
        l2_loss = self.nn.weights_l2()
        # pdb.set_trace()
        # return np.sum(ce_loss) + l2_loss
        return np.sum(ce_loss)/ce_loss.shape[0] + l2_loss

    def g_init_list(self, actual_data, pred_data):
        actual_data_one_hot = batch_to_one_hot(actual_data, pred_data.shape[1])
        g_init_list = -(actual_data_one_hot - pred_data)
        return g_init_list
        # actual_data_one_hot = batch_to_one_hot(actual_data, pred_data.shape[1])
        # g_init_list = -np.divide(actual_data_one_hot, pred_data)
        # # g_init_tmp = np.sum(tmp_out, axis=0)
        # # pdb.set_trace()
        # h_last = self.nn.list_layers[-1].h
        # g_init_out_list = list()
        # for i in range(pred_data.shape[0]):
        #     grad_softmax = softmax_grad(pred_data[i, :], h_last)
        #     g_init_out = np.dot(g_init_list[i, :], grad_softmax)
        #     g_init_out_list.append(g_init_out)
        # return g_init_out_list


if __name__ == '__main__':
    np.random.seed(0)
    train_fname = '../data/train.csv'
    test_fname = '../data/test.csv'
    train_data = data_loader(train_fname)
    test_data = data_loader(test_fname)
    batch_size = 200
    simple_net = neural_network(10, batch_size)
    simple_model = model(simple_net, train_data, test_data, batch_size)
    # (learning_rate, num_hidden_layers, regularizer_lambda):
    # learning_rate = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    # num_hidden_layers = [1, 2, 3, 4, 5]
    # regularizer_lambda = [100, 10, 1, 1e-1, 1e-2]
    # for lr in learning_rate:
    #     simple_model.nn.set_lr(lr)
    #     for reg in regularizer_lambda:
    #         simple_model.nn.set_lamb(reg)

    # train_csv_data = pd.read_csv(train_fname, sep=',')
    # test_csv_data = pd.read_csv(test_fname, sep=',')
    # Things to try :
    # Data augmentation using shuffling
    # batch sgd
    # normalization over training data
    # k-fold validation for hyp param
    # loss_fn with l2 reg
    # momentum update rule
    # num_hidden layers
    # different activations
