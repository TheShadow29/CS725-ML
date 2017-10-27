from __future__ import print_function
from __future__ import division
import numpy as np
import pandas as pd
import pdb


class data_loader(object):
    def __init__(self, fname):
        self.fname = fname
        self.csv_data = pd.read_csv(self.fname, sep=',')
        self.curr_ind = 0
        self.last_loc = self.csv_data.last_valid_index()

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
        return np.array(out_dat)

    def reset_curr_ind(self, new_ind=0):
        self.curr_ind = new_ind
        return


def batch_to_one_hot(vec, one_hot_vec_len):
    one_hot_vec = np.zeros((vec.shape[0], one_hot_vec_len))
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
    if a > 0:
        return a
    else:
        return np.zeros(a.shape)


def relu_grad(a):
    out = np.zeros(a.shape)
    for i, a_i in enumerate(a):
        if a_i > 0:
            out[i] = 1
    return out


def softmax(inp):
    # Assume inp is an array
    # pdb.set_trace()
    # n_t = np.exp(inp)
    # z_t = np.sum(n_t)
    outp = np.zeros(inp.shape)
    for ind, i in enumerate(inp):
        new_inp = inp - i
        outp[ind] = 1./np.sum(np.exp(new_inp))

    return outp


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
    def __init__(self, inp_nodes, out_nodes, act='lin'):
        # self.node_list = list()
        # for _ in range(num_nodes):
        #     new_node = node()
        #     self.node_list.append(new_node)
        # self.tot_nodes = len(self.node_list)
        self.inp_nodes = inp_nodes
        self.out_nodes = out_nodes
        self.W = np.random.random((self.out_nodes, self.inp_nodes))
        # self.W = np.zeros((self.out_nodes, self.inp_nodes))
        # self.W = np.ones((self.out_nodes, self.inp_nodes))
        self.b = np.zeros(self.out_nodes)
        self.activation = act
        self.W_grad = np.zeros(self.W.shape)
        self.W_grad_list = list()
        self.b_grad = np.zeros(self.b.shape)
        self.b_grad_list = list()
        self.a = np.zeros(self.b.shape)
        self.h = np.zeros(self.b.shape)
        self.lr = 1e-2
        self.lr_it = 1
        return

    def weights_l2(self):
        return np.linalg.norm(self.W, 'fro') + np.linalg.norm(self.b)
        # norm = 0
        # for n in self.node_list:
        #     norm += n.curr_val**2
        # return np.sqrt(norm)

    def forward(self, inp):
        self.a = np.dot(self.W, inp) + self.b
        if self.activation == 'lin':
            self.h = self.a
        if self.activation == 'sigmoid':
            self.h = sigmoid(self.a)
        if self.activation == 'relu':
            self.h = relu(self.a)
        # if self.activation == 'softmax':
        #     self.h = softmax(self.a)
        return self.h

    def backward(self, g, h_prev):
        # g is the backprop grad from the layer above
        # g is assumed to be a vector
        # pdb.set_trace()
        if self.activation == 'lin':
            act_grad = np.ones(self.a.shape)
        if self.activation == 'sigmoid':
            act_grad = sigmoid_grad(self.a)
        if self.activation == 'relu':
            act_grad = relu_grad(self.a)
        # if self.activation == 'softmax':
        #     act_grad = softmax_grad(self.a)
        # pdb.set_trace()
        g_new = np.multiply(g, act_grad)
        b_gradt = g_new
        W_gradt = np.dot(g_new[:, None], h_prev[None, :]) + 2 * self.W
        self.W_grad_list.append(W_gradt)
        self.b_grad_list.append(b_gradt)
        g_out = np.dot(g_new, self.W)
        g_out = np.squeeze(g_out)
        return g_out

    def update_weights(self, optim='sgd'):
        if optim == 'sgd':
            # pdb.set_trace()
            curr_batch_size = len(self.W_grad_list)
            self.W_grad = np.sum(self.W_grad_list, axis=0) / curr_batch_size
            self.b_grad = np.sum(self.b_grad_list, axis=0) / curr_batch_size
            self.W = self.W - self.lr * self.W_grad / self.lr_it
            self.b = self.b - self.lr * self.b_grad / self.lr_it
            self.lr_it += 1
            self.clear_grad_lists()
            return

    def clear_grad_lists(self):
        del self.W_grad_list[:]
        del self.b_grad_list[:]
        return


class neural_network(object):
    def __init__(self, inp_nodes):
        self.list_layers = list()
        fc1 = lin_layer(inp_nodes, 10)
        self.list_layers.append(fc1)
        fc2 = lin_layer(10, 10)
        self.list_layers.append(fc2)
        self.tot_layers = len(self.list_layers)
        return

    def weights_l2(self):
        norm = 0
        for l in self.list_layers:
            norm += l.weights_l2() ** 2
        return np.sqrt(norm)

    def feed_forward(self, inp):
        # pdb.set_trace()
        out = self.list_layers[0].forward(inp)
        out = self.list_layers[1].forward(out)
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
            self.back_prop_one_g(g_, inp_list[ind])
        return

    def back_prop_one_g(self, g_init, inp):
        g_prev = g_init
        for i in range(self.tot_layers-1, -1, -1):
            # pdb.set_trace()
            h_prev = self.list_layers[i-1].h
            if i == 0:
                h_prev = inp
            g_next = self.list_layers[i].backward(g_prev, h_prev)
            g_prev = g_next

        return

    def update_weights(self, optim='sgd'):
        for l in self.list_layers:
            l.update_weights(optim)


class model(object):
    def __init__(self, nn, train_data, test_data, loss_fn='cel2', optimizer='sgd'):
        self.nn = nn
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_data = train_data
        self.test_data = test_data
        self.train_frac = 0.8
        self.num_train_data_tot = self.train_data.last_loc + 1
        self.num_train_data = self.num_train_data_tot * self.train_frac
        return

    def train_net(self, num_epoch=15, batch_size=4):

        for epoch in range(num_epoch):
            for it in np.arange(0, self.num_train_data, batch_size):
                curr_loss = self.train_iter1(batch_size)
                if it % 10000 == 0:
                    val_acc = self.validation_net()
                    print(it / 10000, curr_loss, val_acc)
                    # print(val_acc)
        return

    def train_iter1(self, batch_size=4):
        train_data_new = self.train_data.next_item(num=batch_size)
        x_train_data = train_data_new[:, :-1]
        y_train_data = train_data_new[:, -1]
        y_pred_data = self.nn.get_output(x_train_data)
        # pdb.set_trace()
        curr_loss = self.loss(y_train_data, y_pred_data)
        g_init_list = self.g_init_list(y_train_data, y_pred_data)
        self.nn.back_prop(g_init_list, x_train_data)
        self.nn.update_weights(optim='sgd')
        # print(curr_loss, g_init_list)
        # print(curr_loss)
        return curr_loss

    def validation_net(self):
        num_corr = 0
        tot_num = 0
        self.train_data.reset_curr_ind(self.num_train_data)
        val_data_num = 500
        # for it in np.arange(self.num_train_data, self.num_train_data_tot):
        for it in np.arange(self.num_train_data, self.num_train_data + val_data_num):
            val_data_new = self.train_data.next_item(num=1)
            x_val_data = val_data_new[:, :-1]
            y_val_data = val_data_new[:, -1]
            y_pred_dist = self.nn.get_output(x_val_data)
            y_pred = np.argmax(y_pred_dist)
            # pdb.set_trace()
            if y_pred == y_val_data[0]:
                num_corr += 1
            tot_num += 1
        assert tot_num == val_data_num
        return num_corr / tot_num

    def loss(self, actual_data, pred_data):
        ce_loss = cross_entropy(actual_data, pred_data)
        l2_loss = self.nn.weights_l2()
        # pdb.set_trace()
        # return np.sum(ce_loss) + l2_loss
        return np.sum(ce_loss)/ce_loss.shape[0] + l2_loss

    def g_init_list(self, actual_data, pred_data):
        actual_data_one_hot = batch_to_one_hot(actual_data, pred_data.shape[1])
        g_init_list = -np.divide(actual_data_one_hot, pred_data)
        # g_init_tmp = np.sum(tmp_out, axis=0)
        # pdb.set_trace()
        h_last = self.nn.list_layers[-1].h
        g_init_out_list = list()
        for i in range(pred_data.shape[0]):
            grad_softmax = softmax_grad(pred_data[i, :], h_last)
            g_init_out = np.dot(g_init_list[i, :], grad_softmax)
            g_init_out_list.append(g_init_out)
        return g_init_out_list


if __name__ == '__main__':
    np.random.seed(0)
    train_fname = '../data/train.csv'
    test_fname = '../data/test.csv'
    train_data = data_loader(train_fname)
    test_data = data_loader(test_fname)
    simple_net = neural_network(10)
    simple_model = model(simple_net, train_data, test_data)
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
