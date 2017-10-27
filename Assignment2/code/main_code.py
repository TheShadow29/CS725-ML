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

    def reset_curr_ind(self):
        self.curr_ind = 0
        return


def cross_entropy(actual_data, pred_data):
    # pred data will be of shape batch_size x 10
    one_hot_vec_len = pred_data.shape[1]
    batch_s = pred_data.shape[0]
    out1 = np.log(pred_data)
    actual_data_one_hot = np.zeros((batch_s, one_hot_vec_len))
    actual_data_one_hot[:, actual_data] = 1
    out2 = -np.multiply(actual_data_one_hot, out1)
    return np.sum(out2)


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


class lin_layer(object):
    def __init__(self, inp_nodes, out_nodes):
        # self.node_list = list()
        # for _ in range(num_nodes):
        #     new_node = node()
        #     self.node_list.append(new_node)
        # self.tot_nodes = len(self.node_list)
        self.inp_nodes = inp_nodes
        self.out_nodes = out_nodes
        self.W = np.zeros((self.out_nodes, self.inp_nodes))
        self.b = np.zeros(self.out_nodes)
        return

    def weights_l2(self):
        return np.linalg.norm(self.W, 'fro') + np.linalg.norm(self.b)
        # norm = 0
        # for n in self.node_list:
        #     norm += n.curr_val**2
        # return np.sqrt(norm)

    def forward(self, inp):
        return np.dot(self.W, inp) + self.b


class neural_network(object):
    def __init__(self, inp_nodes):
        self.list_layers = list()
        fc1 = lin_layer(inp_nodes, 50)
        self.list_layers.append(fc1)
        fc2 = lin_layer(50, 10)
        self.list_layers.append(fc2)
        return

    def weights_l2(self):
        norm = 0
        for l in self.list_layers:
            norm += l.weights_l2() ** 2
        return np.sqrt(norm)

    def softmax(self, inp):
        # Assume inp is an array
        n_t = np.exp(inp)
        z_t = np.sum(n_t)
        return n_t / z_t

    def feed_forward(self, inp):
        # pdb.set_trace()
        out = self.list_layers[0].forward(inp)
        out = self.list_layers[1].forward(out)
        out = self.softmax(out)
        return out

    def get_output(self, inp):
        rows = inp.shape[0]
        out_list = list()
        for r in range(rows):
            out = self.feed_forward(inp[r, :])
            out_list.append(out)
        return np.array(out_list)


class model(object):
    def __init__(self, nn, train_data, test_data, loss_fn='cel2', optimizer='sgd'):
        self.nn = nn
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_data = train_data
        self.test_data = test_data

    def train_net(self, num_epoch=15, batch_size=4):
        num_train_data = self.train_data.last_loc + 1
        for epoch in range(num_epoch):
            for _ in np.arange(0, num_train_data, batch_size):
                train_data_new = self.train_data.next_item(num=batch_size)
                x_train_data = train_data_new[:, :-1]
                y_train_data = train_data_new[:, -1]
                y_pred_data = self.nn.get_output(x_train_data)
                # pdb.set_trace()
                curr_loss = self.loss(y_train_data, y_pred_data)
                # self.nn.update_weights()
                print(curr_loss)

    # def test_net(self):
    #     num_corr = 0
    #     tot_num = 0
    #     num_train_data = self.train_data.last_loc + 1
    #     for _ in np.arange(0, num_train_data, batch_size):
    #         test_data_new

    def loss(self, actual_data, pred_data):
        ce_loss = cross_entropy(actual_data, pred_data)
        l2_loss = self.nn.weights_l2()
        return ce_loss + l2_loss


if __name__ == '__main__':
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
