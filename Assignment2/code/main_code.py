import numpy as np
import pandas as pd


class data_loader(object):
    def __init__(self, fname):
        self.fname = fname
        self.csv_data = pd.read_csv(self.fname, sep=',')
        self.curr_ind = 0
        self.last_loc = self.csv_data.last_valid_index()

    def next_item(self, num=1, columns='all'):
        # if columns == 'all':
        if self.curr_ind + num < self.last_loc:
            out_dat = self.csv_data.loc[self.curr_ind:self.curr_ind + num]
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
    actual_data_one_hot = np.zeros(batch_s, one_hot_vec_len)
    actual_data_one_hot[:, actual_data] = 1
    out2 = -np.multiply(actual_data_one_hot, out1)
    return np.sum(out2)


class node(object):
    def __init__(self, gradient_back=0, curr_val=0):
        self.curr_val = curr_val
        self.gradient_back = gradient_back
        return

    def reset_grad(self):
        self.gradient_back = 0

    def reset_curr_val(self):
        self.curr_val = 0

    def reset_node(self):
        self.gradient_back = 0
        self.curr_val = 0


class layer(object):
    def __init__(self, num_nodes=5):
        self.node_list = list()
        for _ in range(num_nodes):
            new_node = node()
            self.node_list.append(new_node)

    def reset_layer_grad(self):
        for n in self.node_list:
            n.reset_grad()

    def reset_layer_curr_val(self):
        for n in self.node_list:
            n.reset_curr_val()
        return

    def reset_layer(self):
        for n in self.node_list:
            n.reset_node()
        return

    def weights_l2(self):
        norm = 0
        for n in self.node_list:
            norm += n.curr_val**2
        return np.sqrt(norm)


class lin_layer(layer):
    def __init__(self, num_nodes=5):
        super(lin_layer, self).__init__()


class neural_network(object):
    def __init__(self):
        self.list_layers = list()
        fc1 = lin_layer()
        self.list_layers.append(fc1)
        fc2 = lin_layer()
        self.list_layers.append(fc2)
        return

    def weights_l2(self):
        norm = 0
        for l in self.list_layers:
            norm += l.weights_l2() ** 2
        return np.sqrt(norm)


class model(object):
    def __init__(self, nn, train_data, test_data, loss_fn='cel2', optimizer='sgd'):
        self.nn = nn
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_data = train_data
        self.test_data = test_data

    def train_net(self, num_epoch=15, batch_size=4):
        num_train_data = self.train_data.last_loc + 1
        for epoch in num_epoch:
            for _ in np.arange(0, num_train_data, batch_size):
                train_data_new = self.train_data.next_item(num=batch_size)
                x_train_data = train_data_new[:, :-1]
                y_train_data = train_data_new[:, -1]
                y_pred_data = self.nn.get_output(x_train_data)
                curr_loss = self.loss(y_train_data, y_pred_data)
                self.nn.update_weights()

    def loss(self, actual_data, pred_data):
        ce_loss = cross_entropy(actual_data, pred_data)
        l2_loss = self.nn.weights_l2()
        return ce_loss + l2_loss


if __name__ == '__main__':
    train_fname = '../data/train.csv'
    test_fname = '../data/test.csv'
    train_data = data_loader(train_fname)
    test_data = data_loader(test_fname)
    simple_net = neural_network()
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
