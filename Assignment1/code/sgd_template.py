#!/bin/python3

import numpy as np
import pickle
import pandas as pd
import pdb
import re
# from sgd_solver import model
"""
NOTE: All functions mentioned below MUST be implemented
      All functions must be reproducible, i.e., repeated function calls with the
      same parameters must result in the same output. Look into numpy RandomState
      to achieve this.
"""


def get_feature_matrix(file_path):
    """
    file path: path to  the file assumed to be in the same format as
               either train.csv or test_features.csv in the Kaggle competition


    Return: A 2-D numpy array of size n x m where n is the number of examples in
            the file and m your feature vector size

    NOTE: Preserve the order of examples in the file
    """
    data_reader = pd.read_csv(file_path, sep=',')
    is_test = 'test' in file_path
    # +1 is for bias
    # For now not using date and time
    # Now using only month and hour
    pat = re.compile(r'(\d*)-(\d*)-(\d*)\s(\d*):(\d*):(\d*)')
    date_time = np.array(pat.findall('\n'.join(data_reader['date'])), dtype='int')
    if is_test:
        phi_matrix = np.zeros((data_reader.last_valid_index() + 1, 3 +
                               len(data_reader.loc[0][2:]) + 1))
        for ind in range(data_reader.last_valid_index() + 1):
            d_info = data_reader.loc[ind]
            tmp1 = np.append(d_info[2:], 1)
            tmp2 = date_time[ind, 1:4]
            phi_matrix[ind, :] = np.append(tmp2, tmp1)

    else:
        # pdb.set_trace()
        phi_matrix = np.zeros((data_reader.last_valid_index() + 1, 3 +
                               len(data_reader.loc[0][2:-1]) + 1))
        # pdb.set_trace()
        for ind in range(data_reader.last_valid_index() + 1):
            d_info = data_reader.loc[ind]
            tmp1 = np.append(d_info[2:-1], 1)
            tmp2 = date_time[ind, 1:4]
            phi_matrix[ind, :] = np.append(tmp2, tmp1)

    return phi_matrix


def get_output(file_path):
    """
    file_path: path to a file in the same format as in the Kaggle competition

    Return: an n x 1 numpy array where n is the number of examples in the file.
            The array must contain the Output column values of the file

    NOTE: Preserve the order of examples in the file
    """
    data_reader = pd.read_csv(file_path, sep=',')
    # Assuming that file is train.csv
    output_vec = data_reader['Output']
    return output_vec.values.reshape(output_vec.shape[0], 1)


def get_weight_vector(feature_matrix, output, lambda_reg, p):
    """
    feature_matrix: an n x m 2-D numpy array where n is the number of samples
                    and m the feature size.
    output: an n x 1 numpy array reprsenting the outputs for the n samples
    lambda_reg: regularization parameter
    p: p-norm for the regularized regression

    Return: an m x 1 numpy array weight vector obtained through stochastic gradient descent
            using the provided function parameters such that the matrix product
            of the feature_matrix matrix with this vector will give you the
            n x 1 regression outputs

    NOTE: While testing this function we will use feature_matrices not obtained
          from the get_feature_matrix() function but you can assume that all elements
          of this matrix will be of type float
    """
    # sgd_model = model(feature_matrix, output, lambda_reg, p)
    # sgd_model.sgd()



def get_my_best_weight_vector():
    """
    Return: your best m x 1 numpy array weight vector used to predict the output for the
            kaggle competition.

            The matrix product of the feature_matrix, obtained from get_feature_matrix()
            call with file as test_features.csv, with this weight vector should
            result in you best prediction for the test dataset.

    NOTE: For your final submission you are expected to provide an output.csv containing
          your predictions for the Kaggle test set and the weight vector returned here
          must help us to EXACTLY reproduce that output.csv

          We will also be using this weight to evaluate on a separate hold out test set

          We expect this function to return fast. So you are encouraged to return a pickeled
          file after all your experiments with various values of p and lambda_reg.
    """
    # directly pickle the best weight vector


# if __name__ == '__main__':
#     fname = '../data/train.csv'
#     # fname = '../data/test_features.csv'
#     feature_matrix = get_feature_matrix(fname)
#     out_vec = get_output(fname)
#     lambda_reg = 1
#     p = 2
#     weight_vector = get_weight_vector(feature_matrix, out_vec, lambda_reg, p)
#     # Regression with p-norm regularization
#     # loss = ||y - y'||^2 + lambda * ||w||^p
#     # Can try to use k-fold cross validation
#     # to check if the sgd code is indeed working
#     # do a sanity check for the case p=2
#     lambda_reg = 1
#     p = 2
#     # sgd_model = model(lamb, p, all_data)
#     sgd_model = model(feature_matrix, out_vec, lambda_reg, p)
#     sgd_model.sgd()
