# from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import LabelEncoder
import keras
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.metrics import MeanAbsoluteError, MeanSquaredError
import tensorflow as tf
# from tensorflow.keras.losses import MeanAbsoluteError
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sktime.forecasting.model_selection import SlidingWindowSplitter

import csv
from math import ceil

eps=1e-10



def my_mase(y_true, y_pred, multioutput='raw_values'):
    numer = my_mae(y_true, y_pred, multioutput='raw_values')
    if y_true.shape[0] == 1:
        denom = 1
    else:
        denom = mae_naive(y_true, multioutput='raw_values')
    cur_mase = numer / np.maximum(denom, eps)
    if multioutput == 'uniform_average':
        return np.mean(cur_mase)
    return cur_mase


def my_mae(y_true, y_pred, multioutput='raw_values'):
    cur_mae = np.mean(np.abs(y_pred - y_true), axis=0)
    if multioutput == 'uniform_average':
        return np.mean(cur_mae)
    elif multioutput == 'raw_values':
        return cur_mae
    else:
        assert(False)


def mae_naive(data, multioutput='raw_values'):
    """
    Args:
        data : list of ndarray (Ni_samples, N_features)
    """
    if isinstance(data, np.ndarray):
        data = [data]
    cnt = sum([part.shape[0] - 1 for part in data]) * data[0].shape[1]
    cur_mae = np.sum(np.row_stack([np.array(np.sum(np.abs(part[1:, ...] - part[:-1, ...]), axis=0)) for part in data]), axis=0)
    if multioutput == 'raw_values':
        return cur_mae / cnt
    elif multioutput == 'uniform_average':
        return np.sum(cur_mae) / cnt / cur_mae.shape[0]
    else:
        assert(False)

# class Datagen(keras.utils.Sequence):
#     def __init__(self, X, batch_size=1, window_size=1):
#         #X: list of (N_i_samples, N_features)
#         self.X = []
#         self.y = []
#         for part in X:
#             if len(part) < window_size:
#                 continue
#             part_X, part_y = create_windows(part, window_size=window_size)
#             self.X = np.concatenate(self.X, part_X, axis=0)
#             self.y = np.concatenate(self.y, part_y, axis=0)
#         self.batch_size = batch_size
#         assert(len(self.X) == len(self.y))
#         assert(len(self.X) > 0)
    
#     def __getitem__(self, idx): 
#         start, end = idx * self.batch_size, (idx + 1) * self.batch_size
#         return self.X[start:end], self.y[start, end]
    
#     def __len__(self):
#         return np.ceil(len(self.X) / self.batch_size)



def learn(dataset_X, dataset_y, window_size=None, valid_X=None, valid_y=None):
    """
    Learn LSTM model on dataset
    Args:
        dataset_X (ndarray): (M, window_size, N_features)
        dataset_y (ndarray): (M, N_features)
    Returns:
        model, history
    # Args:
    #     dataset (list): list of continuos datasets_i: (N_i_samples, N_features)
    #     window_size (int): size of window for LSTM layer
    """
    N_features = dataset_X.shape[-1]
    if window_size is None:
        window_size = dataset_X.shape[1]
    model = Sequential()
    #batch_size, (time_steps, units)
    model.add(LSTM(units = 50, activation="tanh", recurrent_activation="sigmoid", input_shape = (window_size, N_features)))
    model.add(Dense(N_features))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mae', run_eagerly=True) #, metrics=[MeanAbsoluteError()]) #loss=mae ?
    # plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    # model.summary()
    batch_size = 64
    # history = model.fit(Datagen(dataset, batch_size=batch_size, window_size=window_size), \
    #         epochs=10, batch_size=batch_size, shuffle=False, verbose=1) # validation_data=(test_X, test_y)
    if valid_X is not None:
        my_early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", verbose=1, mode="min", patience=2, restore_best_weights=True)
        history = model.fit(dataset_X, dataset_y, epochs=40, batch_size=batch_size, shuffle=True, verbose=1, validation_data=(valid_X, valid_y), callbacks=[my_early_stopping])
    else:
        history = model.fit(dataset_X, dataset_y, epochs=20, batch_size=batch_size, shuffle=True, verbose=1) # validation_data=(test_X, test_y)
    return model, history


def create_windows(data, window_size=1):
    """
    Args:
        data (ndarray): (N, Q)
    return windows (N - windows_size, window_size, Q), answer (N - windows_size, Q)
    """
    if isinstance(data, list):
        result_windows, result_answers = [], []
        for part in data:
            part_windows, part_answers = create_windows(part, window_size=window_size)
#             assert(part_windows.shape[0] > 0 and len(part_windows.shape) == 3)
            if part_windows.shape[0] == 0 or len(part_windows.shape) == 2:
                part_windows = part_answers = None
            result_windows.append(part_windows)
            result_answers.append(part_answers)
        return result_windows, result_answers
    
    windows = np.zeros((data.shape[0], window_size, data.shape[1])) #(N, SL, Q)
    for i in range(window_size): 
        windows[:, i, :] = np.roll(data, -i, axis=0)
    answers = data[window_size:, ...]
    return windows[:-window_size, ...], answers #:data.shape[0] - window_size


class MyStandardScaler:
    def __init__(self, dif=True):
        self.dif_flag = dif

    def fit(self, data):
        if isinstance(data, np.ndarray):
            data = [data]
        # self.original_data = data
        self.data = []
        for i in range(len(data)):
            if self.dif_flag:
                self.data.append(np.diff(data[i], axis=0))
            else:
                self.data.append(data[i])
        cur_sum = np.sum(np.row_stack([np.sum(part, axis=0) for part in self.data]), axis=0)
        cur_cnt = sum([part.shape[0] for part in self.data])
        self.mean = cur_sum / cur_cnt
        self.std = np.sqrt(np.sum(np.row_stack([np.sum(np.square(part - self.mean), axis=0) / cur_cnt for part in self.data]), axis=0))


    def transform(self, data):
        if isinstance(data, np.ndarray):
            data = [data]
        result_data = []
        for i in range(len(data)):
            if self.dif_flag:
                result_data.append(np.diff(data[i], axis=0))
            else:
                result_data.append(data[i])
            result_data[i] = (result_data[i] - self.mean) / np.maximum(self.std, eps)
        return result_data 

    def inverse_transform(self, data):
        """
        Args:
            data - list of ndarrays: (N_i_samples, N_features)
        Returns:
            result_data - list of ndarrays (N_i_samples, N_features)
        """
        if isinstance(data, np.ndarray):
            data = [data]
        result_data = []
        for i in range(len(data)):
            result_data.append(data[i] * self.std + self.mean)
            # print(f"{result_data[i].shape}")
            if self.dif_flag:
                result_data[i] = np.concatenate([np.zeros((1, result_data[i].shape[1])), result_data[i]], axis=0).cumsum(axis=0) #add first element
                result_data[i] = result_data[i][1:, ...] #delete first zeroes
        return result_data

    # def add_first_element(self, data, window_size=1, part_of_test=0.2):
    #     """
    #     original_data - list of ndarrays (N_samples, N_features) - test_y
    #     data - 
    #     """
    #     if isinstance(data, np.ndarray):
    #         data = [data]
    #     answer = []
    #     for i in range(len(data)):
    #         n = self.original_data[i].shape[0]
    #         print(int((n - 1 - window_size) * (1 - part_of_test)))
    #         answer.append(self.original_data[i][int((n - 1 - window_size) * (1 - part_of_test)), ...])
    #     return answer

    def add_first_element(self, data, ind, window_size=1):
        """
        """
        data += self.original_data[ind]
        return data




def split_to_train_test(dataset_X, dataset_y, part_of_test=0.2, part_of_valid=None):
    """
    Args:
        dataset_X (list of ndarrays) or (ndarray): ..., (M, W, Q)
        ...
    Returns:
        dataset_X (ndarray): (M_sum_train, W, Q), ...
        indices_of_test_starts
    """
    if isinstance(dataset_X, list):
        for i in range(len(dataset_X)):
            if dataset_X[i] is not None:
                W, Q = dataset_X[i].shape[1:3]
                break
        else:
            return None
        result_train_X, result_train_y, result_test_X, result_test_y, result_valid_X, result_valid_y, test_ind = np.zeros((1, W, Q)), \
                np.zeros((1, Q)), np.zeros((1, W, Q)), np.zeros((1, Q)), np.zeros((1, W, Q)), np.zeros((1, Q)), np.zeros((1))
        cur_pos = 0
        for X, y in zip(dataset_X, dataset_y):
            if X is None:
                continue
            if part_of_valid is None:
                train_X, train_y, test_X, test_y, ind = split_to_train_test(X, y, part_of_test=part_of_test)
            else:
                train_X, train_y, valid_X, valid_y, test_X, test_y, ind = split_to_train_test(X, y, part_of_test=part_of_test, part_of_valid=part_of_valid)
#                 print(f"    IN SPLIT: {train_X.shape=}, {valid_X.shape=}, {test_X.shape=}. {X.shape=}, {y.shape=}") 
            test_ind = np.concatenate([test_ind, ind])
            result_train_X = np.concatenate([result_train_X, train_X])
            result_train_y = np.concatenate([result_train_y, train_y])
            result_test_X = np.concatenate([result_test_X, test_X])
            result_test_y = np.concatenate([result_test_y, test_y])
            cur_pos += X.shape[0] + 1
            if part_of_valid is not None:
                result_valid_X = np.concatenate([result_valid_X, valid_X])
                result_valid_y = np.concatenate([result_valid_y, valid_y])
                return result_train_X[1:, ...], result_train_y[1:, ...], result_valid_X[1:, ...], result_valid_y[1:, ...], result_test_X[1:, ...], result_test_y[1:, ...], test_ind[1:]
            else:
                return result_train_X[1:, ...], result_train_y[1:, ...], result_test_X[1:, ...], result_test_y[1:, ...], test_ind[1:]
    n_split = round((1 - part_of_test) * dataset_X.shape[0])
    test_X = dataset_X[n_split:, ...]
    test_y = dataset_y[n_split:, ...]
    test_ind = np.arange(n_split, dataset_y.shape[0])
    if part_of_valid is None:
        train_X = dataset_X[:n_split, ...]
        train_y = dataset_y[:n_split, ...]
        return train_X, train_y, test_X, test_y, test_ind
    valid_n_split = round((1 - part_of_test - part_of_valid) * dataset_X.shape[0])
    valid_X = dataset_X[valid_n_split:n_split, ...]
    valid_y = dataset_y[valid_n_split:n_split, ...]
    train_X = dataset_X[:valid_n_split, ...]
    train_y = dataset_y[:valid_n_split, ...]
    return train_X, train_y, valid_X, valid_y, test_X, test_y, [n_split]

