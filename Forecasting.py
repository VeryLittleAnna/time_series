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
import csv
from math import ceil

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



def learn(dataset_X, dataset_y, window_size=None):
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
    model.add(LSTM(units = 50, input_shape = (window_size, N_features)))
    model.add(Dense(N_features))
    model.compile(optimizer='adam', loss='mae') #, metrics=[MeanAbsoluteError()]) #loss=mae ?
    # plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    # model.summary()
    batch_size = 128
    # history = model.fit(Datagen(dataset, batch_size=batch_size, window_size=window_size), \
    #         epochs=10, batch_size=batch_size, shuffle=False, verbose=1) # validation_data=(test_X, test_y)
    history = model.fit(dataset_X, dataset_y, epochs=20, batch_size=batch_size, shuffle=False, verbose=1) # validation_data=(test_X, test_y)
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
            result_windows.append(part_windows)
            result_answers.append(part_answers)
        return result_windows, result_answers
    
    windows = np.zeros((data.shape[0], window_size, data.shape[1])) #(N, SL, Q)
    for i in range(window_size): 
        windows[:, i, :] = np.roll(data, -i, axis=0)
    answers = data[window_size:, ...]
    return windows[window_size:, ...], answers #:data.shape[0] - window_size


class MyStandardScaler:
    def __init__(self):
        pass
    def fit(self, data):
        if isinstance(data, np.ndarray):
            data = [data]
        self.data = []
        for i in range(len(data)):
            self.data.append(np.diff(data[i], axis=0))
        cur_sum = np.sum(np.row_stack([np.sum(part, axis=0) for part in self.data]), axis=0)
        cur_cnt = sum([part.shape[0] for part in self.data])
        self.mean = cur_sum / cur_cnt
        self.std = np.sqrt(np.sum(np.row_stack([np.sum(np.square(part - self.mean), axis=0) / cur_cnt for part in self.data]), axis=0))


    def transform(self, data):
        eps=1e-8
        result_data = []
        for i in range(len(data)):
            result_data.append(np.diff(data[i], axis=0))
            result_data[i] = (result_data[i] - self.mean) / np.maximum(self.std, eps)
        return result_data



def split_to_train_test(dataset_X, dataset_y, percent_of_test=20):
    """
    Args:
        dataset_X (list of ndarrays) or (ndarray): ..., (M, W, Q)
        ...
    Returns:
        dataset_X (ndarray): (M_sum_train, W, Q), ...
    """
    if isinstance(dataset_X, list):
        W, Q = dataset_X[0].shape[1:3]
        result_train_X, result_train_y, result_test_X, result_test_y = np.zeros((1, W, Q)), \
                np.zeros((1, Q)), np.zeros((1, W, Q)), np.zeros((1, Q))
        for X, y in zip(dataset_X, dataset_y):
            train_X, train_y, test_X, test_y = split_to_train_test(X, y, percent_of_test=percent_of_test)
            result_train_X = np.concatenate([result_train_X, train_X])
            result_train_y = np.concatenate([result_train_y, train_y])
            result_test_X = np.concatenate([result_test_X, test_X])
            result_test_y = np.concatenate([result_test_y, test_y])
        # print(f"In the end split: {result_test_X.shape=}, {result_test_y.shape=}")
        return result_train_X[1:, ...], result_train_y[1:, ...], result_test_X[1:, ...], result_test_y[1:, ...]
    n_split = round((1 - percent_of_test/ 100) * dataset_X.shape[0])
    train_X = dataset_X[:n_split, ...]
    train_y = dataset_y[:n_split, ...]
    test_X = dataset_X[n_split:, ...]
    test_y = dataset_y[n_split:, ...]
    # print(f"In split: {train_X.shape=}, {train_y.shape=}, {test_X.shape=}, {test_y.shape=}")
    return train_X, train_y, test_X, test_y


#no....
def create_train_test_in_clusters(dataset_X, dataset_y, cluster_labels, N_clusters):
    clusters_data_X = [dataset_X[cluster_labels == k] for k in range(N_clusters)]
    clusters_data_y = [dataset_y[cluster_labels == k] for k in range(N_clusters)]
    clusters_train_X, clusters_train_y = [], []
    clusters_test_X, clusters_test_y = [], []
    for i in range(N_clusters):
        train_X, train_y, test_X, test_y = split_to_train_test(clusters_data_X[i], clusters_data_y[i])
        clusters_train_X.append(train_X)
        clusters_train_y.append(train_y)
        clusters_test_X.append(test_X)
        clusters_test_y.append(test_y)
    return clusters_train_X, clusters_train_y, clusters_test_X, clusters_test_y

