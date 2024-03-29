# from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import LabelEncoder
import keras
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.metrics import MeanAbsoluteError, MeanSquaredError
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_absolute_error as mae
import tensorflow as tf
# from tensorflow.keras.losses import MeanAbsoluteError
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from sktime.forecasting.model_selection import SlidingWindowSplitter

import csv
from math import ceil
import json
import csv
import pickle
import Clustering
from sklearn.preprocessing import StandardScaler
config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 16} ) 
sess = tf.compat.v1.Session(config=config) 
tf.compat.v1.keras.backend.set_session(sess)
from numpy.lib.stride_tricks import sliding_window_view

eps=1e-15
MAX_ITERS_KMEANS = 200
MAX_EPOCHS = 100

TRAIN_MODE, VALID_MODE, TEST_MODE, GLOBAL_TEST_MODE = [0, 1, 2, 3]


def scaled_model(model, scaler, diff=True, verbose=False):
    mean, scale = scaler.mean_, scaler.scale_
    verbose and print(f"Base model input: {model.input_shape}")
    _, lag, var = model.input_shape
    if diff:
        input_ = keras.layers.Input(shape=(lag+1, var))
        verbose and print(f"New input: {input_.shape}")
        output_ = input_[:, 1:] - input_[:, :-1] # дифференцируем вход
    else:
        input_ = keras.layers.Input(shape=(lag, var))
        output_ = input_
    output_ = (output_ - mean) / scale # scale.transform
    output_ = model(output_)
    verbose and print(f"Base model output: {model.output_shape}")
#     if len(output_.shape) < 3:
#         output_ = tf.expand_dims(output_, -1)
#         verbose and print(f"New output: {output_.shape}")
    output_ = output_ * scale + mean # scale.inverse_transform
    verbose and print(f"{output_.shape=}")
    if diff: # возвращаем выход(ы) от производных
        last = input_[:, -1] # последнее значение лагов
#         output_ = tf.concat([last, output_], axis=1) # last, diff1, diff2 ...
#         output_ = tf.math.cumsum(output_, axis=1)[:, 1:] # last + diff1, last + diff1 + diff2 ...
        output_ += input_[..., -1, :]
        verbose and print(f"{output_.shape=}")

    return keras.models.Model(inputs=input_, outputs=output_)


def apply_forecasting_training(dataset, clusters_labels, W=10, model=None):
    scaler = StandardScaler()
    scaled_dataset = np.diff(dataset, axis=0)
    N, Q = dataset.shape
    clusters_labels = clusters_labels[1:]
    scaled_dataset = scaler.fit_transform(scaled_dataset)
    full_results = np.zeros((N - W - 1, 2 * Q + 2))
    N_clusters = len(np.unique(clusters_labels))
    cur_models = [0] * N_clusters
    metrics = [0] * N_clusters
    clusters_X, clusters_labels = Clustering.split_to_clusters(scaled_dataset, clusters_labels, W=W+1)
    if model is None:
        model = Sequential(
            LSTM(units = 50, activation="tanh", recurrent_activation="sigmoid", input_shape = (W, Q)),
            Dense(Q)
        )
        my_early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", verbose=1, mode="min", patience=2, restore_best_weights=True)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mae', run_eagerly=True)
        
    clusters_X, clusters_labels = Clustering.split_to_clusters(scaled_dataset, clusters_labels, W=W+1)
    for cluster_num in range(N_clusters):
        clusters_mask = (clusters_labels == cluster_num)
        X, y = split_X_y(clusters_X[cluster_num])
        train_X, train_y, valid_X, valid_y, test_X, test_y, mask = split_to_train_test(X, y, part_of_test=0.2, part_of_valid=0.2)
        model.fit(train_X, train_y, epochs=EPOCHS, batch_size=batch_size, shuffle=True, verbose=1, callbacks=[my_early_stopping])
        cur_models[cluster_num] = scaled_model(model, scaler)
#         test_predict = model.predict(test_X)
        full_results[clusters_mask] = calc_results(cluster_num, train_X, train_y, valid_X, valid_y, test_X, test_y, mask, model, scaler, dataset, W)
    metrics[cluster_num] = calc_metrics_for_full_results(full_results[clusters_mask])
    return cur_models, full_results, metrics                                               
                             

def my_mase(y_true, y_pred, multioutput='raw_values'):
    numer = my_mae(y_true, y_pred, multioutput='raw_values')
    if y_true.shape[0] == 1:
        denom = 1
    else:
        denom = my_mae(y_true[1:], y_true[:-1], multioutput='raw_values')
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



def calc_metrics(y_true, y_pred):
    # metrics = ["mae", "mape", "mase"]
    cur_mae = mae(y_true, y_pred, multioutput='raw_values')
    cur_mape = mape(y_true, y_pred, multioutput='raw_values')
    cur_mase = my_mase(y_true, y_pred, multioutput='raw_values')
    return {"mae":cur_mae, "mape":cur_mape, "mase":cur_mase}

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
#     tf.debugging.set_log_device_placement(True)
    N_features = dataset_X.shape[-1]
#     plt.hist(dataset_y[:, 9])
#     plt.title(dataset_X.shape)
#     plt.show()
    
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
        history = model.fit(dataset_X, dataset_y, epochs=MAX_EPOCHS, batch_size=batch_size, shuffle=True, verbose=1, validation_data=(valid_X, valid_y), callbacks=[my_early_stopping])
    else:
        history = model.fit(dataset_X, dataset_y, epochs=20, batch_size=batch_size, shuffle=True, verbose=1) # validation_data=(test_X, test_y)
    return model


def calc_results(cluster_num, train_X, train_y, valid_X, valid_y, test_X, test_y, mask_train_test, model, scaler, dataset, W):
    print(f"In calc_results: {train_X.shape[0]}, {valid_X.shape[0]}, {test_X.shape[0]}, sum = {train_X.shape[0] + valid_X.shape[0] + test_X.shape[0]}")
    N = train_y.shape[0] + valid_y.shape[0] + test_y.shape[0]
    X = np.concatenate((train_X, valid_X, test_X), axis=0)
    y = np.concatenate((train_y, valid_y, test_y), axis=0)
    Q = train_X.shape[-1]
    results = np.zeros((y.shape[0], 2 * Q + 2), dtype=np.float64)
#     results = {}
    predicted = model(X)
    assert(y.shape[0] == predicted.shape[0])
    predicted_original = scaler.inverse_transform(predicted)
    y_true = scaler.inverse_transform(y)
#     predicted_original = scaler.add_first_elements(predicted_original)
    results[:, :Q] = y_true
    results[:, Q:2 * Q] = predicted_original
#     results[:, 2*Q:3*Q] = y
#     results[:, 3*Q:4*Q] = predicted
    results[:, 2 * Q] = cluster_num
    results[:, 2 * Q + 1] = mask_train_test
    return results
#     return , predicted_original + dataset[W:]
#     return results 

def calc_metrics_for_full_results(results):
    Q = results.shape[-1] // 2 - 1
    y_true = results[:, :Q]
    y_pred = results[:, Q:2*Q]
    cur_mae = mae(y_true, y_pred, multioutput='raw_values')
    cur_mase = my_mase(y_true, y_pred, multioutput='raw_values')
    return {"mae":cur_mae, "mase":cur_mase}


def apply_forecasting_training_simple_version(dataset, clusters_labels, W=10):
    scaler = StandardScaler()
    scaled_dataset = np.diff(dataset, axis=0)
    N, Q = dataset.shape
    clusters_labels = clusters_labels[1:]
    scaled_dataset = scaler.fit_transform(scaled_dataset)
    full_results = np.zeros((N - W - 1, 2 * Q + 2))
    N_clusters = len(np.unique(clusters_labels))
    cur_models = [0] * N_clusters
    metrics = [0] * N_clusters
    clusters_X, clusters_labels = Clustering.split_to_clusters(scaled_dataset, clusters_labels, W=W+1)
    for cluster_num in range(N_clusters):
        clusters_mask = (clusters_labels == cluster_num)
        if clusters_X[cluster_num] is None or not clusters_X[cluster_num].any():
           continue 
        X, y = split_X_y(clusters_X[cluster_num])
        train_X, train_y, valid_X, valid_y, test_X, test_y, mask = split_to_train_test(X, y, part_of_test=0.2, part_of_valid=0.2)
        model = learn(train_X, train_y, valid_X=valid_X, valid_y=valid_y)
        cur_models[cluster_num] = scaled_model(model, scaler)
#         test_predict = model.predict(test_X)
        full_results[clusters_mask] = calc_results(cluster_num, train_X, train_y, valid_X, valid_y, test_X, test_y, mask, model, scaler, dataset, W)
        metrics[cluster_num] = calc_metrics_for_full_results(full_results[clusters_mask]) #?
    full_results[:, :Q] += dataset[W:-1]
    full_results[:, Q:2*Q] += dataset[W:-1]
    return cur_models, full_results, metrics
#     return model, full_results, metrics



def predict_through_clusters(dataset, clusters_model, prediction_models, W=10):
    """"
    Test the whole process: classify the input window and forecast next step for it
    Args:
        dataset (ndarray) : (N_samples, N_features)
        clusters_model (kmeans model) : 
        prediction_models (list) : list of forecasting models
        window_size_clustering (int)
        window_size_forecasting (int)
    Returns:
        y_pred (ndarray) : (N_samples - window_size_forecasting + 1, N_features)
    """
    N, Q = dataset.shape
    N_clusters = len(prediction_models)
    clusters_labels = clusters_model.predict(dataset)
    clusters_X, clusters_labels = Clustering.split_to_clusters(dataset, clusters_labels, W=W+1, N_clusters=N_clusters)
#     clusters_labels = clusters_labels[1:]
    full_results = np.zeros((N - W, Q + 2)) #predicted_values, cluster_num
    for cluster_num in range(N_clusters):
        mask = (clusters_labels == cluster_num)
        if np.sum(mask) == 0:
            continue
        assert(mask.sum() == clusters_X[cluster_num].shape[0])
        cur_windows = clusters_X[cluster_num]
        predicted = prediction_models[cluster_num].predict(cur_windows)
        full_results[mask, :Q] = predicted
        full_results[mask,  Q] = cluster_num
    full_results[:, Q + 1] = GLOBAL_TEST_MODE
    return full_results




def apply_forecasting_training_new_version(dataset, clusters_labels, learning_function=learn, W=10, logs=None):
    if logs is None:
        logs = open("logs", "w")
    clusters_X, clusters_labels = Clustering.split_to_clusters(dataset, clusters_labels, W=W + 2)
    full_results = np.zeros((clusters_labels.shape[0], 2 * dataset.shape[-1] + 2)) # real_Q, Q, cluster_num, mask
    print("clustes sizes: ", [clusters_X[i].shape for i in range(len(clusters_X))], file=logs)
    N_clusters = len(np.unique(clusters_labels))
    metrics = [0] * N_clusters
    cur_models = [0] * N_clusters
    scalers = [0] * N_clusters
    for cluster_num in range(N_clusters):
        sc = MyStandardScaler()
#         sc.fit(clusters_X[cluster_num])
#         prepared_data = sc.transform(clusters_X[cluster_num])
        prepared_data = sc.fit_transform(clusters_X[cluster_num])
        prepared_X, prepared_y = split_X_y(prepared_data)
#         sc.save_first_elements(clusters_X[cluster_num], y=1)
        scalers[cluster_num] = sc
        train_X, train_y, valid_X, valid_y, test_X, test_y, mask = split_to_train_test(prepared_X, prepared_y, part_of_test=0.2, part_of_valid=0.2)
        print(f"Before prediction: {train_X.shape=}, {train_y.shape=}, {test_X.shape=}, {test_y.shape=}", file=logs)
        try:
            assert(len(test_X.shape) == 3 and test_X.shape[0] > 0)
            assert(len(valid_X.shape) == 3 and valid_X.shape[0] > 0)
            assert(len(train_X.shape) == 3 and train_X.shape[0] > 0)
        except AssertionError:
            print(f"FAIL - {test_X.shape=}, {valid_X.shape=}, {train_X.shape=}", file=logs)
            tmp = np.array([np.inf] * dataset.shape[-1])
            metrics[cluster_num] = {"mae": tmp, "mape": tmp, "mase": tmp}
            continue
        model = learning_function(train_X, train_y, valid_X=valid_X, valid_y=valid_y)
        predicted = model.predict(test_X) #(N, Q)
        predicted_original = sc.inverse_transform(predicted)
#         predicted_original = sc.add_first_elements(predicted_original)
        y_true = clusters_X[cluster_num][-test_X.shape[0]:, -1, :]
        metrics[cluster_num] = calc_metrics(y_true, predicted_original) #?
        cur_models[cluster_num] = model
        clusters_mask = (clusters_labels == cluster_num)
        full_results[clusters_mask] = calc_results(cluster_num, train_X, train_y, valid_X, valid_y, test_X, test_y, mask, model, sc)
    model = {'models':cur_models[:], "scalers":scalers}
    clusters_sizes = np.array([np.sum(clusters_labels == i) for i in range(N_clusters)])
    weighted_mase = np.average(np.row_stack([metrics[i]["mase"] for i in range(N_clusters)]), axis=0, weights=clusters_sizes)
    weighted_mae =  np.average(np.row_stack([metrics[i]["mae"] for i in range(N_clusters)]), axis=0, weights=clusters_sizes)
#     mae_on_max_cluster = metrics[np.argmax(clusters_sizes)]['mae']
    return model, full_results, metrics, weighted_mase, weighted_mae 
        
def apply_forecasting_training_last(dataset, clusters_labels, W=10):
    logs = open("logs", "w")
    clusters_X, clusters_labels = Clustering.split_to_clusters(dataset, clusters_labels, W=W + 2)
    full_results = np.zeros((clusters_labels.shape[0], 2 * dataset.shape[-1] + 2)) # real_Q, Q, cluster_num, mask
    print("clustes sizes: ", [clusters_X[i].shape for i in range(len(clusters_X))], file=logs)
    N_clusters = len(np.unique(clusters_labels))
    metrics = [0] * N_clusters
    cur_models = [0] * N_clusters
    scalers = [0] * N_clusters
    for cluster_num in range(N_clusters):
        sc = MyStandardScaler()
        sc.fit(clusters_X[cluster_num])
        print(f"{sc.mean=}, {sc.std=}", file=logs)
        prepared_data = sc.transform(clusters_X[cluster_num])
        prepared_X, prepared_y = split_X_y(prepared_data)
        sc.save_first_elements(clusters_X[cluster_num], y=1)
        scalers[cluster_num] = sc
        train_X, train_y, valid_X, valid_y, test_X, test_y, mask = split_to_train_test(prepared_X, prepared_y, part_of_test=0.2, part_of_valid=0.2)
        print(f"Before prediction: {train_X.shape=}, {train_y.shape=}, {test_X.shape=}, {test_y.shape=}", file=logs)
        try:
            assert(len(test_X.shape) == 3 and test_X.shape[0] > 0)
            assert(len(valid_X.shape) == 3 and valid_X.shape[0] > 0)
            assert(len(train_X.shape) == 3 and train_X.shape[0] > 0)
        except AssertionError:
            print(f"FAIL - {test_X.shape=}, {valid_X.shape=}, {train_X.shape=}", file=logs)
            tmp = np.array([np.inf] * dataset.shape[-1])
            metrics[cluster_num] = {"mae": tmp, "mape": tmp, "mase": tmp}
            continue
        model = learn(train_X, train_y, valid_X=valid_X, valid_y=valid_y)
        predicted = model.predict(test_X) #(N, Q)
        predicted_original = sc.inverse_transform(predicted)
        predicted_original = sc.add_first_elements(predicted_original)
        y_true = clusters_X[cluster_num][-test_X.shape[0]:, -1, :]
        metrics[cluster_num] = calc_metrics(y_true, predicted_original) #?
        cur_models[cluster_num] = model
        clusters_mask = (clusters_labels == cluster_num)
        full_results[clusters_mask] = calc_results(cluster_num, train_X, train_y, valid_X, valid_y, test_X, test_y, mask, model, sc)
    model = {'models':cur_models[:], "scalers":scalers}
    clusters_sizes = np.array([np.sum(clusters_labels == i) for i in range(N_clusters)])
    weighted_mase = np.average(np.row_stack([metrics[i]["mase"] for i in range(N_clusters)]), axis=0, weights=clusters_sizes)
    weighted_mae =  np.average(np.row_stack([metrics[i]["mae"] for i in range(N_clusters)]), axis=0, weights=clusters_sizes)
#     mae_on_max_cluster = metrics[np.argmax(clusters_sizes)]['mae']
    return model, full_results, metrics, weighted_mase, weighted_mae 
    
def try_parameters(parameters, dataset):
    """
    Args:
        parameters (dict): dict with keys from {N_clusters, window_size_for_clustering, dif}
    """
    best_model_mae, best_clusters_model = None, None
    best_model, answer = {}, {}
    best_full_results = None
    window_size_forecasting = 10
    cluster_metrics = Clustering.ClustersMetrics()
    for cluster_algorithm in parameters["cluster_algs"]:
#         print(cluster_algorithm.info())
        clusters_model = cluster_algorithm.fit_predict(dataset)
        print(type(clusters_model))
        clusters_labels = clusters_model.labels_
        N_clusters = len(np.unique(clusters_labels))
        print(f"{cluster_algorithm}: {N_clusters=}")
        plt.clf()
        plt.hist(clusters_labels)
        plt.savefig(f"Hist_clusters_sizes_{cluster_algorithm}_W={cluster_algorithm.W}_N={N_clusters}_03-07_1.png")
        cluster_metrics.calc_DB(dataset, clusters_labels, W=cluster_algorithm.W, N=N_clusters)

        clusters_labels = np.pad(clusters_labels, (dataset.shape[0] - clusters_labels.shape[0], 0), mode='constant', constant_values=(clusters_labels[0])) 
        print(f"{len(np.unique(clusters_labels))=}")
        clusters_X, clusters_labels = Clustering.split_to_clusters(dataset, clusters_labels, W=window_size_forecasting + 2)
        full_results = np.zeros((clusters_labels.shape[0], 2 * dataset.shape[-1] + 2)) # real_Q, Q, cluster_num, mask
        # print(f"{clusters_X[0].shape=}, {clusters_y[0].shape=}, {clusters_X[1].shape=}, {clusters_y[1].shape}")
        # clusters_labels = clusters_labels[-sum([clusters_X[]])]
        print("meow: ", [clusters_X[i].shape for i in range(len(clusters_X))], file=logs)
        metrics = [0] * N_clusters
        cur_models = [0] * N_clusters
        scalers = [0] * N_clusters
        for cluster_num in range(N_clusters):
            # clusters_mask = (clusters_labels == cluster_num)
            print(f"All: {np.sum(clusters_labels == cluster_num)}", file=logs)
            sc = MyStandardScaler()
            sc.fit(clusters_X[cluster_num])
            print(f"{sc.mean=}, {sc.std=}", file=logs)
            prepared_data = sc.transform(clusters_X[cluster_num])
            prepared_X, prepared_y = split_X_y(prepared_data)
            sc.save_first_elements(clusters_X[cluster_num], y=1)
            scalers[cluster_num] = sc
            train_X, train_y, valid_X, valid_y, test_X, test_y, mask = split_to_train_test(prepared_X, prepared_y, part_of_test=0.2, part_of_valid=0.2)
            print(f"Before prediction: {train_X.shape=}, {train_y.shape=}, {test_X.shape=}, {test_y.shape=}", file=logs)
            try:
                assert(len(test_X.shape) == 3 and test_X.shape[0] > 0)
                assert(len(valid_X.shape) == 3 and valid_X.shape[0] > 0)
                assert(len(train_X.shape) == 3 and train_X.shape[0] > 0)
            except AssertionError:
                print(f"FAIL - {test_X.shape=}, {valid_X.shape=}, {train_X.shape=}", file=logs)
                tmp = np.array([1e12] * dataset.shape[-1])
                metrics[cluster_num] = {"mae": tmp, "mape": tmp, "mase": tmp}
                continue
            model = learn(train_X, train_y, valid_X=valid_X, valid_y=valid_y)
            predicted = model.predict(test_X) #(N, Q)
            print(f"{train_X[0]=}\n{train_y[0]=}\n{model(train_X[None, 0])=}\n", file=logs)
            predicted_original = sc.inverse_transform(predicted)
            predicted_original = sc.add_first_elements(predicted_original)
            y_true = clusters_X[cluster_num][-test_X.shape[0]:, -1, :]
            metrics[cluster_num] = calc_metrics(y_true, predicted_original)
            print(f"{metrics[cluster_num]=}", file=logs)
            cur_models[cluster_num] = model
            clusters_mask = (clusters_labels == cluster_num)
            full_results[clusters_mask] = calc_results(cluster_num, train_X, train_y, valid_X, valid_y, test_X, test_y, mask, model, sc)                
        clusters_sizes = np.array([np.sum(clusters_labels == i) for i in range(N_clusters)])
        weighted_mase = np.average(np.row_stack([metrics[i]["mase"] for i in range(N_clusters)]), axis=0, weights=clusters_sizes)
        weighted_mape = np.average(np.row_stack([metrics[i]["mape"] for i in range(N_clusters)]), axis=0, weights=clusters_sizes)
        weighted_mae =  np.average(np.row_stack([metrics[i]["mae"] for i in range(N_clusters)]), axis=0, weights=clusters_sizes)
        if best_model_mae is None or np.mean(weighted_mae) < best_model_mae:
            best_model_mae = np.mean(weighted_mae)
            best_model = {'models':cur_models[:], "scalers":scalers, 'clusters_model':clusters_model}
            best_full_results = np.copy(full_results)
        answer[(cluster_algorithm.W, N_clusters)] = ["str cluster_metrics clusters_model metrics clusters_sizes weighted_mase weighted_mape", cluster_metrics, \
                clusters_model, metrics, clusters_sizes, weighted_mase, weighted_mape]
    output = open('output_metrics_03-08_1.pickle', 'wb')
    pickle.dump(answer, output)
    output.close()
    #cluster_metrics.dump()
    return best_model, best_model_mae, best_full_results


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

def split_X_y(dataset):
    X = dataset[:, :-1, :]
    y = dataset[:, -1, :]
    return X, y


class MyStandardScaler(StandardScaler):
    def __init__(self, **kwargs):
#         super().__init__(**kwargs)
        self.scaler = StandardScaler(**kwargs)
    
    def _reshape(self, data, shape_len=2):
        self.origin_shape_len = len(data.shape)
        print(f"{self.origin_shape_len=} vs {shape_len=}")
        if len(data.shape) > shape_len:
            self.Q = data.shape[-1]
            self.n = data.shape[0]
            data = data.reshape(-1, data.shape[-1])
            print(data.shape)
        elif len(data.shape) < shape_len:
            data = data.reshape(self.n, -1, self.Q)
        return data

    
    def fit(self, data): #add save_first_elements
        data = np.diff(data, axis=1)
        data = self._reshape(data)
        print(f"In fit {data.shape=}")
        self.scaler.fit(data)
        return self
        
    def transform(self, data, y=1):
        self.first_elements = data[:, -y-1, :]
        result_data = np.diff(data, axis=1)
        data = self._reshape(data)
        result_data = self.scaler.transform(data)
        result_data = self._reshape(result_data, shape_len=self.origin_shape_len)
        return result_data
    
    def fit_transform(self, data):
        return self.fit(data).transform(data)
    
    def inverse_transform(self, data):
        data = self._reshape(data)
        result_data = self.scaler.inverse_transform(data)
        result_data = self._reshape(result_data, shape_len=self.origin_shape_len)
#         result_data += self.first_elements[:data.shape[0]]
        return result_data
#         result_data = self.std * data + self.mean
#         return data + self.first_elements[:data.shape[0]] #костыли

        

class MyStandardScaler_last:
    def __init__(self):
        pass
    def fit(self, data):
        if len(data.shape) == 2:
            data = data.reshape(data.shape[0], 1, data.shape[1])
        # self.first_elements = data[:, 0, :]
        data = np.diff(data, axis=1)
        self.mean = np.mean(data, axis=(0, 1))
        self.std = np.std(data, axis=(0, 1))

    def save_first_elements(self, data, y=0):
        if len(data.shape) == 2:
            self.first_elements = np.zeros(data.shape[-1])
        self.first_elements = data[:, -y-1, :]


    def transform(self, data, dif=True):
        if dif:
            result_data = np.diff(data, axis=1)
        else:
            result_data = data
        result_data = (data - self.mean) / np.maximum(self.std, eps)
        return result_data 

    def inverse_transform(self, data):
        """
        Args:
            data - list of ndarrays: (N_i_samples, N_features)
        Returns:
            result_data - list of ndarrays (N_i_samples, N_features)
        """
        # if len(data.shape) == 2:
        #     data = data.reshape(data.shape[0], 1, data.shape[2])
        result_data = self.std * data + self.mean
        # result_data = np.cumsum(np.concatenate([np.zeros((data.shape[0], 1, data.shape[2])), data], axis=1), axis=1)
        # result_data = result_data[1:, ...] #delete first zeroes
        return result_data

    def add_first_elements(self, data):
        return data + self.first_elements[:data.shape[0]] #костыли




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
        #not updated
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
    print(f"In split_to_train_test: {dataset_X.shape=}, {dataset_y.shape=}")
    n_split = round((1 - part_of_test) * dataset_X.shape[0])
    test_X = dataset_X[n_split:, ...]
    test_y = dataset_y[n_split:, ...]
    test_ind = np.arange(n_split, dataset_y.shape[0])
    mask = np.zeros(dataset_y.shape[0])
    mask[:n_split] = TRAIN_MODE #train
    mask[n_split:] = TEST_MODE #test
    if part_of_valid is None:
        train_X = dataset_X[:n_split, ...]
        train_y = dataset_y[:n_split, ...]
        return train_X, train_y, test_X, test_y, test_ind, mask
    valid_n_split = round((1 - part_of_test - part_of_valid) * dataset_X.shape[0])
    valid_X = dataset_X[valid_n_split:n_split, ...]
    valid_y = dataset_y[valid_n_split:n_split, ...]
    train_X = dataset_X[:valid_n_split, ...]
    train_y = dataset_y[:valid_n_split, ...]
    mask[valid_n_split:n_split] = VALID_MODE #valid
    return train_X, train_y, valid_X, valid_y, test_X, test_y, mask


