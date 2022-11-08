import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

from sklearn.cluster import KMeans
from collections import defaultdict


N_clusters=5

def KMeans_for_windows(dataset, W=5, N_clusters=8, max_iter=200):
    if isinstance(dataset, pd.DataFrame):
        dataset = dataset.to_numpy()
    if len(dataset.shape) < 3:
        windows = np.array([dataset[i:i+W].flatten() for i in range(dataset.shape[0] - W)])
    else:
        windows = np.array([dataset[i].flatten() for i in range(dataset.shape[0])])
    
    model = KMeans(n_clusters=N_clusters, max_iter=max_iter, init='random') #n_jobs ??
    res = model.fit_predict(windows)
    return res

def flatten_from_interceting_windows(dataset, labels, N_clusters=None, W=1):
    """
    Args:
        dataset (ndarray): (M, W, N_features)
        labels (ndarray): (M, ) - labels of clusters
        W (int)
    Return:
        dataset (list): for each cluster (list)- of continuos parts (ndarray): (N_i_samples, N_features)
    """
    assert(len(dataset) == len(labels))
    dataset_result = [[] for i in range(N_clusters)]
    i = 0
    while i < len(dataset):
        tmp = dataset[i]
        cluster_num = labels[i]
        # dataset[cluster_num]
        i += 1
        while i < len(dataset) and labels[i] == cluster_num:
            tmp = np.row_stack((tmp, dataset[i][-1]))
            i += 1
        if len(tmp.shape) < 3:
            tmp = tmp[None, ...]
        # print(f"{tmp.shape=}, {cluster_num=}, {len(dataset_result)=}, {np.max(labels)=}")
        dataset_result[cluster_num].append(tmp.reshape(tmp.shape[0] * tmp.shape[1], tmp.shape[2]))
    return dataset_result
