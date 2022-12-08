import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

from sklearn.cluster import KMeans
from collections import defaultdict
from sklearn.metrics import davies_bouldin_score


N_clusters=5

def KMeans_for_windows(dataset, W=5, N_clusters=8, max_iter=200):
    if isinstance(dataset, pd.DataFrame):
        dataset = dataset.to_numpy()
    if len(dataset.shape) < 3:
        windows = np.array([dataset[i:i+W].flatten() for i in range(dataset.shape[0] - W)])
    else:
        windows = np.array([dataset[i].flatten() for i in range(dataset.shape[0])])
    if N_clusters == 1:
        return np.zeros((windows.shape[0]), dtype=np.int8)
    model = KMeans(n_clusters=N_clusters, max_iter=max_iter, init='random') #n_jobs ??
    res = model.fit_predict(windows)
    return model

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

def split_to_clusters(dataset, labels, W=1):
    print(f"In split_to_clusters: {len(dataset)=}, {(len(labels) + W)=}")
    assert(len(dataset) == (len(labels) + W))
    N_clusters = np.max(np.array(labels)) + 1
    dataset_result = [[] for i in range(N_clusters)]
    i, start = 0, 0
    while i < len(labels):
        j = i
        while j < len(labels) and labels[j] == labels[i]:
            j +=1
        dataset_result[labels[i]].append(dataset[i:j + W - 1, ...])
        i = j
    return dataset_result


def calc_clusters_metrics(dataset, labels, centroids=None):
    """
    dataset - windows
    Returns:
        Davies-Bouldin Index
    """
    if len(dataset.shape) == 3:
        dataset = np.array([dataset[i].flatten() for i in range(dataset.shape[0])])
    answer = {}
    answer["DB"] = davies_bouldin_score(dataset, labels)
    # dist_in = np.array([labels[i] == k for i in range(len(labels)) for k in range N_clusters])
    return answer

def create_segments(data, segment_size=1):
    """
    Args:
        data (ndarray): (N, Q)
    return segments (N // segment_size, segment, Q)
    """
    if isinstance(data, list):
        return [create_segments(part, segment_size=segment_size) for part in data]
    N, Q = data.shape
    data = data[:N - N % segment_size, ...]
    segments = data.reshape(N // segment_size, segment_size, Q)
    return segments
