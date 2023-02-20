import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

from sklearn.cluster import KMeans
from collections import defaultdict
from sklearn.metrics import davies_bouldin_score

from numpy.lib.stride_tricks import sliding_window_view

from sklearn.cluster import MeanShift


N_clusters=5


class Clusterization:
    def __init__(self, W=5, max_iter=200):
        self.W = W
        self.max_iter = max_iter
    def prepare(self, dataset):
        if isinstance(dataset, pd.DataFrame):
            dataset = dataset.to_numpy()
        if len(dataset.shape) < 3:
            windows = np.array([dataset[i:i+self.self.W].flatten() for i in range(dataset.shape[0] - self.self.W)])
        else:
            windows = np.array([dataset[i].flatten() for i in range(dataset.shape[0])])
        return windows

class Kmeans_for_windows(Clusterization):
    def __init__(self, N_clusters=7, **kwargs):
        super().__init__(kwargs)
        self.N_clusters = N_clusters
    def fit_predict(self, dataset):
        windows = self.prepare(dataset)
        if N_clusters == 1:
            return np.zeros((windows.shape[0]), dtype=np.int8)
        model = KMeans(n_clusters=N_clusters, max_iter=self.max_iter, init='random') #n_jobs ??
        self.model = model.fit_predict(windows)
        return self.model
    def __str__(self):
        return 'Kmeans_for_windows'

class MeanShift_for_windows(Clusterization):
    def __init__(self, **kwargs):
        super().__init__(kwargs)
    def fit_predict(self, dataset):
        windows = self.prepare(dataset)
        model = MeanShift(max_iter=self.max_iter, n_jobs=-1) #n_jobs ??
        self.model = model.fit_predict(windows)
        return self.model
    def __str_(self):
        return 'MeanShift_for_windows'

# def KMeans_for_windows(dataset, W=5, N_clusters=8, max_iter=200):
#     if isinstance(dataset, pd.DataFrame):
#         dataset = dataset.to_numpy()
#     if len(dataset.shape) < 3:
#         windows = np.array([dataset[i:i+W].flatten() for i in range(dataset.shape[0] - W)])
#     else:
#         windows = np.array([dataset[i].flatten() for i in range(dataset.shape[0])])
#     if N_clusters == 1:
#         return np.zeros((windows.shape[0]), dtype=np.int8)
#     model = KMeans(n_clusters=N_clusters, max_iter=max_iter, init='random') #n_jobs ??
#     res = model.fit_predict(windows)
#     return model
    



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

def split_to_clusters1(dataset, labels, W=1):
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


class DatasetClusters:
    def __init__(self, dataset, labels):
        self.dataset = dataset
        self.label = labels
    def __iter__(self):
        self.cur_num_cluster = 0
        return self
    
    def __next__(self):
        mask = (self.labels == self.cur_num_cluster)
        
        self.cur_num_cluster += 1


def split_to_clusters(dataset, labels, W=11):
    N_clusters = np.max(labels) + 1
    print(f"{N_clusters=}")
    labels = labels[W-2:-1] #класс относится к последнему в X
    clusters_datasets, clusters_y = [], []
    dataset_windows = sliding_window_view(dataset, (W, dataset.shape[-1])) #(N, 1, W, Q)
    print(f"{dataset_windows.shape=}, {labels.shape=}")
    assert(dataset_windows.shape[0] == labels.shape[0])
    for cluster_num in range(N_clusters):
        mask = (labels == cluster_num)
        clusters_datasets.append(dataset_windows[mask, 0, :, :])
        # clusters_y.append(dataset_windows[mask, 0, W, :])
    return clusters_datasets, labels #, clusters_y



class ClustersMetrics:
    answers_DB = {} #(W, N)

    def __init__(self):
        pass
    def calc_DB(self, dataset, labels, W=1, N=1):
        """
        dataset - windows
        Returns:
            Davies-Bouldin Index
        """
        if len(dataset.shape) == 3:
            data = np.array([dataset[i].flatten() for i in range(dataset.shape[0])])
        else:
            data = dataset[:]
        ClustersMetrics.answers_DB[(W, N)] = davies_bouldin_score(data, labels)

    def dump(self):
        with open("clusters_metrics.csv", "w") as csvfile:
            Ns = sorted(list(set([k[1] for k in ClustersMetrics.answers_DB.keys()])))
            Ws = sorted(list(set([k[0] for k in ClustersMetrics.answers_DB.keys()])))
            writer = csv.writer(csvfile)
            writer.writerow([' ']+Ns)
            for W in Ws:
                writer.writerow([W] + [ClustersMetrics.answers_DB[(W, N)] for N in Ns])
    


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
