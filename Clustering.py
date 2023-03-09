import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

from sklearn.cluster import KMeans
from collections import defaultdict
from sklearn.metrics import davies_bouldin_score

from numpy.lib.stride_tricks import sliding_window_view

from sklearn.cluster import MeanShift

from sklearn.cluster import AgglomerativeClustering


N_clusters=5


class Clusterization:
    def __init__(self, W=1):
        self.W = W
        print(f"Clusterization __init__: {W=}, {type(self.W)}")
    def prepare(self, dataset):
        if isinstance(dataset, pd.DataFrame):
            dataset = dataset.to_numpy()
        if len(dataset.shape) == 2:
            dataset_windows = sliding_window_view(dataset, (self.W, dataset.shape[-1])) #(N, 1, W, Q)
            new_shape = dataset_windows.shape
            dataset_windows = dataset_windows.reshape(new_shape[0], new_shape[2] * new_shape[3]) #(N, W, Q)
        dataset_windows = dataset_windows.reshape(dataset_windows.shape[0], -1)
#         dataset_windows = np.array([dataset_windows[i].flatten() for i in range(dataset.shape[0])])
        return dataset_windows
    def str(self):
        return "Clusterization algorithm"
    def info(self):
        attrs = [x for x in dir(self) if not callable(x)]
        return str(self) + " " + "; ".join([str(a) + "=" + str(getattr(self, a)) for a in attrs])

class Kmeans_for_windows(Clusterization):
    def __init__(self, N_clusters=7, max_iter=200, **kwargs):
        super().__init__(**kwargs)
        self.N_clusters = N_clusters
        self.max_iter = max_iter

    def fit_predict(self, dataset):
        windows = self.prepare(dataset)
        if self.N_clusters == 1:
            return np.zeros((windows.shape[0]), dtype=np.int8)
        model = KMeans(n_clusters=self.N_clusters, max_iter=self.max_iter, init='random') #n_jobs ??

        labels = model.fit_predict(windows)
        self.model = model
        return model
    def __str__(self):
        return 'Kmeans_for_windows'

class MeanShift_for_windows(Clusterization):
    def __init__(self, max_iter=200, **kwargs):
        super().__init__(**kwargs)
        self.max_iter = max_iter

    def fit_predict(self, dataset):
        windows = self.prepare(dataset)
        model = MeanShift(max_iter=self.max_iter, n_jobs=-1) 
        labels = model.fit_predict(windows)
        self.model = model
        return model
    def __str_(self):
        return 'MeanShift_for_windows'
    
class AgglomerativeClustering_for_windows(Clusterization):
    def __init__(self, N_clusters=4, **kwargs):
        super().__init__(**kwargs)
        self.N_clusters = N_clusters
    def fit_predict(self, dataset):
        windows = self.prepare(dataset)
        model = AgglomerativeClustering(n_clusters=self.N_clusters, affinity ="euclidean", linkage="ward") 
        labels = model.fit_predict(windows)
        self.model = model
        return model
    def __str_(self):
        return 'AgglomerativeClustering_for_windows'

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
    

    
def apply_clustering(dataset, clustering_algorithm):
    clusters_model = clustering_algorithm.fit_predict(dataset)
    print(type(clusters_model))
    clusters_labels = clusters_model.labels_
    N_clusters = len(np.unique(clusters_labels))
    print(f"{clustering_algorithm}: {N_clusters=}")
    plt.clf()
    plt.hist(clusters_labels)
    plt.savefig(f"Hist_clusters_sizes_{clustering_algorithm}_W={clustering_algorithm.W}_N={N_clusters}_03-07_1.png")
#     cluster_metrics.calc_DB(dataset, clusters_labels, W=cluster_algorithm.W, N=N_clusters)
    clusters_labels = np.pad(clusters_labels, (dataset.shape[0] - clusters_labels.shape[0], 0), mode='constant', constant_values=(clusters_labels[0])) 
    #if Agglomerative -> classifier
    return clusters_labels, clusters_model
        


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


def split_to_clusters(dataset, labels, W): #W=11
    N_clusters = len(np.unique(labels))
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
        answers_DB = {}
    def calc_DB(self, dataset, labels, W=1, N=1):
        """
        dataset (N_samples, Q)
        Returns:
            Davies-Bouldin Index
        """
        if len(dataset.shape) == 2:
            dataset_windows = sliding_window_view(dataset, (W, dataset.shape[-1])) #(N, 1, W, Q)
            new_shape = dataset_windows.shape
            dataset_windows = dataset_windows.reshape(new_shape[0], new_shape[2] * new_shape[3]) #(N, W, Q)
        ClustersMetrics.answers_DB[(W, N)] = davies_bouldin_score(dataset_windows, labels)

    def dump(self):
        print(ClustersMetrics.answers_DB.keys())
        with open("clusters_metrics_03-08_1.csv", "w") as csvfile:
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
