from enum import Enum

import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import squareform, pdist
from sklearn.cluster import AgglomerativeClustering

from clustering.clustering_algorithm import ClusteringAlgorithm

"""
    AgglomerateClustering class
"""


def plot_elbow_curve(inertia_values):
    plt.plot(range(1, len(inertia_values) + 1), inertia_values, marker='o')
    plt.title('Elbow Curve')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()


def compute_inertia(labels, distance_matrix):
    inertia = 0
    for cluster_label in np.unique(labels):
        cluster_distance_matrix = distance_matrix[np.ix_(labels == cluster_label, labels == cluster_label)]
        inertia += np.sum(cluster_distance_matrix) / 2
    return inertia


class AgglomerateClustering(ClusteringAlgorithm):
    """
        :param data: data to fit
        :param n_clusters: Number of clusters
    """

    def __init__(self, data, n_clusters, linkage):
        self.n_clusters = n_clusters
        self.linkage = linkage
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage.value)
        super().__init__(data, model)

    def create_elbow_inertia_chart(self, n_clusters_range):
        inertia_values = []
        data = self.data
        X = data.drop('Class', axis=1)
        distance_matrix = squareform(pdist(X))
        for n in range(1, n_clusters_range + 1):
            alg = AgglomerativeClustering(n_clusters=n, linkage=self.linkage.value).fit(X)
            inertia = compute_inertia(alg.labels_, distance_matrix)
            inertia_values.append(inertia)

        plot_elbow_curve(inertia_values)


class Linkage(Enum):
    WARD = 'ward'
    COMPLETE = 'complete'
    AVERAGE = 'average'
    SINGLE = 'single'
