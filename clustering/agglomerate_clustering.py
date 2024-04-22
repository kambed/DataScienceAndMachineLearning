from enum import Enum

import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import squareform, pdist
from sklearn.cluster import AgglomerativeClustering

from clustering.clustering_algorithm import ClusteringAlgorithm

"""
    AgglomerateClustering class
"""


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
        elbow_data = self.data.drop('Class', axis=1)
        mean_dist = []
        distance_matrix = squareform(pdist(elbow_data))
        number_of_clusters = range(1, n_clusters_range + 1)
        for n_clusters in number_of_clusters:
            alg = AgglomerativeClustering(n_clusters=n_clusters, linkage=self.linkage.value)
            alg.fit(elbow_data)
            inertia = compute_inertia(alg.labels_, distance_matrix)
            mean_dist.append(inertia)
        plt.plot(number_of_clusters, mean_dist, marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('Interia')
        plt.title('Elbow Curve')
        plt.show()


class Linkage(Enum):
    WARD = 'ward'
    COMPLETE = 'complete'
    AVERAGE = 'average'
    SINGLE = 'single'
