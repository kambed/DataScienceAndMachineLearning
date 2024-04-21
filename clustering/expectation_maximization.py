from enum import Enum

import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist, squareform, pdist
from sklearn.metrics import silhouette_score

from clustering.clustering_algorithm import ClusteringAlgorithm
from sklearn.mixture import GaussianMixture


def compute_inertia(labels, distance_matrix):
    inertia = 0
    for cluster_label in np.unique(labels):
        cluster_distance_matrix = distance_matrix[np.ix_(labels == cluster_label, labels == cluster_label)]
        inertia += np.sum(cluster_distance_matrix) / 2
    return inertia

"""
    ExpectationMaximization class
"""


class ExpectationMaximization(ClusteringAlgorithm):
    """
        :param data: data to fit
        :param n_components: Number of components
        :param covariance_type: Type of covariance
    """

    def __init__(self, data, n_components, covariance_type):
        self.n_components = n_components
        self.covariance_type = covariance_type
        model = GaussianMixture(n_components=n_components, covariance_type=covariance_type.value)
        super().__init__(data, model)

    """
        :param n_clusters_range: Number of clusters range
    """
    def create_elbow_inertia_chart(self, n_clusters_range):
        elbow_data = self.data.drop('Class', axis=1)
        mean_dist = []
        distance_matrix = squareform(pdist(elbow_data))
        number_of_clusters = range(1, n_clusters_range + 1)
        for n_clusters in number_of_clusters:
            gaussian_mixture = GaussianMixture(n_components=n_clusters, covariance_type=self.covariance_type.value)
            gaussian_mixture.fit(elbow_data)
            labels = gaussian_mixture.predict(elbow_data)
            inertia = compute_inertia(labels, distance_matrix)
            mean_dist.append(inertia)
        plt.plot(number_of_clusters, mean_dist, marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('Interia')
        plt.title('Elbow Curve')
        plt.show()


class CovarianceType(Enum):
    FULL = 'full'
    TIED = 'tied'
    DIAG = 'diag'
    SPHERICAL = 'spherical'
