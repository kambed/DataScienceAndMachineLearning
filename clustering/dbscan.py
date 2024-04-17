import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from clustering.clustering_algorithm import ClusteringAlgorithm

"""
    Dbscan class
"""


class Dbscan(ClusteringAlgorithm):
    """
            :param data: data to fit
            :param eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other
            :param min_samples: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point
        """

    def __init__(self, data, eps, min_samples):
        self.eps = eps
        self.min_samples = min_samples
        model = DBSCAN(eps=eps, min_samples=min_samples)
        super().__init__(data, model)

    def create_elbow_inertia_chart(self, n_clusters_range):
        elbow_data = self.data.drop('Class', axis=1)
        neighbors = NearestNeighbors(n_neighbors=n_clusters_range)
        neighbors_fit = neighbors.fit(elbow_data)
        distances, indices = neighbors_fit.kneighbors(elbow_data)
        distances = np.sort(distances, axis=0)
        distances = distances[:, 1]
        plt.plot(distances)
        plt.xlim(7750, 8500)
        plt.ylim(0, 500)
        plt.show()
