from sklearn.cluster import DBSCAN

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
        # Implement chart creation here
        pass
