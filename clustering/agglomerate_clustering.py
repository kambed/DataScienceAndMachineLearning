from sklearn.cluster import AgglomerativeClustering

from clustering.clustering_algorithm import ClusteringAlgorithm

"""
    AgglomerateClustering class
"""


class AgglomerateClustering(ClusteringAlgorithm):

    """
        :param data: data to fit
        :param n_clusters: Number of clusters
    """
    def __init__(self, data, n_clusters):
        self.n_clusters = n_clusters
        model = AgglomerativeClustering(n_clusters=n_clusters)
        super().__init__(data, model)
