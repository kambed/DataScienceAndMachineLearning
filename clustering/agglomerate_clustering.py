from enum import Enum

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

    def __init__(self, data, n_clusters, linkage):
        self.n_clusters = n_clusters
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage.value)
        super().__init__(data, model)


class Linkage(Enum):
    WARD = 'ward'
    COMPLETE = 'complete'
    AVERAGE = 'average'
    SINGLE = 'single'
