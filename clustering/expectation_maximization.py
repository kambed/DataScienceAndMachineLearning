from enum import Enum

from sklearn.cluster import EXPECTATION_MAXIMIZATION

from clustering.clustering_algorithm import ClusteringAlgorithm

"""
    ExpectationMaximization class
"""


class ExpectationMaximization(ClusteringAlgorithm):
    """
        :param data: data to fit
        :param n_components: Number of components
        :param covariance_type: Type of covariance
    """

    def create_elbow_inertia_chart(self, n_clusters_range):
        pass

    def __init__(self, data, n_components, covariance_type):
        self.n_components = n_components
        self.covariance_type = covariance_type
        model = EXPECTATION_MAXIMIZATION(n_components=n_components, covariance_type=covariance_type)
        super().__init__(data, model)


class CovarianceType(Enum):
    FULL = 'full'
    TIED = 'tied'
    DIAG = 'diag'
    SPHERE = 'sphere'
