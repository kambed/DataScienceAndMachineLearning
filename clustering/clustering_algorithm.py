from abc import abstractmethod, ABC
from enum import Enum


class ClusteringAlgorithm(ABC):

    def __init__(self, data, model):
        self.data = data
        self.model = model

    def fit(self):
        data_to_fit = self.data
        X = data_to_fit.drop('Class', axis=1)
        y = data_to_fit['Class']
        labels = self.model.fit_predict(X)
        return labels, X, y

    @abstractmethod
    def create_elbow_inertia_chart(self, n_clusters_range):
        pass


class Clustering(Enum):
    K_MEANS = 'K_MEANS'
    DBSCAN = 'DBSCAN'
    AGGLOMERATIVE = 'AGGLOMERATIVE'
    EXPECTATION_MAXIMIZATION = 'EXPECTATION_MAXIMIZATION'
