from enum import Enum


class ClusteringAlgorithm:

    def __init__(self, data, model):
        self.data = data
        self.model = model

    def fit(self):
        data_to_fit = self.data
        X = data_to_fit.drop('Class', axis=1)
        y = data_to_fit['Class']
        labels = self.model.fit_predict(X)
        return labels, X, y

    def fit(self, data):
        data_to_fit = data
        X = data_to_fit.drop('Class', axis=1)
        y = data_to_fit['Class']
        labels = self.model.fit_predict(X)
        return labels, X, y


class Clustering(Enum):
    K_MEANS = 'K_MEANS'
    DBSCAN = 'DBSCAN'
    AGGLOMERATIVE = 'AGGLOMERATIVE'
    EXPECTATION_MAXIMIZATION = 'EXPECTATION_MAXIMIZATION'
