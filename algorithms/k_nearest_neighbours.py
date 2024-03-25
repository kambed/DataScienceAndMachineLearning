from enum import Enum
from sklearn.neighbors import KNeighborsClassifier

from algorithms.classification import Classification

"""
    KNearestNeighbours class
"""


class KNearestNeighbours(Classification):

    """
        :param n_neighbors: Number of neighbours
        :param metric: Metric used to calculate distance
        :param learn_data: Data used to learn
        :param test_data: Data used to test
    """

    def __init__(self, learn_data, test_data, n_neighbors, metric):
        self.n_neighbors = n_neighbors
        self.metric = metric
        model = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric.value)
        super().__init__(learn_data, test_data, model)


"""
    Metric enum
"""


class Metric(Enum):
    EUCLIDEAN = 'euclidean'
    MANHATTAN = 'manhattan'
