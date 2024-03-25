from enum import Enum

from sklearn.neighbors import KNeighborsClassifier

from alghoritms.data_model import DataModel

"""
    KNearestNeighbours class
"""


class KNearestNeighbours(DataModel):
    """
        :param n_neighbors: Number of neighbours
        :param metric: Metric used to calculate distance
        :param learn_data: Data used to learn
        :param test_data: Data used to test
    """

    def __init__(self, n_neighbors, metric, learn_data, test_data):
        super().__init__(learn_data, test_data)
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric.value)

    """
        Train the model
    """

    def train(self):
        x = self.learn_data.iloc[:, :-1].values
        y = self.learn_data.iloc[:, -1].values
        self.model.fit(x, y)

    """
        Predict the model
        :param x: Data to predict
    """

    def predict(self, x):
        return self.model.predict(x)

    """
        Predict the test data
    """

    def predict_test_data(self):
        x = self.test_data.iloc[:, :-1].values
        y = self.test_data.iloc[:, -1].values
        return self.predict(x), y


"""
    Metric enum
"""


class Metric(Enum):
    EUCLIDEAN = 'euclidean'
    MANHATTAN = 'manhattan'
