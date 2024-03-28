from algorithms.classification import Classification
from enum import Enum
from sklearn.svm import SVC


class SupportVectorType(Enum):
    LINEAR = 'linear'
    POLY = 'poly'
    RBF = 'rbf'
    SIGMOID = 'sigmoid'


class SupportVector(Classification):
    def __init__(self, learn_data, test_data, kernel, c, gamma):
        self.kernel = kernel.value
        self.C = c
        self.gamma = gamma
        model = SVC(kernel=self.kernel, C=self.C, gamma=self.gamma)
        super().__init__(learn_data, test_data, model)


