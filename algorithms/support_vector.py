from algorithms.classification import Classification
from enum import Enum
from sklearn.svm import SVC


class SupportVectorKernelType(Enum):
    LINEAR = 'linear'
    POLY = 'poly'
    RBF = 'rbf'
    SIGMOID = 'sigmoid'


class SupportVectorGammaType(Enum):
    AUTO = 'auto'
    SCALE = 'scale'


class SupportVector(Classification):
    def __init__(self, learn_data, test_data, kernel, c, gamma):
        self.kernel = kernel.value
        self.C = c
        self.gamma = gamma.value
        model = SVC(kernel=self.kernel, C=self.C, gamma=self.gamma)
        super().__init__(learn_data, test_data, model)


