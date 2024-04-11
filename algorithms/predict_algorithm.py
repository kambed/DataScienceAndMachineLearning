from enum import Enum


class PredictAlgorithm(Enum):
    NAIVE_BAYES = 'NAIVE_BAYES'
    KNN = 'KNN'
    SUPPORT_VECTOR = 'SUPPORT_VECTOR'
    DECISION_TREE = 'DECISION_TREE'
    RANDOM_FOREST = 'RANDOM_FOREST'
