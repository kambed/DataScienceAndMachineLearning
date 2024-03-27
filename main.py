from algorithms.algorithm import Algorithm
from algorithms.decision_tree import DecisionTree
from algorithms.k_nearest_neighbours import KNearestNeighbours, Metric
from algorithms.naive_bayes import NaiveBayes, NaiveBayesType
from algorithms.random_forest import RandomForest
from helper.argument_helper import ArgumentHelper
from helper.confusion_matrix_helper import ConfusionMatrixHelper
from helper.data_csv_helper import DataCsvHelper
from helper.data_split_helper import DataSplitHelper
from helper.roc_curve_helper import RocCurveHelper
from helper.learning_curve_helper import LearningCurveHelper


def create_classification_algorithm(learn_data, test_data):
    algorithm = ArgumentHelper.get_enum_argument("algorithm", Algorithm)
    if algorithm == Algorithm.NAIVE_BAYES:
        naive_bayes_type = ArgumentHelper.get_enum_argument("naive_bayes_type", NaiveBayesType)
        return NaiveBayes(learn_data, test_data, naive_bayes_type)
    elif algorithm == Algorithm.DECISION_TREE:
        max_depth = ArgumentHelper.get_int_argument("max_depth")
        min_samples_leaf = ArgumentHelper.get_int_argument("min_samples_leaf")
        return DecisionTree(learn_data, test_data, max_depth, min_samples_leaf)
    elif algorithm == Algorithm.RANDOM_FOREST:
        n_estimators = ArgumentHelper.get_int_argument("n_estimators")
        max_features = ArgumentHelper.get_int_argument("max_features")
        return RandomForest(learn_data, test_data, n_estimators, max_features)
    elif algorithm == Algorithm.KNN:
        n_neighbors = ArgumentHelper.get_int_argument("n_neighbors")
        metric = ArgumentHelper.get_enum_argument("metric", Metric)
        return KNearestNeighbours(learn_data=learn_data, test_data=test_data, n_neighbors=n_neighbors, metric=metric)
    else:
        raise NotImplementedError(f"Algorithm {algorithm} not found!")


if __name__ == '__main__':
    labels = ['DERMASON', 'SIRA', 'SEKER']
    test_size = 0.3

    train_size = 1 - test_size
    train_step = 0.05

    splitter = DataSplitHelper(DataCsvHelper.read_csv(labels=labels), test_size)
    learn_data, test_data = splitter.split()

    classification = create_classification_algorithm(learn_data, test_data)

    lch = LearningCurveHelper(classification, train_size, train_step)
    predicted, expected = lch.train_and_predict()

    print("\n")
    cmh = ConfusionMatrixHelper(expected, predicted, labels)
    cmh.display_confusion_matrix()
    cmh.display_classification_report()

    roc = RocCurveHelper(actual=expected, predicted=predicted)
    roc.plot_roc_curve()

    lch.display_learning_curve()
