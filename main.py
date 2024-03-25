from alghoritms.alghoritm import Algorithm
from alghoritms.decision_tree import DecisionTree
from alghoritms.k_nearest_neighbours import KNearestNeighbours, Metric
from alghoritms.naive_bayes import NaiveBayes
from alghoritms.random_forest import RandomForest
from helper.argument_helepr import ArgumentHelper
from helper.confusion_matrix_helper import ConfusionMatrixHelper
from helper.data_csv_helper import DataCsvHelper
from helper.data_split_helper import DataSplitHelper
from helper.roc_curve_helper import RocCurveHelper


def create_classification_algorithm(algorithm_name, learn_data, test_data):
    if algorithm_name == Algorithm.NAIVE_BAYES:
        return NaiveBayes(learn_data, test_data)
    elif algorithm_name == Algorithm.DECISION_TREE:
        max_depth = ArgumentHelper.get_int_argument("max_depth")
        min_samples_leaf = ArgumentHelper.get_int_argument("min_samples_leaf")
        return DecisionTree(learn_data, test_data, max_depth, min_samples_leaf)
    elif algorithm_name == Algorithm.RANDOM_FOREST:
        n_estimators = ArgumentHelper.get_int_argument("n_estimators")
        max_features = ArgumentHelper.get_int_argument("max_features")
        return RandomForest(learn_data, test_data, n_estimators, max_features)
    elif algorithm_name == Algorithm.KNN:
        n_neighbors = ArgumentHelper.get_int_argument("n_neighbors")
        metric = ArgumentHelper.get_enum_argument("metric", Metric)
        return KNearestNeighbours(learn_data=learn_data, test_data=test_data, n_neighbors=n_neighbors, metric=metric)
    else:
        raise NotImplementedError(f"Algorithm {algorithm_name} not found!")


if __name__ == '__main__':
    algorithm = ArgumentHelper.get_enum_argument("algorithm", Algorithm)

    labels = ['DERMASON', 'SIRA', 'SEKER']
    splitter = DataSplitHelper(DataCsvHelper.read_csv(labels=labels), 0.3)

    classification = create_classification_algorithm(algorithm, *splitter.split())
    classification.train()
    predicted, expected = classification.predict_test_data()

    print("\n")
    cmh = ConfusionMatrixHelper(expected, predicted, labels)
    cmh.display_confusion_matrix()
    cmh.display_classification_report()
    roc = RocCurveHelper(actual=expected, predicted=predicted)
    roc.plot_roc_curve()
