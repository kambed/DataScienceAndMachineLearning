from alghoritms.k_nearest_neighbours import KNearestNeighbours, Metric
from helper.confusion_matrix_helper import ConfusionMatrixHelper
from helper.data_csv_helper import DataCsvHelper
from helper.data_split_helper import DataSplitHelper
from alghoritms.alghoritm import Algorithm
from alghoritms.naive_bayes import NaiveBayes
from helper.roc_curve_helper import RocCurveHelper


def execute_knn(n_neighbors, metric, learn_data, test_data):
    """
        :param n_neighbors: Number of neighbours
        :param metric: Metric used to calculate distance
        :param learn_data: Data used to learn
        :param test_data: Data used to test
    """
    knn = KNearestNeighbours(n_neighbors=n_neighbors, metric=metric, learn_data=learn_data, test_data=test_data)
    knn.train()
    return knn.predict_test_data()


def execute_naive_bayes(learn_data, test_data):
    naive_bayes = NaiveBayes(learn_data, test_data)
    naive_bayes.train()
    return naive_bayes.predict_test_data()


def execute_support_vector_machine(learn_data, test_data):
    return [], []


def execute_decision_tree(learn_data, test_data):
    return [], []


def execute_random_forest(learn_data, test_data):
    return [], []


if __name__ == '__main__':
    algorithm = Algorithm.KNN.value
    labels = ['DERMASON', 'SIRA', 'SEKER']
    splitter = DataSplitHelper(DataCsvHelper.read_csv(labels=labels), 0.3)

    if algorithm == Algorithm.NAIVE_BAYES.value:
        predicted, expected = execute_naive_bayes(*splitter.split())
    elif algorithm == Algorithm.KNN.value:
        predicted, expected = execute_knn(5, Metric.MANHATTAN, *splitter.split())
    elif algorithm == Algorithm.SUPPORT_VECTOR.value:
        predicted, expected = execute_support_vector_machine(*splitter.split())
    elif algorithm == Algorithm.DECISION_TREE.value:
        predicted, expected = execute_decision_tree(*splitter.split())
    elif algorithm == Algorithm.RANDOM_FOREST.value:
        predicted, expected = execute_random_forest(*splitter.split())
    else:
        raise NotImplementedError(f"Algorithm {algorithm} not found!")

    cmh = ConfusionMatrixHelper(expected, predicted, labels)
    cmh.display_confusion_matrix()
    cmh.display_classification_report()
    roc = RocCurveHelper(expected, predicted)
    roc.plot_roc_curve()
