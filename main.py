from helper.ConfusionMatrixHelper import ConfusionMatrixHelper
from helper.DataCsvHelper import DataCsvHelper
from helper.DataSplitHelper import DataSplitHelper
from alghoritms.Alghoritm import Algorithm
from alghoritms.NaiveBayes import NaiveBayes


def execute_knn(learn_data, test_data):
    return [], []


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
    algorithm = Algorithm.NAIVE_BAYES.value
    labels = ['DERMASON', 'SIRA', 'SEKER']
    splitter = DataSplitHelper(DataCsvHelper.read_csv(labels=labels), 0.3)

    if algorithm == Algorithm.NAIVE_BAYES.value:
        predicted, expected = execute_naive_bayes(*splitter.split())
    elif algorithm == Algorithm.KNN.value:
        predicted, expected = execute_knn(*splitter.split())
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
