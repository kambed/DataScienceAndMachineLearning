from ConfusionMatrixHelper import ConfusionMatrixHelper
from DataCsvHelper import DataCsvHelper
from DataSplitHelper import DataSplitHelper
from NaiveBayes import NaiveBayes


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


if __name__ == '__main__':
    algorithm = 'NAIVE_BAYES' # NAIVE_BAYES, KNN, SUPPORT_VECTOR, DECISION_TREE
    labels = ['DERMASON', 'SIRA', 'SEKER']
    splitter = DataSplitHelper(DataCsvHelper.read_csv(labels=labels), 0.3)

    if algorithm == 'NAIVE_BAYES':
        predicted, expected = execute_naive_bayes(*splitter.split())
    elif algorithm == 'KNN':
        predicted, expected = execute_knn(*splitter.split())
    elif algorithm == 'SUPPORT_VECTOR':
        predicted, expected = execute_support_vector_machine(*splitter.split())
    elif algorithm == 'DECISION_TREE':
        predicted, expected = execute_decision_tree(*splitter.split())
    else:
        raise NotImplementedError(f"Algorithm {algorithm} not found!")

    cmh = ConfusionMatrixHelper(expected, predicted, labels)
    cmh.display_confusion_matrix()
    cmh.display_classification_report()
