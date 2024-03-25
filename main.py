from alghoritms.RandomForest import RandomForest
from alghoritms.DecisionTree import DecisionTree
from helper.ConfusionMatrixHelper import ConfusionMatrixHelper
from helper.DataCsvHelper import DataCsvHelper
from helper.DataSplitHelper import DataSplitHelper
from alghoritms.Alghoritm import Algorithm
from alghoritms.NaiveBayes import NaiveBayes


def create_classification_algorithm(algorithm_name, learn_data, test_data):
    if algorithm_name == Algorithm.NAIVE_BAYES:
        return NaiveBayes(learn_data, test_data)
    elif algorithm_name == Algorithm.DECISION_TREE:
        return DecisionTree(learn_data, test_data, 5, 2)
    elif algorithm_name == Algorithm.RANDOM_FOREST:
        return RandomForest(learn_data, test_data, 100, 2)
    else:
        raise NotImplementedError(f"Algorithm {algorithm_name} not found!")


if __name__ == '__main__':
    algorithm = Algorithm.RANDOM_FOREST
    labels = ['DERMASON', 'SIRA', 'SEKER']
    splitter = DataSplitHelper(DataCsvHelper.read_csv(labels=labels), 0.3)

    classification = create_classification_algorithm(algorithm, *splitter.split())
    classification.train()
    predicted, expected = classification.predict_test_data()

    cmh = ConfusionMatrixHelper(expected, predicted, labels)
    cmh.display_confusion_matrix()
    cmh.display_classification_report()
