from CsvHelper import CsvHelper
from DataSplitter import DataSplitter
from NaiveBayes import NaiveBayes
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    splitter = DataSplitter(CsvHelper.read_csv(), 0.3)
    learn_data, test_data = splitter.split()
    naive_bayes = NaiveBayes(learn_data, test_data)
    naive_bayes.train()
    predicted, expected = naive_bayes.predict_test_data()
    print(accuracy_score(expected, predicted))