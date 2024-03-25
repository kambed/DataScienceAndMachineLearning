from sklearn.naive_bayes import GaussianNB

from alghoritms.Classification import Classification


class NaiveBayes(Classification):
    def __init__(self, learn_data, test_data):
        model = GaussianNB()
        super().__init__(learn_data, test_data, model)