from sklearn.naive_bayes import GaussianNB


class NaiveBayes:
    def __init__(self, learn_data, test_data):
        self.learn_data = learn_data
        self.test_data = test_data
        self.model = GaussianNB()

    def train(self):
        x = self.learn_data.iloc[:, :-1].values
        y = self.learn_data.iloc[:, -1].values
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def predict_test_data(self):
        x = self.test_data.iloc[:, :-1].values
        y = self.test_data.iloc[:, -1].values
        return self.predict(x), y