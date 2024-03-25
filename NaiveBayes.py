from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB, CategoricalNB


class NaiveBayes:
    def __init__(self, learn_data, test_data, naive_bayes_type):
        self.learn_data = learn_data
        self.test_data = test_data
        #GaussianNB, MultinomialNB, ComplementNB, BernoulliNB, CategoricalNB
        if naive_bayes_type == 'GaussianNB':
            self.model = GaussianNB()
        elif naive_bayes_type == 'MultinomialNB':
            self.model = MultinomialNB()
        elif naive_bayes_type == 'ComplementNB':
            self.model = ComplementNB()
        elif naive_bayes_type == 'BernoulliNB':
            self.model = BernoulliNB()
        elif naive_bayes_type == 'CategoricalNB':
            self.model = CategoricalNB()
        else:
            raise NotImplementedError(f"Naive Bayes {naive_bayes_type} type not found!")

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