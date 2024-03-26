class Classification:
    def __init__(self, learn_data, test_data, model):
        self.learn_data = learn_data
        self.test_data = test_data
        self.model = model

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
