class Classification:
    def __init__(self, learn_data, test_data, model):
        self.learn_data = learn_data
        self.test_data = test_data
        self.model = model

    def train(self, part_of_data=1):
        data_to_train = self.learn_data.sample(frac=part_of_data)
        x = data_to_train.iloc[:, :-1].values
        y = data_to_train.iloc[:, -1].values
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def predict_test_data(self):
        x = self.test_data.iloc[:, :-1].values
        y = self.test_data.iloc[:, -1].values
        return self.predict(x), y
