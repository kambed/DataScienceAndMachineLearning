import numpy as np
import pandas as pd


class Classification:
    def __init__(self, learn_data, test_data, model):
        self.learn_data = learn_data
        self.start_test_data = test_data
        self.test_data = test_data
        self.start_model = model
        self.model = model

    def train(self, learn_data_size=0, learn_data_part=0):
        self.model = self.start_model
        self.test_data = self.start_test_data
        if learn_data_size == 0:
            data_to_train = self.learn_data
        else:
            data_to_train_size = int(learn_data_part * self.learn_data.index.size / learn_data_size)
            data_to_train = self.learn_data.head(data_to_train_size)
            self.test_data = pd.concat([self.test_data, self.learn_data.head(data_to_train_size - self.learn_data.index.size)])

        x = data_to_train.iloc[:, :-1].values
        y = data_to_train.iloc[:, -1].values
        self.model.fit(x, y)
        return data_to_train.index.size, self.test_data.index.size

    def predict(self, x):
        return self.model.predict(x)

    def predict_test_data(self):
        x = self.test_data.iloc[:, :-1].values
        y = self.test_data.iloc[:, -1].values
        return self.predict(x), y