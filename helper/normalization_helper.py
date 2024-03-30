from enum import Enum

from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd


class NormalizationHelper:

    def __init__(self, type):
        self.scaler = type.value

    def preprocess(self, learn_data, test_data):
        data = pd.concat([learn_data, test_data])
        number_columns = data.select_dtypes(include='number').columns.tolist()
        self.scaler.fit(data[number_columns])
        learn_data[number_columns] = self.scaler.transform(learn_data[number_columns])
        test_data[number_columns] = self.scaler.transform(test_data[number_columns])
        return learn_data, test_data


class NormalizationType(Enum):
    MIN_MAX = MinMaxScaler()
    STANDARD = StandardScaler()
