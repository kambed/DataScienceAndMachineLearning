import pandas as pd

from helper.data_split_helper import DataSplitHelper


class DataCsvHelper:

    @staticmethod
    def read_csv(labels):
        df = pd.read_csv("resources/Dry_Bean_Dataset.csv")
        selected_types = labels
        return df[df.Class.isin(selected_types)]

    @staticmethod
    def read_csv_data():
        try:
            learn_data = pd.read_csv("resources/learn_data.csv")
            test_data = pd.read_csv("resources/test_data.csv")
            return learn_data, test_data
        except FileNotFoundError:
            raise FileNotFoundError("Files not found!")

    @staticmethod
    def write_csv(labels, test_size):
        splitter = DataSplitHelper(DataCsvHelper.read_csv(labels=labels), test_size)
        learn_data, test_data = splitter.split()
        learn_data.to_csv("resources/learn_data.csv", index=False)
        test_data.to_csv("resources/test_data.csv", index=False)
        return learn_data, test_data
