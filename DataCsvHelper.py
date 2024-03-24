import pandas as pd


class DataCsvHelper:

    @staticmethod
    def read_csv(labels):
        df = pd.read_csv("resources/Dry_Bean_Dataset.csv")
        selected_types = labels
        return df[df.Class.isin(selected_types)]
