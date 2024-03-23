import pandas as pd


class CsvHelper:

    @staticmethod
    def read_csv():
        return pd.read_csv("resources/Dry_Bean_Dataset.csv")
