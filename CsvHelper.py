import pandas as pd


class CsvHelper:

    @staticmethod
    def read_csv():
        df = pd.read_csv("resources/Dry_Bean_Dataset.csv")
        selected_types = ['DERMASON', 'SIRA', 'SEKER']
        return df[df.Class.isin(selected_types)]
