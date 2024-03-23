from sklearn.model_selection import train_test_split

class DataSplitter:
    def __init__(self, data, test_size=0.2):
        self.data = data
        self.test_size = test_size

    def split(self):
        return train_test_split(self.data, test_size=self.test_size)