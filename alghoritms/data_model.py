"""
DataModel class is used to store data for learning and testing.
"""


class DataModel:

    """
    :param learn_data: Data used to learn
    :param test_data: Data used to test
    """
    def __init__(self, learn_data, test_data):
        self.learn_data = learn_data
        self.test_data = test_data
