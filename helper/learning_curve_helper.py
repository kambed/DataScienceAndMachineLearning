import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


class LearningCurveHelper:
    def __init__(self, classification, train_size, train_step):
        self.classification = classification
        self.train_size = train_size
        self.train_step = train_step
        self.train_scores = []
        self.test_scores = []
        self.test_scores_labels = []
        self.train_sizes = np.arange(train_step, train_size + train_step, train_step)

    def train_and_predict(self):
        for size in self.train_sizes:
            learn_size, _ = self.classification.train(self.train_size, size)

            predicted, expected = self.classification.predict_test_data()
            self.test_scores.append(accuracy_score(expected, predicted))

            predicted_train, expected_train = self.classification.predict_train_data(self.train_size, size)
            self.train_scores.append(accuracy_score(expected_train, predicted_train))

            self.test_scores_labels.append(f"{learn_size} ({round(size, 3)}%)")

        self.classification.train()
        return predicted, expected

    def display_learning_curve(self, range_start=0, range_end=0):
        if range_end == 0:
            range_end = len(self.train_sizes)
        plt.figure(figsize=(12, 10))
        plt.plot(self.train_sizes[range_start:range_end], self.test_scores[range_start:range_end], 'o-', color="g", label="Test score")
        plt.plot(self.train_sizes[range_start:range_end], self.train_scores[range_start:range_end], 'o-', color="r", label="Train score")
        plt.xticks(self.train_sizes[range_start:range_end], self.test_scores_labels[range_start:range_end])
        plt.xlabel("Number of training samples")
        plt.ylabel("Accuracy")
        plt.title("Learning Curve")
        plt.xticks(rotation=90)
        plt.legend(loc="best")
        plt.grid()
        plt.show()
