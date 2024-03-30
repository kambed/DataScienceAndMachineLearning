import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


class LearningCurveHelper:
    def __init__(self, classification, train_size, train_step):
        self.classification = classification
        self.train_size = train_size
        self.train_step = train_step
        self.test_scores = []
        self.train_sizes = np.arange(train_step, train_size + train_step, train_step)

    def train_and_predict(self):
        for size in self.train_sizes:
            self.classification.train(self.train_size, size)
            predicted, expected = self.classification.predict_test_data()
            self.test_scores.append(accuracy_score(expected, predicted))

        self.classification.train()
        return predicted, expected

    def display_learning_curve(self):
        plt.figure(figsize=(10, 8))
        plt.plot(self.train_sizes, self.test_scores, 'o-', color="g", label="Test score")
        plt.xticks(self.train_sizes)
        plt.xlabel("Training data percentage")
        plt.gca().set_xticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_xticks()])
        plt.ylabel("Accuracy")
        plt.title("Learning Curve")
        plt.xticks(rotation=90)
        plt.legend(loc="best")
        plt.grid()
        plt.show()
