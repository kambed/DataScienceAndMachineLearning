from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay


class ConfusionMatrixHelper:
    def __init__(self, actual, predicted, labels):
        self.labels = labels
        self.confusion_matrix = confusion_matrix(actual, predicted, labels=labels)
        self.classification_report = classification_report(actual, predicted, output_dict=True)

    def display_confusion_matrix(self):
        disp = ConfusionMatrixDisplay(confusion_matrix=self.confusion_matrix, display_labels=self.labels)
        disp.plot()
        plt.show()

    def display_classification_report(self):
        print(f"Accuracy: {self.classification_report['accuracy']}")
        for label in self.labels:
            print(f"=============={label}==============")
            print(f"Precision: {self.classification_report[label]['precision']}")
            print(f"Recall: {self.classification_report[label]['recall']}")
            print(f"F1-score: {self.classification_report[label]['f1-score']}")
            print(f"Specificity: {1 - float(self.classification_report[label]['recall'])}")