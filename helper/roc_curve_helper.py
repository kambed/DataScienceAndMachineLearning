from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize, LabelEncoder

"""
    This module is used to help plot the ROC curve for a binary classification model.
"""


class RocCurveHelper:

    """
        :param actual: Actual values
        :param predicted: Predicted values
    """
    def __init__(self, actual, predicted):
        self.actual = actual
        self.predicted = predicted

    """
        Plot the ROC curve
    """
    def plot_roc_curve(self):
        colors = ['blue', 'green', 'red']
        le = LabelEncoder()
        actual_labels = le.fit_transform(self.actual)
        classes = le.classes_

        predicted_labels = label_binarize(self.predicted, classes=classes)

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(len(classes)):
            fpr[i], tpr[i], _ = roc_curve(actual_labels == i, predicted_labels[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure()
        for i, color in zip(range(len(classes)), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label='ROC curve of class {0} (area = {1:0.2f})'.format(classes[i], roc_auc[i]))

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()
