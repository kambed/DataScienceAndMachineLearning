from sklearn.tree import DecisionTreeClassifier

from alghoritms.Classification import Classification


class DecisionTree(Classification):
    def __init__(self, learn_data, test_data, max_depth, min_samples_leaf):
        model = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
        super().__init__(learn_data, test_data, model)
