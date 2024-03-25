from sklearn.ensemble import RandomForestClassifier

from alghoritms.Classification import Classification


class RandomForest(Classification):
    def __init__(self, learn_data, test_data, n_estimators, max_features):
        model = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features)
        super().__init__(learn_data, test_data, model)
