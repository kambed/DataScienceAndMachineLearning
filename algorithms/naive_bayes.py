from enum import Enum

from sklearn.naive_bayes import GaussianNB, CategoricalNB, MultinomialNB, ComplementNB, BernoulliNB

from algorithms.classification import Classification


class NaiveBayes(Classification):
    def __init__(self, learn_data, test_data, naive_bayes_type):
        super().__init__(learn_data, test_data, naive_bayes_type.value)


class NaiveBayesType(Enum):
    GAUSSIAN = GaussianNB()
    MULTINOMINAL = MultinomialNB()
    COMPLEMENT = ComplementNB()
    BERNOULLI = BernoulliNB()
    CATEGORICAL = CategoricalNB()
