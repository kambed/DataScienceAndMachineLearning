from enum import Enum

from sklearn.naive_bayes import GaussianNB, CategoricalNB, MultinomialNB, ComplementNB, BernoulliNB

from algorithms.classification import Classification


class NaiveBayes(Classification):
    def __init__(self, learn_data, test_data, naive_bayes_type):
        if naive_bayes_type.value == 'GaussianNB':
            model = GaussianNB()
        elif naive_bayes_type.value == 'MultinomialNB':
            model = MultinomialNB()
        elif naive_bayes_type.value == 'ComplementNB':
            model = ComplementNB()
        elif naive_bayes_type.value == 'BernoulliNB':
            model = BernoulliNB()
        elif naive_bayes_type.value == 'CategoricalNB':
            model = CategoricalNB()
        else:
            raise NotImplementedError(f"Naive Bayes {naive_bayes_type} type not found!")
        super().__init__(learn_data, test_data, model)

class NaiveBayesType(Enum):
    GAUSSIAN = 'GaussianNB'
    MULTINOMINAL = 'MultinomialNB'
    COMPLEMENT = 'ComplementNB'
    BERNOULLI = 'BernoulliNB'
    CATEGORICAL = 'CategoricalNB'
