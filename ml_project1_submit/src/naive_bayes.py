import numpy as np


class BernoulliNaiveBayes:
    """Bernoulli Naive Bayes implemented from scratch."""

    def __init__(self):
        self.classes_ = None

    def fit(self, X, y):
        """Estimate class priors and Bernoulli feature probabilities.

        Use add-one smoothing:
            P(x_d = 1 | y = c) = (N_dc + 1) / (N_c + 2)
        """
        # TODO: implement this function.
        raise NotImplementedError

    def predict_probs(self, X):
        """Return normalized class posterior probabilities for all rows of X."""
        # TODO: implement this function.
        raise NotImplementedError

    def predict(self, X):
        """Predict class labels for all rows of X."""
        # TODO: implement this function.
        raise NotImplementedError
