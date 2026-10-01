import numpy as np


class BernoulliNaiveBayes:
    """Bernoulli Naive Bayes implemented from scratch."""

    def __init__(self):
        self.classes_ = None
        self.class_prior_ = None
        self.class_log_prior_ = None
        self.feature_prob_ = None
        self.feature_log_prob_ = None
        self.feature_log_neg_prob_ = None
        self._log_score_intercept_ = None
        self._log_score_weight_ = None

    def fit(self, X, y):
        """Estimate class priors and Bernoulli feature probabilities.

        Use add-one smoothing:
            P(x_d = 1 | y = c) = (N_dc + 1) / (N_c + 2)
        """
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must be a 2-dimensional array")
        if y.ndim != 1 or len(y) != X.shape[0]:
            raise ValueError("y must be a 1-dimensional array matching X")
        if X.shape[0] == 0:
            raise ValueError("cannot fit on an empty dataset")
        if not np.all(np.isfinite(X)) or not np.all((X == 0) | (X == 1)):
            raise ValueError("X must contain only 0/1 values")
        if not np.all(np.isin(y, (0, 1))):
            raise ValueError("y must contain only labels 0 and 1")

        y = y.astype(np.int64, copy=False)
        self.classes_ = np.array([0, 1], dtype=np.int64)

        counts = np.bincount(y, minlength=2).astype(np.float64)
        self.class_prior_ = counts / len(y)
        self.class_log_prior_ = np.full(2, -np.inf, dtype=np.float64)
        present = counts > 0
        self.class_log_prior_[present] = np.log(self.class_prior_[present])

        # For each class and feature, smooth the number of documents in
        # that class containing the feature with one success and one failure.
        feature_counts = np.empty((2, X.shape[1]), dtype=np.float64)
        for class_label in self.classes_:
            # Sum in chunks so fitting does not materialize a second copy of
            # the full training matrix when X is a compact integer array.
            rows = np.flatnonzero(y == class_label)
            total = np.zeros(X.shape[1], dtype=np.float64)
            for start in range(0, len(rows), 4096):
                total += X[rows[start:start + 4096]].sum(axis=0, dtype=np.float64)
            feature_counts[class_label] = total
        denominators = counts[:, None] + 2.0
        self.feature_prob_ = (feature_counts + 1.0) / denominators
        self.feature_log_prob_ = np.log(self.feature_prob_)
        self.feature_log_neg_prob_ = np.log1p(-self.feature_prob_)
        self._log_score_intercept_ = (
            self.class_log_prior_ + self.feature_log_neg_prob_.sum(axis=1)
        )
        self._log_score_weight_ = (
            self.feature_log_prob_ - self.feature_log_neg_prob_
        ).T

        # Descriptive aliases are useful to callers inspecting the fitted
        # model and preserve the usual BernoulliNB naming conventions.
        self.class_priors_ = self.class_prior_
        self.feature_probs_ = self.feature_prob_
        self.class_priors = self.class_prior_
        self.feature_probs = self.feature_prob_
        return self

    def predict_probs(self, X):
        """Return normalized class posterior probabilities for all rows of X."""
        if self.classes_ is None:
            raise RuntimeError("model must be fitted before prediction")
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[1] != self.feature_prob_.shape[1]:
            raise ValueError("X has the wrong shape for this model")
        if not np.all(np.isfinite(X)) or not np.all((X == 0) | (X == 1)):
            raise ValueError("X must contain only 0/1 values")
        if X.shape[0] == 0:
            return np.empty((0, 2), dtype=np.float64)
        # Equivalent compact form of the Bernoulli log likelihood:
        # intercept + sum_d x_d * (log p_d - log(1-p_d)).
        log_scores = self._log_score_intercept_[None, :] + X @ self._log_score_weight_
        max_scores = np.max(log_scores, axis=1, keepdims=True)
        # Rows with both scores -inf are only possible for a class absent
        # from training; return a well-defined distribution in that case.
        finite_max = np.isfinite(max_scores[:, 0])
        probs = np.zeros_like(log_scores)
        if np.any(finite_max):
            shifted = np.exp(log_scores[finite_max] - max_scores[finite_max])
            probs[finite_max] = shifted / shifted.sum(axis=1, keepdims=True)
        if np.any(~finite_max):
            probs[~finite_max] = 0.5
        return probs

    def predict(self, X):
        """Predict class labels for all rows of X."""
        probabilities = self.predict_probs(X)
        return self.classes_[np.argmax(probabilities, axis=1)]
