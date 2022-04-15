from typing import NoReturn
from ...base import BaseEstimator
import numpy as np

from ...metrics import misclassification_error


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier
        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`
        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`
        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`
        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for
        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        # calculate classes
        y_mean_classified_as_two = np.mean(y == 2)
        y_mean_classified_as_one = np.mean(y == 1)
        y_mean_classified_as_zero = np.mean(y == 0)
        self.classes_ = np.array([y_mean_classified_as_zero, y_mean_classified_as_one, y_mean_classified_as_two])
        # calculate mu
        x_mean_classified_as_two = np.mean(X[y == 2], axis=0)
        x_mean_classified_as_one = np.mean(X[y == 1], axis=0)
        x_mean_classified_as_zero = np.mean(X[y == 0], axis=0)
        self.mu_ = np.transpose(np.array([x_mean_classified_as_zero, x_mean_classified_as_one, x_mean_classified_as_two]))
        # calculate vars matrix
        xi_twos_minus_mu = X[y == 2] - self.mu_[:, 2]
        xi_ones_minus_mu = X[y == 1] - self.mu_[:, 1]
        xi_zeros_minus_mu = X[y == 0] - self.mu_[:, 0]
        self.vars_ = np.matmul(np.transpose(xi_twos_minus_mu), xi_twos_minus_mu)
        self.vars_ += np.matmul(np.transpose(xi_ones_minus_mu), xi_ones_minus_mu)
        self.vars_ += np.matmul(np.transpose(xi_zeros_minus_mu), xi_zeros_minus_mu)
        self.vars_ /= y.size  # using the unbiased estimator
        # calculate vars inverse matrix
        self._vars_inv = np.linalg.inv(self.vars_)
        # calculate pi
        self.pi_ = -0.5 * np.diag(np.matmul(np.matmul(np.transpose(self.mu_), self._vars_inv), self.mu_))

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for
        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        ak = np.matmul(self._vars_inv, self.mu_)
        bk = self.classes_ + self.pi_
        return np.argmax(np.matmul(X, ak) + bk, axis=1)

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.
        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        return np.exp(self.log_likelihood(self, X))

    def log_likelihood(self, X: np.ndarray) -> np.ndarray:
        d_half_log_two_pi = -0.5 * X.size * np.log(2 * np.pi)
        log_cov = -0.5 * np.log(self.vars_)
        X_minus_mu_inv_cov = -0.5 * np.matmul(np.matmul(X - self.mu_, self._vars_inv), X - self.mu_)
        return np.log(self.classes_) + d_half_log_two_pi + log_cov + X_minus_mu_inv_cov

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples
        y : ndarray of shape (n_samples, )
            True labels of test samples
        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return misclassification_error(y, self.predict(X))
