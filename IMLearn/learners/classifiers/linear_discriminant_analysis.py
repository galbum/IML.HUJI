from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv

from ...metrics import misclassification_error


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier
    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`
    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`
    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`
    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`
    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `LDA.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for
        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        # calculate classes
        self.classes_ = np.unique(y)
        # calculate mu
        arr = []
        for i, j in enumerate(self.classes_):
            arr.append(np.mean(X[y == i], axis=0))
        self.mu_ = np.transpose(np.array(arr))
        # calculate cov matrix
        xi_minus_mu = []
        for i, j in enumerate(self.classes_):
            cov = X[y == i] - self.mu_[:, i]
            xi_minus_mu.append(cov)
        xi_minus_mu = np.array(xi_minus_mu, dtype=object)
        self.cov_ = 0
        for i in range(len(xi_minus_mu)):
            self.cov_ += np.matmul(np.transpose(xi_minus_mu[i]), xi_minus_mu[i])
        self.cov_ /= y.size
        # inverse cov matrix
        self._cov_inv = np.linalg.inv(self.cov_)
        # calculate pi
        mean = []
        for i, j in enumerate(self.classes_):
            mean.append(np.mean(y == i))
        self.pi_ = np.array(mean)

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
        X_cov_mu = np.matmul(np.transpose(self._cov_inv), self.mu_)
        mu_covinv_mu = -0.5 * np.diag(np.matmul(np.matmul(np.transpose(self.mu_), self._cov_inv), self.mu_))
        return np.argmax(np.matmul(X, X_cov_mu) + mu_covinv_mu + np.log(self.pi_), axis=1)

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
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        likelihoods = np.ndarray((X.shape[0], self.classes_.size))
        # iterate through X
        for i, x in enumerate(X):
            # iterate through the amount of possible answers
            for classes in range(self.classes_.size):
                pdf = 1
                # iterate through X columns
                for column in range(X.shape[1]):
                    pdf *= (1 / np.sqrt(2 * np.pi * self.cov_[classes][column])) * np.exp(
                        -0.5 * ((x[column] - self.mu_[classes][column]) ** 2) / self.cov_[classes][column])
                likelihoods[i][classes] = self.pi_[classes] * pdf
        return likelihoods

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