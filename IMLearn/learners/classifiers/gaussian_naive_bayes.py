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
        # Classes
        self.classes_ = np.unique(y)
        # mu - same as LDA
        arr = []
        x_column_shape = X.shape[1]
        for i, j in enumerate(self.classes_):
            arr.append(np.mean(X[y == i], axis=0))
        self.mu_ = np.array(arr)
        # vars - different from LDA
        self.vars_ = [[[] for i in range(x_column_shape)] for j in range(self.classes_.size)]
        # iterate through [x value, y corresponding value, the current class]
        for sample in np.r_['-1,2,0', X, y]:
            row = int(sample[-1])
            # iterate through each x value and add the y corresponding value to the vars array
            for x_val, y_corresponding_val in enumerate(sample[:-1]):
                self.vars_[row][x_val].append(y_corresponding_val)
        # iterate through rows
        for row in range(self.classes_.size):
            # iterate through columns and calculate variance
            for column in range(X.shape[1]):
                self.vars_[row][column] = np.var(np.array(self.vars_[row][column]))
        self.vars_ = np.array(self.vars_)
        # pi
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
        # array of predictions
        prediction = np.ndarray(X.shape[0])
        # go through all of X
        for i, x in enumerate(X):
            arr = np.ndarray(self.classes_.size)
            # iterate through the amount of possible answers
            for j in range(self.classes_.size):
                log_pi = np.log(self.pi_[j])
                summ = 0
                # iterate through X columns
                for k in range(X.shape[1]):
                    log_two_pi_vars = np.log(2 * np.pi * self.vars_[j][k])
                    x_minus_mu_squared = (x[k] - self.mu_[j][k]) ** 2
                    summ += sum([log_two_pi_vars + x_minus_mu_squared / self.vars_[j][k]])
                arr[j] = log_pi - 0.5 * summ
            prediction[i] = self.classes_[np.argmax(arr)]
        return prediction

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
                    pdf *= (1 / np.sqrt(2 * np.pi * self.vars_[classes][column])) * np.exp(
                        -0.5 * ((x[column] - self.mu_[classes][column]) ** 2) / self.vars_[classes][column])
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
        return misclassification_error(y, self._predict(X))
