from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    X = np.linspace(-1.2, 2, n_samples)
    epsilon = np.random.normal(0, noise, n_samples)
    poly = lambda x: (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)
    y = poly(X) + epsilon
    train_X, train_y, test_X, test_y = split_train_test(pd.DataFrame(X), pd.Series(y), float(2 / 3))
    train_X, train_y = train_X[0].to_numpy(), train_y.to_numpy()
    test_X, test_y = test_X[0].to_numpy(), test_y.to_numpy()
    # scatter plot
    go.Figure([go.Scatter(x=X, y=poly(X), mode='lines', name="noiseless true model"),
               go.Scatter(x=train_X, y=train_y, mode='markers', name="Train Set"),
               go.Scatter(x=test_X, y=test_y, mode='markers',
                          marker=dict(color="#0000ff"), name="Test Set"), ]) \
        .update_layout(title="True model and test and size sets",
                       xaxis_title="x axis", yaxis_title="y axis"). \
        show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    k = np.arange(11)
    train_err = np.zeros(k.size)
    test_err = np.zeros(k.size)
    for i in k:
        train_err[i], test_err[i] = cross_validate(PolynomialFitting(i), train_X, train_y, mean_square_error, 5)
    go.Figure([go.Scatter(x=k, y=train_err, name="Train error"),
               go.Scatter(x=k, y=test_err, name="Validation error")]) \
        .update_layout(title="Train and Validation error as a function of Polynomial",
                       xaxis_title="degree", yaxis_title="error"). \
        show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    minimum = np.argmin(test_err)
    fit = PolynomialFitting(minimum).fit(train_X, train_y)
    err = round(mean_square_error(test_y, fit.predict(test_X)), 2)
    print(f"The minimum index is = {minimum}, with Error of  {err}")


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    train_x = X[:n_samples, :]
    train_y = y[:n_samples]
    test_x = X[n_samples:, :]
    test_y = y[n_samples:]

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lam = np.linspace(0.000001, 3.5, n_evaluations)
    ridge_train_error = np.zeros(lam.size)
    ridge_test_error = np.zeros(lam.size)
    lasso_train_error = np.zeros(lam.size)
    lasso_test_error = np.zeros(lam.size)
    for i, l in enumerate(lam):
        ridge = RidgeRegression(l)
        ridge_train_error[i], ridge_test_error[i] = cross_validate(ridge, train_x, train_y, mean_square_error)
        lasso = Lasso(l)
        lasso_train_error[i], lasso_test_error[i] = cross_validate(lasso, train_x, train_y, mean_square_error)
    go.Figure([go.Scatter(x=lam, y=ridge_train_error, name="Ridge train error"),
               go.Scatter(x=lam, y=ridge_test_error, name="Ridge validation error"),
               go.Scatter(x=lam, y=lasso_train_error, name="Lasso train error"),
               go.Scatter(x=lam, y=lasso_test_error, name="Lasso validation error")]) \
        .update_layout(title="Lasso and Ridge train and validation error as a function of lambda",
                       xaxis_title="Lambda", yaxis_title="Errors"). \
        show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    minimum = np.argmin(ridge_train_error)
    # Ridge
    ridge = RidgeRegression(lam[minimum]).fit(train_x, train_y)
    ridge_loss = ridge.loss(test_x, test_y)
    # Lasso
    lasso = Lasso(lam[minimum]).fit(train_x, train_y)
    lasso_loss = mean_square_error(lasso.predict(test_x), test_y)
    # Linear Regression
    lr = LinearRegression(lam[minimum]).fit(train_x, train_y)
    lr_loss = lr.loss(test_x, test_y)

    print(f"The minimum lamdbda is = {lam[minimum]}, with Lasso Error of  {round(lasso_loss, 2)},"
          f" and Ridge Error of {round(ridge_loss, 2)},"
          f"and LS Error of {round(lr_loss, 2)} ")


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()  # Question 1-3
    select_polynomial_degree(noise=0)  # Question 4
    select_polynomial_degree(n_samples=1500, noise=10)  # Question 5
    select_regularization_parameter()  # Questions 6-8
