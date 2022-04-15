from matplotlib import pyplot as plt
from plotnine import ggsave, ggplot, geom_smooth, geom_point, labs

from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.io as pio
import plotnine as pl

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    data = pd.read_csv(filename).dropna().drop_duplicates()
    data = data.drop(['id', 'date', 'lat', 'long'], axis=1)
    for index in ['price', 'sqft_living', 'sqft_lot', 'sqft_above',
                  'sqft_living15', 'sqft_lot15', 'floors']:
        data = data[data[index] > 0]
    for index in ['waterfront', 'yr_renovated', 'sqft_basement']:
        data = data[data[index] >= 0]
    data = data[data['view'].isin(range(5))]
    data = data[data['condition'].isin(range(1, 6))]
    data = data[data['grade'].isin(range(1, 13))]
    data = data[data['bathrooms'].isin(range(1, 20))]
    data = data[data['bedrooms'].isin(range(1, 30))]
    data = data[data['sqft_lot'] < 100000]
    data = data[data['sqft_lot15'] < 100000]
    data = data[data['sqft_above'] < 100000]
    data = data[data['sqft_living'] < 100000]
    data = data[data['sqft_living15'] < 100000]
    data = data[data['floors'] < 100]
    data["decade_built"] = data["yr_built"] / 10
    data.drop('yr_built', axis=1)
    data["decade_renoavted"] = data["yr_renovated"] / 10
    data.drop('yr_renovated', axis=1)
    data["last_renovation"] = data[["decade_renoavted", "decade_built"]].max(axis=1)
    data.drop('decade_renoavted', axis=1)
    data.drop('decade_built', axis=1)
    data = pd.get_dummies(data, columns=['decade_built'])
    data = pd.get_dummies(data, columns=['decade_renoavted'])
    data = pd.get_dummies(data, columns=["zipcode"])
    price = data['price']
    data = data.drop('price', axis=1)

    return data, price


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """

    for i in X:
        if 'zipcodes' in X.columns:
            continue
        i_data = X[i]
        num = np.cov(i_data, y)[1, 0]
        den = np.std(i_data) * np.std(y)
        pearson_corr = num / den
        plt.scatter(x=i_data, y=y)
        plt.title("Question 2 - scatter-plot: correlation between "
                  f"{i} and response\n Pearson Correlation ="
                  f" {np.around(pearson_corr, decimals=5)} ")
        plt.xlabel(i)
        plt.ylabel("response labels")
        plt.savefig(f"{output_path}\\{i}.png")
        plt.close()


def fit_model(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray):
    results = []  # mean
    var = []  # variance
    for i in range(10, 101):
        percent = i / 100
        temp = []
        for j in range(10):
            x_i = train_x.sample(frac=percent)
            y_i = train_y[x_i.index]
            lr.fit(x_i, y_i)
            loss = lr.loss(test_x, test_y)
            temp.append(loss)
        var.append(np.std(temp))
        results.append(sum(temp) / len(temp))
    results = np.asarray(results)
    var = np.asarray(var)
    fig, ax = plt.subplots()
    ax.plot(np.arange(10, 101), results)
    plt.xlabel("P % ")
    plt.ylabel("Mean Loss")
    ax.fill_between(np.arange(10, 101), (results - 2 * var), (results + 2 * var), color='b', alpha=.1)
    plt.show()


if __name__ == '__main__':
    np.random.seed(0)
    lr = LinearRegression()
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data("../datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y, "C:\\Users\\galbu\\IML.HUJI\\exercises\\ex 2")

    # Question 3 - Split samples into training- and testing sets.
    train_x, train_y, test_x, test_y = split_train_test(X, y, 0.75)

    # Question 4 - Fit model over increasing percentages of the overall training data
    fit_model(train_x, train_y, test_x, test_y)

    # QUIZ:
    # Question 2:
    y_true = np.array([279000, 432000, 326000, 333000, 437400, 555950])
    y_pred = np.array(
        [199000.37562541, 452589.25533196, 345267.48129011, 345856.57131275, 563867.1347574, 395102.94362135])
    print("MSE of the question in the quiz is: ", mean_square_error(y_true, y_pred))
