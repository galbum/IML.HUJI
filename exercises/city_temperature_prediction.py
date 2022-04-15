from matplotlib import pyplot as plt

import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    data = pd.read_csv(filename, parse_dates=['Date']).dropna().drop_duplicates()
    data['DayOfYear'] = data['Date'].dt.day_of_year
    data = pd.get_dummies(data, columns=["City"])
    data = data.drop(['Date'], axis=1)
    data = data[data['Day'].isin(range(1, 32))]
    data = data[data['Month'].isin(range(1, 13))]
    data = data[data['Temp'] > -40]

    return data


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    data = load_data("../datasets/City_Temperature.csv")
    question2 = data
    question3 = data
    question4 = data
    question5 = data
    # Question 2 - Exploring data for specific country

    # part 1
    data = question2
    data = data[data['Country'] == "Israel"]
    groups = data.groupby('Year')
    fig, ax = plt.subplots()
    for name, group in groups:
        ax.plot(group.DayOfYear, group.Temp, marker='o', linestyle='', ms=2, label=name)
    ax.legend(numpoints=1, loc='upper left')
    plt.title("Question 2 - scatter-plot between average daily temperature and day of year ")
    plt.ylabel("average daily temperature")
    plt.xlabel("day of year")
    plt.show()

    # part 2
    data = question2
    data = data[data['Country'] == "Israel"]
    d = data.groupby(['Month']).agg("std")
    d.Temp.plot.barh()
    plt.title("Standard deviation of the daily temperatures")
    plt.xlabel("Temp")
    plt.show()

    # Question 3 - Exploring differences between countries
    data = question3
    fig, ax = plt.subplots()
    countries = set(data['Country'].to_dict().values())
    for i in countries:
        df_country = data[data['Country'] == i]
        avg = df_country.groupby(['Country', 'Month']).agg('mean')
        std = df_country.groupby(['Country', 'Month']).agg('std')
        ax.errorbar(np.arange(1, 13), avg.Temp, std.Temp, capsize=6)
    plt.title("Average Monthly Temperature")
    plt.xlabel("Months")
    plt.ylabel("Average temp")
    plt.show()

    # Question 4 - Fitting model for different values of `k`
    data = question4
    data = data[data['Country'] == "Israel"]
    train_x, train_y, test_x, test_y = split_train_test(data, data['Temp'], 0.75)

    loss = []
    for i in range(1, 11):
        pf = PolynomialFitting(i)
        pf.fit(train_x.DayOfYear, train_y)
        lost = np.around(pf.loss(test_x.DayOfYear.to_numpy(), test_y.to_numpy()), decimals=2)
        loss.append(lost)
        print(i, lost)

    plt.bar(np.arange(1, 11), loss)
    plt.title("Test error for evey value k")
    plt.xlabel("K")
    plt.ylabel("Test error")
    plt.show()
    print("\n")

    # Question 5 - Evaluating fitted model on different countries
    data = question5
    countries = set(data['Country'].to_dict().values())
    israel = data[data['Country'] == "Israel"]
    k = 5  # I chose k = 5 from last question
    pf = PolynomialFitting(k)
    pf.fit(israel.DayOfYear, israel.Temp)
    countries.remove("Israel")

    fig, ax = plt.subplots()
    loss = []
    index = []
    for i in countries:
        df_country = data[data['Country'] == i]
        lost = np.around(pf.loss(df_country.DayOfYear.to_numpy(), df_country.Temp.to_numpy()), decimals=2)
        loss.append(lost)
        index.append(i)
        print(i, lost)
    ax.bar(index, loss)
    plt.title("Model error other countries")
    plt.xlabel("Country")
    plt.ylabel("Model error")
    plt.show()
