import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test
from IMLearn.model_selection.cross_validate import cross_validate
from IMLearn.metrics.loss_functions import misclassification_error
from sklearn.metrics import roc_curve, auc

import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly
import warnings
from utils import custom


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm
    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted
    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path
    title: str, default=""
        Setting details to add to plot title
    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range
    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range
    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown
    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines",
                                 marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration
    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm
    values: List[np.ndarray]
        Recorded objective values
    weights: List[np.ndarray]
        Recorded parameters
    """
    values, weights = [], []

    def call_back(model, **kwargs):
        values.append(kwargs["val"])
        weights.append(kwargs["weights"])
        return

    return call_back, values, weights


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    def plot_norm(x, y, tit):
        plt.plot(x, y)
        plt.title(tit)
        plt.xlabel("Iterations")
        plt.ylabel("Norm")
        plt.grid()
        plt.show()

    # L1
    for index in etas:
        l1 = L1(init.copy())
        get_GD = get_gd_state_recorder_callback()
        gd = GradientDescent(learning_rate=FixedLR(index), out_type="best", callback=get_GD[0])
        GD_fit = gd.fit(l1, X=None, y=None)
        plotly.offline.plot(plot_descent_path(L1, np.array(get_GD[2]), title=f"L1 module with eta={index}"))
        plot_norm(list(range(len(get_GD[1]))), get_GD[1],
                  f"norm of L1 as a function of the Gradient Desent iteration with eta={index}")
        l1.weights = GD_fit
        print(f"eta: {index}")
        print(f"L1 module with lowest error: {l1.compute_output()}", end='\n')

    # L2
    for index in etas:
        l2 = L2(init.copy())
        get_GD = get_gd_state_recorder_callback()
        gd = GradientDescent(learning_rate=FixedLR(index), out_type="best", callback=get_GD[0])
        GD_fit = gd.fit(l2, X=None, y=None)
        plotly.offline.plot(plot_descent_path(L2, np.array(get_GD[2]), title=f" L2 module with eta={index}"))
        plot_norm(list(range(len(get_GD[1]))), get_GD[1],
                  f"norm of L2 as a function of the Gradient Desent iteration with eta={index}")
        l2.weights = GD_fit
        print(f"eta: {index}")
        print(f"L2 module with lowest error: {l2.compute_output()}")


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    convergence, decay = [], []
    for index in gammas:
        GD_states = get_gd_state_recorder_callback()
        gd = GradientDescent(learning_rate=ExponentialLR(base_lr=eta, decay_rate=index), out_type="best", callback=GD_states[0])
        l1 = L1(init.copy())
        gd.fit(l1, X=None, y=None)
        convergence.append(GD_states[1])
        if index == 0.95:
            decay = GD_states[2]

    # Plot algorithm's convergence for the different values of gamma
    for index in range(4):
        plt.plot(list(range(len(convergence[index]))), convergence[index])
    plt.title("Convergence rate for all decay rates")
    plt.xlabel("Iterations")
    plt.ylabel("Norm")
    plt.legend(["0.9", "0.95", "0.99", "1"])
    plt.grid()
    plt.show()

    print(f"minimum norm for L1 is: {np.min([np.min(convergence[i]) for i in range(4)])}")
    # Plot descent path for gamma=0.95
    plotly.offline.plot(plot_descent_path(L1, np.array(decay)))
    plotly.offline.plot(plot_descent_path(L2, np.array(decay)))


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion
    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset
    train_portion: float, default=0.8
        Portion of dataset to use as a training set
    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set
    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples
    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set
    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heart Disease dataset
    X_train, y_train, X_test, y_test = load_data()
    X_train, y_train, X_test, y_test = X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy(), y_test.to_numpy()
    # Plotting convergence rate of logistic regression over SA heart disease data
    custom_lst = [custom[0], custom[-1]]
    lg = LogisticRegression()
    lg.fit(X_train, y_train)
    false, true, thresholds = roc_curve(y_train, lg.predict_proba(np.c_[np.ones(len(X_train)), X_train]))
    go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash='dash'), name="Random Class Assignment"),
              go.Scatter(x=false, y=true, mode='markers+lines', text=thresholds, name="", showlegend=False, marker_size=5,
                         marker_color=custom_lst[1][1], hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(title=rf"$\text{{ROC Curve Of Fitted Model - Logistic Regression}}={auc(false, true):.6f}$",
                         xaxis=dict(title=r"$\text{False Positive Rate }$"), yaxis=dict(title=r"$\text{True Positive Rate }$"))).show()
    lg.alpha_ = round(thresholds[np.argmax(true - false)], 2)
    loss = lg.loss(X_test, y_test)
    print(f"a star = {lg.alpha_} \n model test error: {loss}")

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    lam = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    L1_train_err, L1_val_err, L2_train_err, L2_val_err = [], [], [], []
    #L1
    L1_lg = LogisticRegression(penalty="l1", alpha=0.5)
    for i in lam:
        L1_lg.lam_ = i
        t_L1, v_L1 = cross_validate(L1_lg, X_train, y_train, misclassification_error)
        L1_train_err.append(t_L1)
        L1_val_err.append(v_L1)
    L1_lg.lam_ = lam[np.argmin(L1_val_err)]
    L1_lg.fit(X_train, y_train)
    test_error = L1_lg.loss(X_test, y_test)
    print(f"L1 with lambda: {L1_lg.lam_}")
    print(f"test error: {test_error}\n")

    # L2
    lg_L2 = LogisticRegression(penalty="l2", alpha=0.5)
    for i in lam:
        lg_L2.lam_ = i
        t_L2, v_L2 = cross_validate(lg_L2, X_train, y_train, misclassification_error)
        L2_train_err.append(t_L2)
        L2_val_err.append(v_L2)
    lg_L2.lam_ = lam[np.argmin(L2_val_err)]
    test_error = lg_L2.loss(X_test, y_test)
    print(f"L2 with lambda: {lg_L2.lam_}")
    print(f"test error: {test_error}\n")


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    compare_exponential_decay_rates()
    fit_logistic_regression()