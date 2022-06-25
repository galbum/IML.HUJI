import numpy as np
from typing import Tuple

from IMLearn.learners.classifiers.decision_stump import DecisionStump
from IMLearn.metalearners.adaboost import AdaBoost
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IMLearn.metrics import accuracy


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size
    Parameters
    ----------
    n: int
        Number of samples to generate
    noise_ratio: float
        Ratio of labels to invert
    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples
    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)
    ab = AdaBoost(DecisionStump, n_learners).fit(train_X, train_y)
    ab.fit(train_X, train_y)

    # Question 1: Train and test errors of AdaBoost
    train, test = [], []
    learners = np.arange(start=1, stop=n_learners + 1)
    for i in learners:
        train.append(ab.partial_loss(train_X, train_y, i))
        test.append(ab.partial_loss(test_X, test_y, i))
    fig = go.Figure([
        go.Scatter(x=learners, y=train, mode='markers + lines', name=r'Train'),
        go.Scatter(x=learners, y=test, mode='markers + lines', name=r'Test')])
    fig.update_layout(
        title=f"Question 1:"
              f"<br> AdaBoost Train and test errors " f"<br>decision tree with {noise} noise.",
        xaxis=dict(title="Learners"),
        yaxis=dict(title="loss"))
    fig.show()

    # Question 2: Plotting decision surfaces
    iteration = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    fig = make_subplots(rows=2, cols=2, subplot_titles=[f"{i} learners" for i in iteration],
                        horizontal_spacing=.015, vertical_spacing=.035, print_grid=True)
    for j, i in enumerate(iteration):
        fig.add_traces([decision_surface(
            lambda X: ab.partial_predict(X, i), lims[0], lims[1], showscale=False),
            go.Scatter(x=test_X[:, 0], y=test_X[:, 1],
                       mode="markers", showlegend=False,
                       marker=dict(color=(test_y == 1).astype(int),
                                   symbol=class_symbols[test_y.astype(int)],
                                   colorscale=[custom[0], custom[-1]],
                                   line=dict(color="black", width=0.5)))],
            rows=(j // 2) + 1, cols=(j % 2) + 1)
    fig.update_layout(
        title=f"Question 2: Decision Boundaries Of AdaBoost with decision tree"
              f"<br>with {noise} noise, as a function of learners.",
        width=780, height=780, margin=dict(t=100)
    ).update_xaxes(visible=False).update_yaxes(visible=False)
    fig.show()

    # Question 3: Decision surface of best performing ensemble
    size = np.argmin(test) + 1
    fig = go.Figure(data=[decision_surface(
        lambda X: ab.partial_predict(X, size), lims[0], lims[1], showscale=False),
        go.Scatter(x=test_X[:, 0], y=test_X[:, 1],
                   mode="markers", showlegend=False,
                   marker=dict(color=(test_y == 1).astype(int),
                               symbol=class_symbols[test_y.astype(int)],
                               colorscale=[custom[0], custom[-1]],
                               line=dict(color="black", width=0.5)))])
    fig.update_layout(
        title=f"Question 3:"
              f"<br>Decision surface of AdaBoost"
              f"<br>using decision tree with {noise} noise."
              f"<br>Ensemble size={size}, Accuracy="
              f"{accuracy(test_y, ab.partial_predict(test_X, size))}",
        width=820, height=820, margin=dict(t=100))\
        .update_xaxes(visible=False).update_yaxes(visible=False)
    fig.show()

    # Question 4: Decision surface with weighted samples
    fig = go.Figure(data=[decision_surface(
        ab.predict, lims[0], lims[1], showscale=False),
        go.Scatter(x=train_X[:, 0], y=train_X[:, 1],
                   mode="markers", showlegend=False,
                   marker=dict(color=(train_y == 1).astype(int),
                               symbol=class_symbols[train_y.astype(int)],
                               colorscale=[custom[0], custom[-1]],
                               line=dict(color="black", width=1),
                               size=(ab.D_ / np.max(ab.D_)) * 5))])
    fig.update_layout(
        title=f"Question 4:"
              f"<br>AdaBoost with weighted samples - Decision surface "
              f"<br>using decision tree with {noise} noise.",
        width=820, height=820, margin=dict(t=100))\
        .update_xaxes(visible=False).update_yaxes(visible=False)
    fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)