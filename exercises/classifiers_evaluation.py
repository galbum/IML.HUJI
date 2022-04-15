from matplotlib import pyplot as plt

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
import numpy as np
from typing import Tuple
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from math import atan2, pi

from IMLearn.metrics import accuracy

pio.templates.default = "simple_white"


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    losses = []
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset(f"../datasets/{f}")
        # array for keeping the loss values of each change of the perceptron algorithm
        losses = []
        # Fit Perceptron and record loss in each fit iteration
        perceptron = Perceptron(callback=lambda p, X, y: losses.append(p.loss(X, y)))
        perceptron.fit(X, y)
        # Plot figure
        plt.plot(losses)
        # plot titles and labels
        plt.title(n)
        plt.xlabel("number of iterations")
        plt.ylabel("Loss value")
        plt.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix
    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse
    cov: ndarray of shape (2,2)
        Covariance of Gaussian
    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for n, f in (["gaussian1", "gaussian1.npy"], ["gaussian2", "gaussian2.npy"]):
        # Load dataset
        X, y = load_dataset(f"../datasets/{f}")
        # Fit models and predict over training set
        # LDA
        lda = LDA()
        lda.fit(X, y)
        lda_predict = lda.predict(X)
        lda_mu = np.array(lda.mu_)
        lda_cov = lda.cov_
        # GNB
        gnb = GaussianNaiveBayes()
        gnb.fit(X, y)
        gnb_predict = gnb.predict(X)
        gnb_mu = np.array(gnb.mu_)
        gnb_vars = np.array(gnb.vars_)
        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        symbols = np.array(["cross", "circle"])
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=(
                                f'Gaussian Naive Bayes predictions'
                                f'<br>Accuracy = {accuracy(y, gnb_predict)}',
                                f'LDA predictions'
                                f'<br>Accuracy = {accuracy(y, lda_predict)}'),
                            horizontal_spacing=0.05)
        fig.update_layout(title=n)

        # Add traces for data-points setting symbols and colors
        for i, y_pred in enumerate((gnb_predict, lda_predict)):
            comp = [1 if y_pred[j] == y[j] else 0 for j in range(len(y_pred))]
            fig.add_traces([go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
                                       marker=dict(color=y_pred,
                                                   symbol=symbols[comp],
                                                   line=dict(color="black", width=1)))],
                           rows=(i // 2) + 1, cols=(i % 2) + 1)

        # Add ellipses
        fig.add_traces([
            get_ellipse(gnb_mu[:, 0], gnb_vars),
            get_ellipse(gnb_mu[:, 1], gnb_vars),
            get_ellipse(gnb_mu[:, 2], gnb_vars)
        ], rows=1, cols=1)
        fig.add_traces([
            get_ellipse(lda_mu[:, 0], lda_cov),
            get_ellipse(lda_mu[:, 1], lda_cov),
            get_ellipse(lda_mu[:, 2], lda_cov)
        ], rows=1, cols=2)
        # Add `X`
        fig.add_trace(go.Scatter(x=gnb_mu[0, :],
                                 y=gnb_mu[1, :], mode="markers",
                                 showlegend=False,
                                 marker=dict(color="black", symbol="x",
                                             line=dict(color="black", width=2),
                                             size=20)), row=1, col=1)
        fig.add_trace(go.Scatter(x=lda_mu[0, :],
                                 y=lda_mu[1, :], mode="markers",
                                 showlegend=False,
                                 marker=dict(color="black", symbol="x",
                                             line=dict(color="black", width=2),
                                             size=20)), row=1, col=2)
        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    # run_perceptron()
    compare_gaussian_classifiers()
