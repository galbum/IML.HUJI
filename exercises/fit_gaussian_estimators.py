from matplotlib import pyplot as plt

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    X = np.random.normal(10, 1, 1000)
    univariate_guassian = UnivariateGaussian()
    univariate_guassian.fit(X)
    print(f"({univariate_guassian.mu_}, {univariate_guassian.var_})")
    # Question 2 - Empirically showing sample mean is consistent
    results = []
    for i in range(10, 1010, 10):
        results.append(abs(univariate_guassian.mu_ - (np.ndarray.sum(X[0:i])) / X[0:i].size))
    plt.title("Models on increasing size samples")
    plt.xlabel("samples")
    plt.ylabel("distance estimated and real expectation")
    plt.plot(np.arange(10, 1010, 10), results, color="blue")
    plt.show()
    # Question 3 - Plotting Empirical PDF of fitted model
    plt.title("Epirical PDF function")
    plt.xlabel("samples")
    plt.ylabel("function value")
    plt.scatter(X, univariate_guassian.pdf(X), color="blue")
    plt.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    sigma = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    y = np.random.multivariate_normal(mu, sigma, 1000)
    multivariate_gaussian = MultivariateGaussian()
    multivariate_gaussian.fit(y)
    print(multivariate_gaussian.mu_)
    print(multivariate_gaussian.cov_)
    # Question 5 - Likelihood evaluation
    size = 200
    start = -10
    end = 10
    f1 = np.linspace(start, end, size)
    f3 = np.linspace(start, end, size)
    matrix = np.zeros(shape=(f1.size, f3.size))
    maximum = -np.infty
    max_index = ()
    for i in range(f1.size):
        for j in range(f3.size):
            mu = np.array([f1[i], 0, f3[j], 0])
            matrix[i][j] = MultivariateGaussian.log_likelihood(mu, sigma, y)
            if matrix[i][j] > maximum:
                maximum = matrix[i][j]
                max_index = (i, j)
    heatmap = plt.pcolor(matrix, cmap='Accent_r')
    plt.colorbar(heatmap)
    plt.title("Log liklihood for functions")
    plt.xlabel("f1")
    plt.ylabel("f3")
    plt.xticks(np.arange(0, f1.size, 9), np.rint(f1).astype(int)[::9])
    plt.yticks(np.arange(0, f3.size, 9), np.rint(f3).astype(int)[::9])
    plt.tight_layout()
    plt.show()
    # Question 6 - Maximum likelihood
    print("maximum index: ", max_index)
    print(f1[max_index[0]], f3[max_index[1]])
    print(matrix.max())


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
