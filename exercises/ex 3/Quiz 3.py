import numpy as np
from matplotlib import pyplot as plt
from exercises.classifiers_evaluation import load_dataset

from IMLearn.learners.classifiers import LDA, GaussianNaiveBayes, Perceptron

# # Quiz question 1
# X = np.array([0, 1, 2, 3, 4, 5, 6, 7])
# y = np.array([0, 0, 1, 1, 1, 1, 2, 2])
# # Quiz question 2
# X = np.array([[1, 1], [1, 2], [2, 3], [2, 4], [3, 3], [3, 4]])
# y = np.array([0, 0, 1, 1, 1, 1])
# Quiz question 3
#
# X = np.array([0, 1, 2, 3, 4, 5, 6, 7])
# y = np.array([0, 0, 1, 1, 1, 1, 2, 2])
# # # LDA
# # lda = LDA()
# # lda.fit(X, y)
# # lda_predict = lda.predict(X)
# # lda_mu = np.array(lda.mu_)
# # lda_cov = lda.cov_
# #
# # GNB
# from sklearn.naive_bayes import GaussianNB
#
# # gnb = GaussianNaiveBayes()
# gnb = GaussianNB()
# gnb.fit(X, y)
# gnb_predict = gnb.predict(X)
# print("mu: ", gnb.theta_)
# print("var: ", gnb.var_)
# print("gnb predict: ", gnb_predict)


# # Perceptron Question 4
# losses = []
# for n, f in [("Linearly Separable", "linearly_separable.npy")]:
#              # ("Linearly Inseparable", "linearly_inseparable.npy")]:
#     # Load dataset
#     X, y = load_dataset(f"C:/Users/galbu/IML.HUJI/datasets/{f}")
#     # array for keeping the loss values of each change of the perceptron algorithm
#     losses = []
#     # Fit Perceptron and record loss in each fit iteration
#     perceptron = Perceptron(callback=lambda p, X, y: losses.append(p.loss(X, y)))
#     perceptron.fit(X, y)
#     # Plot figure
#     plt.plot(losses)
#     # plot titles and labels
#     plt.title(n)
#     plt.xlabel("number of iterations")
#     plt.ylabel("Loss value")
#     plt.show()
#
# # Perceptron Question 5
# losses = []
# for n, f in [("Linearly Separable", "linearly_separable.npy")]:
#              # ("Linearly Inseparable", "linearly_inseparable.npy")]:
#     # Load dataset
#     X, y = load_dataset(f"C:/Users/galbu/IML.HUJI/datasets/{f}")
#     # array for keeping the loss values of each change of the perceptron algorithm
#     losses = []
#     # Fit Perceptron and record loss in each fit iteration
#     perceptron = Perceptron(include_intercept = False, callback=lambda p, X, y: losses.append(p.loss(X, y)))
#     perceptron.fit(X, y)
#     # Plot figure
#     plt.plot(losses)
#     # plot titles and labels
#     plt.title(n)
#     plt.xlabel("number of iterations")
#     plt.ylabel("Loss value")
#     plt.show()