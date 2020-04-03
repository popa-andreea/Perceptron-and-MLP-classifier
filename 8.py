from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing

import numpy as np

import warnings
warnings.filterwarnings("ignore", category = UserWarning)

#exercitiul 1

def plot3d_data(X, y):
    ax = plt.axes(projection='3d')
    ax.scatter3D(X[y == -1, 0], X[y == -1, 1], X[y == -1, 2], 'b');
    ax.scatter3D(X[y == 1, 0], X[y == 1, 1], X[y == 1, 2], 'r');
    plt.show()

def plot3d_data_and_decision_function(X, y, W, b):
    ax = plt.axes(projection='3d')
    # create x,y
    xx, yy = np.meshgrid(range(10), range(10))
    # calculate corresponding z
    # [x, y, z] * [coef1, coef2, coef3] + b = 0
    zz = (-W[0] * xx - W[1] * yy - b) / W[2]
    ax.plot_surface(xx, yy, zz, alpha=0.5)
    ax.scatter3D(X[y == -1, 0], X[y == -1, 1], X[y == -1, 2], 'b');
    ax.scatter3D(X[y == 1, 0], X[y == 1, 1], X[y == 1, 2], 'r');
    plt.show()

def normalize(X_train, X_test):
    sc = preprocessing.StandardScaler()
    sc.fit(X_train)
    X_train = sc.transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test

def train_test_perceptron(X_train, y_train, X_test, y_test):
    X_train, X_test = normalize(X_train, X_test)

    perceptron = Perceptron(eta0 = 0.1, tol = 1e-5)
    perceptron.fit(X_train, y_train)
    print("accuracy on train set: ", perceptron.score(X_train, y_train))
    print("accuracy on test set:", perceptron.score(X_test, y_test))

    weights = perceptron.coef_.reshape(3,1)
    bias = perceptron.intercept_
    print("weights:\n", weights)
    print("bias: ", bias)
    print("number of epochs until convergence:", perceptron.n_iter_)

    plot3d_data_and_decision_function(X_train, y_train, weights, bias)

X = np.loadtxt("data/3d-points/x_train.txt")
y = np.loadtxt("data/3d-points/y_train.txt", "int")

X_test = np.loadtxt("data/3d-points/x_test.txt")
y_test = np.loadtxt("data/3d-points/y_test.txt", "int")

print("Exercitiul 1")
train_test_perceptron(X, y, X_test, y_test)

#exercitiul 2

X = np.loadtxt("data/mnist/train_images.txt")
y = np.loadtxt("data/mnist/train_labels.txt", "int")

X_test = np.loadtxt("data/mnist/test_images.txt")
y_test = np.loadtxt("data/mnist/test_labels.txt", "int")

X, X_test = normalize(X, X_test)

def train_test(classifier, X_train, y_train, X_test, y_test):
    classifier.fit(X_train, y_train)
    print("accuracy on train set: ", classifier.score(X_train, y_train))
    print("accuracy on test set:", classifier.score(X_test, y_test))
    print("number of epochs until convergence:", classifier.n_iter_)

print("\nExericitiul 2\na:")
mlp_clf = MLPClassifier(hidden_layer_sizes = (1,), activation = "tanh", solver = "sgd",
                        learning_rate_init = 0.01, momentum = 0)
train_test(mlp_clf, X, y, X_test, y_test)

print("b:")
mlp_clf = MLPClassifier(hidden_layer_sizes = (10,), activation = "tanh", solver = "sgd",
                        learning_rate_init = 0.01, momentum = 0)
train_test(mlp_clf, X, y, X_test, y_test)

print("c:")
mlp_clf = MLPClassifier(hidden_layer_sizes = (10,), activation = "tanh", solver = "sgd",
                        learning_rate_init = 0.00001, momentum = 0)
train_test(mlp_clf, X, y, X_test, y_test)

print("d:")
mlp_clf = MLPClassifier(hidden_layer_sizes = (10,), activation = "tanh", solver = "sgd",
                        learning_rate_init = 10, momentum = 0)
train_test(mlp_clf, X, y, X_test, y_test)

print("e:")
mlp_clf = MLPClassifier(hidden_layer_sizes = (10,), activation = "tanh", solver = "sgd",
                        learning_rate_init = 0.01, momentum = 0, max_iter = 20)
train_test(mlp_clf, X, y, X_test, y_test)