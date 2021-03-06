import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


# class LinearRegression:
#     def __init__(self, n_features: int):  # initialize parameters
#         np.random.seed(10)
#         self.W = np.random.randn(n_features, 1)  # randomly initialise weight
#         self.b = np.random.randn(1)  # randomly initialise bias

#     def __call__(self, X):  # how do we calculate output from an input in our model?
#         ypred = np.dot(X, self.W) + self.b
#         return ypred  # return prediction

#     def update_params(self, W, b):
#         self.W = W  # set this instance's weights to the new weight value passed to the function
#         self.b = b  from sklearn.metrics import mean_squared_error
# do the same for the bias


df = fetch_california_housing(as_frame=True)['data']

X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

X_validation, X_test, y_validation, y_test = train_test_split(
    X_test, y_test, test_size=0.5
)

# print(X_train.shape, y_train.shape)

# model = LinearRegression(n_features=8)  # instantiate our linear model
# y_pred = model(X_test)  # make prediction on data
# print("Predictions:\n", y_pred[:10])  # print first 10 predictions


def plot_predictions(y_pred, y_true):
    samples = len(y_pred)
    plt.figure()
    plt.scatter(np.arange(samples), y_pred, c='r', label='predictions')
    plt.scatter(np.arange(samples), y_true, c='b',
                label='true labels', marker='x')
    plt.legend()
    plt.xlabel('Sample numbers')
    plt.ylabel('Values')
    plt.show()


# def mean_squared_error(y_pred, y_true):  # define our criterion (loss function)
#     errors = y_pred - y_true  # calculate errors
#     squared_errors = errors ** 2  # square errors
#     return np.mean(squared_errors)


# def minimize_loss(X_train, y_train):
#     X_with_bias = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
#     optimal_w = np.matmul(
#         np.linalg.inv(np.matmul(X_with_bias.T, X_with_bias)),
#         np.matmul(X_with_bias.T, y_train),
#     )
#     return optimal_w[1:], optimal_w[0]


# weights, bias = minimize_loss(X_train, y_train)
# print(weights, bias)
# cost = mean_squared_error(y_pred, y_train)
# print(cost)

# model.update_params(weights, bias)
# y_pred = model(X_train)
# cost = mean_squared_error(y_pred, y_train)
# print(cost)

data = pd.read_csv('cleaned_dataset.csv')
y = data['Result']
X = data.drop('Result', inplace=False, axis=1)

model = LinearRegression()

model.fit(X, y)

y_pred = model.predict(X)

plot_predictions(y_pred[:10], y_test[:10])

mean_squared_error(y, y_pred)
