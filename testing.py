from sklearn.feature_selection import VarianceThreshold
import pandas as pd
import pickle
from joblib import dump, load
# from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
import contextlib
import time
# import numba
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier, \
    GradientBoostingClassifier, GradientBoostingRegressor, \
    RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_regression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from utils import visualise_predictions

'''
class LinearRegression:
    def __init__(self, n_features: int):  # initialize parameters
        np.random.seed(10)
        self.W = np.random.randn(n_features, 1)  # randomly initialise weight
        self.b = np.random.randn(1)  # randomly initialise bias

    def __call__(self, X):
# how do we calculate output from an input in our model?
        ypred = np.dot(X, self.W) + self.b
        return ypred  # return prediction

    def update_params(self, W, b):
        self.W = W
        # set this instance's weights
        # to the new weight value passed to the function
        self.b = b  from sklearn.metrics import mean_squared_error
do the same for the bias


df = fetch_california_housing(as_frame=True)['data']

X, y = fetch_california_housing(return_X_y=True)GradientBoostingRegressor
print(X_train.shape, y_train.shape)

model = LinearRegression(n_features=8)  # instantiate our linear model
y_pred = model(X_test)  # make prediction on data
print("Predictions:\n", y_pred[:10])  # print first 10 predictions
'''


@contextlib.contextmanager
def timer(function):
    start = time.time()
    yield
    print(f"Elapsed time for {function.__name__}: {(time.time() - start)}")


def plot_predictions(y_pred, y_true, title):
    samples = len(y_pred)
    plt.figure()
    plt.title(title)
    plt.scatter(np.arange(samples), y_pred, c='r', label='predictions')
    plt.scatter(np.arange(samples), y_true, c='b',
                label='true labels', marker='x')
    plt.legend()
    plt.xlabel('Sample numbers')
    plt.ylabel('Values')
    plt.show()


# def mean_squared_error(y_pred, y_true):
# # define our criterion (loss function)
#     errors = y_pred - y_true  # calculate errors
#     squared_errors = errors ** 2  # square errors
#     return np.mean(squared_errors)


# def minimize_loss(X_train, y_train):
#     X_with_bias = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
#     optimal_w = np.matmul(
#         np.linalg.inv(np.matmul(X_with_bias.T, X_with_bias)),
#         np.matmul(X_with_bias.T, y_train),
#     )from sklearn.metrics import classification_report, confusion_matrix
#     return optimal_w[1:], optimal_w[0]


# weights, bias = minimize_loss(X_train, y_train)
# print(weights, bias)
# cost = mean_squared_error(y_pred, y_train)
# print(cost)

# model.update_params(weights, bias)
# y_pred = model(X_train)
# cost = mean_squared_error(y_pred, y_train)
# print(cost)

# sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
# X = sel.fit_transform(X)

# X = SelectKBest(chi2, k=20).fit_transform(X, y)

# X_validation, X_test, y_validation, y_test = train_test_split(
#     X_test, y_test, test_size=0.5
# )

bad_models = [  # MLPRegressor(),
    KNeighborsClassifier(n_neighbors=151),
    tree.DecisionTreeClassifier(max_depth=50),
    # AdaBoostRegressor(),
    # RandomForestRegressor()
]

models = [  # LinearRegression(),
    # Lasso(alpha=0.1),
    MLPClassifier(hidden_layer_sizes=(150, 100, 50), max_iter=1000,
                  activation='tanh', solver='adam', random_state=1,
                  learning_rate='adaptive'),
    AdaBoostClassifier(learning_rate=1.0, n_estimators=100),
    RandomForestClassifier(),
    GradientBoostingClassifier(),
    # GradientBoostingRegressor(),
    XGBClassifier(),
    # XGBRegressor()
]


def model_comparisons(models, columns):
    data = pd.read_csv('cleaned_dataset.csv')
    y = data['Result'].values
    y = group_goals(y)
    X = data.drop(['Result', 'Date_New', 'Link'], inplace=False, axis=1)
    X_sc = scale_array(X[columns])
    X_train, X_test, y_train, y_test = feature_selection(columns)
    for model in models:
        model = model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        plot_predictions(y_pred[:10], y_test[:10], model)
        print(f'Mean Squared Error:{mean_squared_error(y_test, y_pred)}')
        score = model.score(X_sc, y)
        print(f'Score: {score}')
        try:
            cm = confusion_matrix(y_test, y_pred)
            print(cm)
            print(
                f'Classification Report:  \
                    {classification_report(y_test, y_pred)}')
            plt.figure(figsize=(6, 6))
            sns.heatmap(cm, annot=True, fmt="d", linewidths=.5,
                        square=True, cmap="Blues_r")
            plt.ylabel("Actual label")
            plt.xlabel("Predicted label")
            all_sample_title = f"Accuracy Score:{score}"
            plt.title(all_sample_title, size=15)
            print(accuracy(cm))
        except ValueError:
            pass
        # print(accuracy_score(y_pred, y_test))
        print(sum(y_pred.round() == y_test)/len(y_test))
        # dump(model, f'{str(model)}.joblib')


def MLPGridSearch(X_sc):
    mlp_gs = MLPClassifier(max_iter=1000)
    parameter_space = {
        'hidden_layer_sizes': [(10, 30, 10), (20,), (150, 100, 50)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant', 'adaptive'],
    }
    clf = GridSearchCV(mlp_gs, parameter_space, n_jobs=-1, cv=5)
    clf.fit(X_sc, y)  # X is train samples and y is the corresponding labels


def scale_array(df):
    scaler = StandardScaler()
    scaler.fit(df)
    X_sc = scaler.transform(df)
    return X_sc


def group_goals(column_vec):
    enc = LabelEncoder()
    label_encoder = enc.fit(column_vec)
    column_vec = label_encoder.transform(column_vec)
    return column_vec


important_columns = ['Season', 'Home_Team_Goals_For_This_Far',
                     'Home_Team_Goals_Against_This_Far',
                     'Away_Team_Goals_For_This_Far',
                     'Away_Team_Goals_Against_This_Far',
                     'Home_Team_Points',
                     'Away_Team_Points', 'Elo_home', 'Elo_away', 'Capacity',
                     'Home_Team_Yellows_This_Far',
                     'Away_Team_Yellows_This_Far']

data = pd.read_csv('cleaned_dataset.csv')
y = data['Result'].values
y = group_goals(y)
X = data.drop(['Result', 'Date_New', 'Link'], inplace=False, axis=1)
X_sc = scale_array(X[important_columns])

important_features = [0, 3, 4, 5, 6, 7, 8, 15, 16, 17, 20, 22]


def feature_selection(features):
    data = pd.read_csv('cleaned_dataset.csv')
    y = data['Result'].values
    X = data.drop(['Result', 'Date_New', 'Link'], inplace=False, axis=1)
    X = X[features].values
    X_sc = scale_array(X)
    y = group_goals(y)
    X_train, X_test, y_train, y_test = train_test_split(X_sc, y, test_size=0.2)
    return X_train, X_test, y_train, y_test


def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements


def feature_select_RF():
    model = RandomForestClassifier()
    data = pd.read_csv('cleaned_dataset.csv')
    y = data['Result'].values
    X = data.drop(['Result', 'Date_New', 'Link'], inplace=False, axis=1)
    X_train, X_test, y_train, y_test = feature_selection(important_columns)
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    plot_predictions(y_pred[:10], y_test[:10], model)
    print(f'Mean Squared Error:{mean_squared_error(y_test, y_pred)}')
    score = model.score(scale_array(X[important_columns]), group_goals(y))
    print(f'Score: {score}')
    try:
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        print(
            f'Classification Report:  \
                {classification_report(y_test, y_pred)}')
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt="d", linewidths=.5,
                    square=True, cmap="Blues_r")
        plt.ylabel("Actual label")
        plt.xlabel("Predicted label")
        all_sample_title = f"Accuracy Score:{score}"
        plt.title(all_sample_title, size=15)
    except ValueError:
        pass
    # print(accuracy_score(y_pred, y_test))
    print(sum(y_pred.round() == y_test)/len(y_test))


if __name__ == '__main__':
    # model_comparisons(models)
    pass
