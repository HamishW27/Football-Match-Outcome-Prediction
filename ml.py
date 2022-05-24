from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import numpy as np

np.random.seed(7)

data = pd.read_csv('cleaned_dataset.csv')
y = data['Result']
X = data.drop(['Result', 'Date_New', 'Link'], inplace=False, axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7)
X_test, X_validation, y_test, y_validation = train_test_split(
    X_test, y_test, train_size=0.7)

print(f'Training Size: {len(y_train)}')
print(f'Validation  Size: {len(y_validation)}')
print(f'Test Size: {len(y_test)}')

models = [
    DecisionTreeRegressor(splitter="random"),
    SVR(),
    LinearRegression()]

for model in models:
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_validation_pred = model.predict(X_validation)
    y_test_pred = model.predict(X_test)

    train_loss = mean_squared_error(y_train, y_train_pred)
    validation_loss = mean_squared_error(y_validation, y_validation_pred)
    test_loss = mean_squared_error(y_test, y_test_pred)

    print(
        f"{model.__class__.__name__}: "
        f"Train Loss: {train_loss} | Validation Loss: {validation_loss} | "
        f"Test Loss: {test_loss}"
    )


def calculate_validation_loss(X_train, y_train, X_validation, y_validation):
    model = LinearRegression()

    # Without data leakage, train on train, validate on validation
    model.fit(X_train, y_train)
    y_validation_pred = model.predict(X_validation)
    validation_loss = mean_squared_error(y_validation, y_validation_pred)

    print(f"Validation loss: {validation_loss}")


# Without data leakage, train on train, validate on validation
calculate_validation_loss(X_train, y_train, X_validation, y_validation)

# With data leakage, 50 samples from validation added
fail_X_train = np.concatenate((X_train, X_validation[:50]))
fail_y_train = np.concatenate((y_train, y_validation[:50]))

calculate_validation_loss(fail_X_train, fail_y_train,
                          X_validation, y_validation)
