# Football-Match-Outcome-Prediction

## Milestone 1

An EDA has been performed on the dataset and has found trends including an advantage to the home side as well as a tendency to approximate the previous seasons results. To do this the data had to be cleaned with basic string manipulation to create new, more useful, data columns.

```python
def add_elo(league, years):
    df = clean_data(league, years)
    elo = pd.read_pickle('elo_dict.pkl')
    elo = pd.DataFrame.from_dict(elo, orient='index')
    elo.index.name = 'Link'
    return pd.merge(df, elo, on='Link')
```

In my report, I've described relationships which may affect a team's likelihood of winning. These include whether a team is playing at home or away, their recent performance as recorded by their winning and unbeaten streaks.

## Milestone 2

At this stage, I'd extracted the elo of each team and added it to the dataframe as well as specific information about each team. I then remove non-numerical information so as to prepare the data for machine learning.

```python
def normalise_data(leagues, years):
    df = merge_data(leagues, years)
    new_df = df.fillna(0)
    new_df = new_df.iloc[:, [4, 5, 7, 8, 9, 12, 13,
                             14, 15, 16, 17, 18, 19, 20, 21, 22, 23]]
    return new_df
```

## Milestone 3

By this point, it had become necessary to set up a database to store and retrieve the amended dataframe. I'd used SQLAlchemy to upload the data to an RDS instance setup through AWS.

```python
def download_data(db_name):
    return pd.read_sql_table(db_name, engine)
```
The function above would then download the dataframe which is kept updated by another function.

## Milestone 4

At this stage, it was important to find a model to classify the data. After trialing around 20 different models and tuning them using GridSearchCV, the most promising were RandomForestClassifier, XGBClassifier, and AdaBoostClassifier. RandomForest had the best results after performing feature selection and removing outdated football matches from the data.

Unsurprisingly, the Classifiers were much better at predicting the match outcomes than Regressors.

``` python
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def scale_array(df):
    scaler = StandardScaler()
    scaler.fit(df)
    X_sc = scaler.transform(df)
    return X_sc

def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements

y = data['Result'].values
X = data.drop(['Result', 'Date_New', 'Link'], inplace=False, axis=1)
X.League = X.League.astype('category').cat.codes
X_sc = scale_array(X[svm_cols])
X_train, X_test, y_train, y_test = train_test_split(X_sc, y, test_size=0.1)
model = RandomForestClassifier(
        criterion='entropy', max_depth=128,
        max_features='log2', n_estimators=1024)
model = model.fit(X_train, y_train)
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy(cm))
```

## Milestone 5

Now that the model has been tested and optimised, it can be tested on new data such as the 2022 season. This produced promising results with a model score of 47.8%. This is better than the initial models though indicates that matches are either inherently unpredictable or, as I suspect, rely heavily on data not so easily available. For improved results, the model will probably require information on individual players and a team's team sheet on a match's game day. This variation in the quality of a team's players is why I suspect player form is a key part of the model though is inherently more difficult to add this model. To do so would require transfer histories, injury records, and historical team sheets to be added to the dataframe which, while possible, would be very time consuming.

``` python
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix

def scale_array(df):
    scaler = MinMaxScaler()
    scaler.fit(df)
    X_sc = scaler.transform(df)
    return X_sc

def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements

X_test = scale_array(read_2022_data(svm_cols))
y_real = get_2022_results()
y_pred = model.predict(X_test)

cm = confusion_matrix(y_pred, y_real,labels=[2,1,0])
print(cm)
print(accuracy(cm))
```