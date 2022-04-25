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

In my report, I've described relationships which may affect a team's likelihood of winning. These include whether a team is playing at home or away, their recently performance as 

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