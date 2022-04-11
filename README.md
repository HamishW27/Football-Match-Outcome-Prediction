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