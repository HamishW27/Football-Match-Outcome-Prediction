import exploratory
from sqlalchemy import create_engine
import pandas as pd


def download_data(db_name):
    return pd.read_sql_table(db_name, engine)


DATABASE_TYPE = 'postgresql'
DBAPI = 'psycopg2'
ENDPOINT = 'kashin.db.elephantsql.com'  # Change it for your AWS endpoint
USER = 'ndpjaeig'
PASSWORD = '6CC7jnIs9o--70M_dy3Bf1GPF8ko1MWi'
PORT = 5432
DATABASE = 'ndpjaeig'
engine = create_engine(
    f"{DATABASE_TYPE}+{DBAPI}:\
        //{USER}:{PASSWORD}@{ENDPOINT}:{PORT}/{DATABASE}")

league_names = [x['Name'] for x in exploratory.leagues]
df = exploratory.normalise_data(league_names, exploratory.years)

old_df = download_data('football', index=False)

if len(old_df) != len(df):
    df.to_csv('cleaned_dataset.csv', index=False)
    df.to_sql('football', engine, if_exists='replace')
else:
    pass
