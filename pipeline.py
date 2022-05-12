import exploratory
from sqlalchemy import create_engine
import pandas as pd

leagues_and_urls = [['eredivisie', ''],
                    ['eerste_divisie', '/group1']
                    ['ligue_2', ''],
                    ['serie_a', ''],
                    ['championship', '/group1'],
                    ['premier_league', ''],
                    ['segunda_liga', ''],
                    ['bundesliga', ''],
                    ['ligue_1', '']
                    ['2_liga', ''],
                    ['primeira_liga', '']
                    ['segunda_division', ''],
                    ['serie_b', ''],
                    ['primera_division', '']]


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

db_df = download_data('football', index=False)
db_links = db_df['Link']

scraper = exploratory.WebScraper(exploratory.leagues)
new_db_links = scraper.scrape_all_leagues(leagues_and_urls)

if set(new_db_links).issubset(db_links):
    print('Database is up to date')
else:
    for league in leagues_and_urls:
        scraper.export_table(league[0], '2022', url_ext=league[1])
    cleaner = exploratory.DataCleaner(league_names, exploratory.years)
    df = cleaner.merge_data(league_names, exploratory.years)
    df.to_csv('cleaned_dataset.csv', index=False)
    df.to_sql('football', engine, if_exists='replace')
