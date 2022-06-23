import exploratory
from sqlalchemy import create_engine
import pandas as pd

leagues_and_urls = [['eredivisie', ''],
                    ['eerste_divisie', '/group1'],
                    ['ligue_2', ''],
                    ['serie_a', ''],
                    ['championship', '/group1'],
                    ['premier_league', ''],
                    ['segunda_liga', ''],
                    ['bundesliga', ''],
                    ['ligue_1', ''],
                    ['2_liga', ''],
                    ['primeira_liga', ''],
                    ['segunda_division', ''],
                    ['serie_b', ''],
                    ['primera_division', '']]

DATABASE_TYPE = 'postgresql'
DBAPI = 'psycopg2'
ENDPOINT = 'kashin.db.elephantsql.com'
USER = 'ndpjaeig'
PASS = '6CC7jnIs9o--70M_dy3Bf1GPF8ko1MWi'
PORT = 5432
DATABASE = 'ndpjaeig'
engine = create_engine(
    f"{DATABASE_TYPE}+{DBAPI}://{USER}:{PASS}@{ENDPOINT}:{PORT}/{DATABASE}")


def download_data(db_name, columns=None):
    '''
    This is a function to read the columns of, or the entirety
    of, a SQL database.
    Attributes:
        db_name(String): The name of the database.
        engine(sqlalchemy.engine.base.Engine): Defined above,
        this is a sqlalchemy engine which allows access to the database
        columns(List): A list of columns that exist in the database. If
        None, the function downloads the entire database.
    '''
    return pd.read_sql_table(db_name, engine, columns=columns)


def join_lists(list_of_lists):
    '''
    A function to merge the items in a list of sublists into a list
    containing the elements of all the sublists. When this is done,
    links are amended to resemble the links in the amended csv files.
    Attributes:
        list_of_lists(List). A list of lists.
    '''
    mylist = [item for sublist in list_of_lists for item in sublist]
    mylist = [exploratory.modify_link(link[0]) for link in mylist]
    return mylist


league_names = [x['Name'] for x in exploratory.leagues]
cleaner = exploratory.DataCleaner(league_names, exploratory.years)
scraper = exploratory.WebScraper(exploratory.leagues)

db_links = download_data('football', columns=['Link'])
print('Downloaded sql database')

new_db_links = scraper.scrape_all_leagues(leagues_and_urls, '2022')
new_db_links = join_lists(new_db_links)

if set(new_db_links).issubset(set(db_links['Link'])):
    print('Database is up to date')
else:
    print('Database out of date')
    for league in leagues_and_urls:
        print(f'Scraping data for {league[0]} 2022')
        scraper.export_table(league[0], '2022', url_ext=league[1])
    df = cleaner.normalise_data(league_names, exploratory.years)
    df.to_csv('cleaned_dataset.csv', index=False)
    df.to_sql('football', engine, if_exists='replace')
    Latest_Season = df[df.League == '2022']
    Latest_Season.to_csv('cleaned_results.csv', index=False)
