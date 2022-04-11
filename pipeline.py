import pandas as pd
from sqlalchemy import create_engine

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


def format_formatted_data(df, csv):
    old_data = pd.read_csv(csv)
    new_data = df
    total_data = pd.concat(old_data, new_data)
    total_data.to_csv('cleaned_dataset', index=False)


def clean_data(csv, league_teams):
    df = pd.read_csv(csv)

    league_teams = league_teams

    Home_Team_Goals = [int(df['Result'][x].split(
        '-')[0]) for x in range(len(df['Result']))]

    Away_Team_Goals = [int(df['Result'][x].split(
        '-')[1]) for x in range(len(df['Result']))]

    df['Teams in League'] = league_teams
    df['Home_Team_Goals'] = Home_Team_Goals
    df['Away_Team_Goals'] = Away_Team_Goals
    df['Result'], df['Winners'], df['Losers'] = find_winners(df)
    return df


def find_winners(dataframe):
    Results = []
    Winners = []
    Losers = []
    for i in range(len(dataframe)):
        if dataframe['Home_Team_Goals'][i] > dataframe['Away_Team_Goals'][i]:
            Results.append('Home_Team_Win')
            Winners.append(dataframe['Home_Team'][i])
            Losers.append(dataframe['Away_Team'][i])
        elif dataframe['Home_Team_Goals'][i] == \
                dataframe['Away_Team_Goals'][i]:
            Results.append('Draw')
            Winners.append(None)
            Losers.append(None)
        else:
            Results.append('Away_Team_Win')
            Winners.append(dataframe['Away_Team'][i])
            Losers.append(dataframe['Home_Team'][i])
    return Results, Winners, Losers


def clean_ref(string):
    return string.split('/')[0].strip('\r\n').strip('Referee: ')


def clean_match_info(match_info_csv):
    Home_Team = [match_info_csv['Link'][x].split(
        '/')[2] for x in range(len(match_info_csv['Link']))]
    Away_Team = [match_info_csv['Link'][x].split(
        '/')[3] for x in range(len(match_info_csv['Link']))]
    Year = [int(match_info_csv['Link'][x].split(
        '/')[4]) for x in range(len(match_info_csv['Link']))]
    Referee = [clean_ref(match_info_csv['Referee'][x])
               for x in range(len(match_info_csv['Referee']))]
    match_info_csv['Home_Team'] = Home_Team
    match_info_csv['Away_Team'] = Away_Team
    match_info_csv['Year'] = Year
    match_info_csv['Referee'] = Referee
    return match_info_csv


def add_elo(csv, league_teams):
    df = clean_data(csv, league_teams)
    elo = pd.read_pickle('elo_dict.pkl')
    elo = pd.DataFrame.from_dict(elo, orient='index')
    elo.index.name = 'Link'
    return pd.merge(df, elo, on='Link')


def add_gf_thus_far(csv, league_teams):
    df = add_elo(csv, league_teams)
    teams = df['Home_Team'].drop_duplicates()
    df.insert(12, 'Home_Team_Goals_For_This_Far', [None] * len(df))
    df.insert(13, 'Away_Team_Goals_For_This_Far', [None] * len(df))
    for team in teams:
        goals = []
        team_games = (df['Home_Team'] == team) | (df['Away_Team'] == team)
        mini_df = df[team_games]
        for row, value in mini_df.iterrows():
            if value['Home_Team'] == team and goals:
                goals.append(goals[-1] + value['Home_Team_Goals'])
            elif value['Home_Team'] == team:
                goals.append(value['Home_Team_Goals'])
            elif value['Away_Team'] == team and goals:
                goals.append(goals[-1] + value['Away_Team_Goals'])
            else:
                goals.append(value['Away_Team_Goals'])
        for location, goal_tally in zip(mini_df.index.values, goals):
            if df.loc[int(location)]['Home_Team'] == team:
                df.at[int(location), 'Home_Team_Goals_For_This_Far'
                      ] = goal_tally
            else:
                df.at[int(location), 'Away_Team_Goals_For_This_Far'
                      ] = goal_tally
    return df


def add_ga_thus_far(csv, league_teams):
    df = add_gf_thus_far(csv, league_teams)
    teams = df['Home_Team'].drop_duplicates()
    df.insert(13, 'Home_Team_Goals_Against_This_Far', [None] * len(df))
    df.insert(15, 'Away_Team_Goals_Against_This_Far', [None] * len(df))
    for team in teams:
        goals = [0]
        team_games = (df['Home_Team'] == team) | (df['Away_Team'] == team)
        mini_df = df[team_games]
        for row, value in mini_df.iterrows():
            if value['Home_Team'] == team:
                goals.append(goals[-1] + value['Away_Team_Goals'])
            else:
                goals.append(goals[-1] + value['Home_Team_Goals'])
        for location, goal_tally in zip(mini_df.index.values, goals[1:]):
            if df.loc[int(location)]['Home_Team'] == team:
                df.at[int(location),
                      'Home_Team_Goals_Against_This_Far'] = goal_tally
            else:
                df.at[int(location),
                      'Away_Team_Goals_Against_This_Far'] = goal_tally
    return df


def add_u_streak(csv, league_teams):
    df = add_ga_thus_far(csv, league_teams)
    teams = df['Home_Team'].drop_duplicates()
    df.insert(16, 'Home_Team_Unbeaten_Streak', [None] * len(df))
    df.insert(17, 'Away_Team_Unbeaten_Streak', [None] * len(df))
    for team in teams:
        streak = [0]
        team_games = (df['Home_Team'] == team) | (df['Away_Team'] == team)
        mini_df = df[team_games]
        for row, value in mini_df.iterrows():
            if value['Winners'] in [None, team]:
                streak.append(streak[-1] + 1)
            else:
                streak.append(0)
        for location, streak_tally in zip(mini_df.index.values, streak[1:]):
            if df.loc[int(location)]['Home_Team'] == team:
                df.at[int(location),
                      'Home_Team_Unbeaten_Streak'] = streak_tally
            else:
                df.at[int(location),
                      'Away_Team_Unbeaten_Streak'] = streak_tally
    return df


def add_streak(csv, league_teams):
    df = add_u_streak(csv, league_teams)
    teams = df['Home_Team'].drop_duplicates()
    df.insert(16, 'Home_Team_Winning_Streak', [None] * len(df))
    df.insert(17, 'Away_Team_Winning_Streak', [None] * len(df))
    for team in teams:
        streak = [0]
        team_games = (df['Home_Team'] == team) | (df['Away_Team'] == team)
        mini_df = df[team_games]
        for row, value in mini_df.iterrows():
            if value['Winners'] == team:
                streak.append(streak[-1] + 1)
            else:
                streak.append(0)
        for location, streak_tally in zip(mini_df.index.values, streak[1:]):
            if df.loc[int(location)]['Home_Team'] == team:
                df.at[int(location),
                      'Home_Team_Winning_Streak'] = streak_tally
            else:
                df.at[int(location),
                      'Away_Team_Winning_Streak'] = streak_tally
    return df


def add_points(csv, league_teams):
    df = add_streak(csv, league_teams)
    teams = df['Home_Team'].drop_duplicates()
    df.insert(16, 'Home_Team_Points', [None] * len(df))
    df.insert(17, 'Away_Team_Points', [None] * len(df))
    for team in teams:
        streak = [0]
        team_games = (df['Home_Team'] == team) | (df['Away_Team'] == team)
        mini_df = df[team_games]
        for row, value in mini_df.iterrows():
            if value['Winners'] == team:
                streak.append(streak[-1] + 3)
            elif value['Winners'] is None:
                streak.append(streak[-1] + 1)
            else:
                streak.append(streak[-1])
        for location, points_tally in zip(mini_df.index.values, streak[1:]):
            if df.loc[int(location)]['Home_Team'] == team:
                df.at[int(location),
                      'Home_Team_Points'] = points_tally
            else:
                df.at[int(location),
                      'Away_Team_Points'] = points_tally
    return df


def merge_data(csv, league_teams):
    team_info = pd.read_csv('Team_Info.csv')
    big_df = pd.DataFrame()
    df = add_points(csv, league_teams)
    big_df = pd.merge(df, team_info, on='Home_Team')
    return big_df


def normalise_data(csv, league_teams):
    df = merge_data(csv, league_teams)
    new_df = df.fillna(0)
    new_df = new_df.iloc[:, [4, 5, 7, 8, 9, 12, 13,
                             14, 15, 16, 17, 18, 19, 20, 21, 22, 23]]
    return new_df


def download_data(db_name):
    return pd.read_sql_table(db_name, engine)


if __name__ == '__main__':
    '''
    histogram('premier_league', 2003)
    bar_graph('premier_league', 2003)
    for league in leagues:
        wp_graph(league['Name'], years)
    '''
    old_data = download_data('football')
    new_data = normalise_data('path/to/csv.csv', 20)
    total_data = pd.concat(old_data, new_data)
    total_data.to_csv('cleaned_dataset.csv', index=False)
    total_data.to_sql('football', engine, if_exists='replace')
