import pandas as pd
import plotly.express as px
import os
from datetime import datetime
import re

'''
{'Name': 'eerste_divisie', 'Teams': 20}
{'Name': 'segunda_liga', 'Teams': 20}
Eerste divisie and segunda_liga discarded for lack of data
'''
leagues = [{'Name': 'eredivisie', 'Teams': 20},
           {'Name': 'ligue_2', 'Teams': 20},
           {'Name': 'serie_a', 'Teams': 20},
           {'Name': 'championship', 'Teams': 24},
           {'Name': 'premier_league', 'Teams': 20},
           {'Name': 'bundesliga', 'Teams': 18},
           {'Name': 'ligue_1', 'Teams': 20},
           {'Name': '2_liga', 'Teams': 18},
           {'Name': 'primeira_liga', 'Teams': 18},
           {'Name': 'segunda_division', 'Teams': 22},
           {'Name': 'serie_b', 'Teams': 20},
           {'Name': 'primera_division', 'Teams': 20}]

years = list(range(1990, 2021))


def read_data(league_name):
    tables = []
    for table in sorted(os.listdir('data/' + league_name)):
        tables.append(
            {"Year": table[:12], "Table":
             pd.read_csv('data/' + league_name + '/' + table)})
    return tables


def find_data(league_name, table_name):
    table = next(item for item in league_name if item["Year"] == table_name)
    return table['Table']


def teams_in_league(league_name):
    teams = next(item for item in leagues if item['Name'] == league_name)
    return teams['Teams']


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


def update_link(string):
    return 'https://www.besoccer.com' + string


def change_date(string):
    string = re.sub(', ..:..', '', string)
    return datetime.strptime(string, '%A, %d %B %Y')


def clean_match_info(match_info_csv):
    match_info = pd.read_csv(match_info_csv)
    Link = [update_link(match_info['Link'][x])
            for x in range(len(match_info))]
    Date = [change_date(match_info['Date_New'][x])
            for x in range(len(match_info))]
    Home_Team = [match_info['Link'][x].split(
        '/')[2] for x in range(len(match_info))]
    Away_Team = [match_info['Link'][x].split(
        '/')[3] for x in range(len(match_info))]
    Year = [int(match_info['Link'][x].split(
        '/')[4]) for x in range(len(match_info))]
    Referee = [clean_ref(match_info['Referee'][x])
               for x in range(len(match_info))]
    match_info['Date_New'] = Date
    match_info['Link'] = Link
    match_info['Home_Team'] = Home_Team
    match_info['Away_Team'] = Away_Team
    match_info['Year'] = Year
    match_info['Referee'] = Referee
    return match_info


def clean_data(league, year):
    league_to_read = league

    league_table = read_data(league)
    df = find_data(league_table, 'Results_' + str(year))

    league_teams = teams_in_league(league_to_read)

    Home_Team_Goals = [int(df['Result'][x].split(
        '-')[0]) for x in range(len(df['Result']))]

    Away_Team_Goals = [int(df['Result'][x].split(
        '-')[1]) for x in range(len(df['Result']))]

    df['Teams_in_League'] = league_teams
    df['Home_Team_Goals'] = Home_Team_Goals
    df['Away_Team_Goals'] = Away_Team_Goals
    df['Result'], df['Winners'], df['Losers'] = find_winners(df)
    return df


def histogram(league, year):
    df = clean_data(league, year)
    fig = px.histogram(df, "Result",
                       title="Wins by Home and Away Teams {} {}".format(
                           str(df.League[0]), str(df.Season[0])),
                       text_auto=True, histnorm='percent')
    fig.update_xaxes(categoryorder='category descending')
    fig.show()


def bar_graph(league, year):
    df = clean_data(league, year)
    fig = px.bar(df.loc[:, ['Winners', 'Losers']], barmode='group',
                 title='Games Won and Lost by Team')
    fig.update_xaxes(categoryorder='category ascending')
    fig.show()


def find_wins_losses_draws(df, team):
    games_played = sum(df['Home_Team'] == team) + \
        sum(df['Away_Team'] == team)
    games_won = sum(df['Winners'] == team)
    games_lost = sum(df['Losers'] == team)
    games_drawn = games_played - games_won - games_lost
    return games_won, games_lost, games_drawn


def find_most_wins(df):
    return df['Winners'].value_counts().to_frame().iloc[0].name


def find_win_percentage(wins, losses, draws):
    return wins / (wins + losses + draws)


def find_top_win_percentage(df):
    winner = find_most_wins(df)
    wld = find_wins_losses_draws(df, winner)
    wp = find_win_percentage(wld[0], wld[1], wld[2])
    return wp


def win_percentage_over_time(league, years):
    list_of_wins = []
    for year in years:
        df = clean_data(league, year)
        if df.empty:
            pass
        else:
            list_of_wins.append(
                {'Year': df['Season'][0], 'WP': find_top_win_percentage(df)})
    df_wp = pd.DataFrame(list_of_wins)
    return df_wp


def calculate_league_table(league, year):
    df = clean_data(league, year)
    teams = df['Home_Team'].drop_duplicates()
    league_table = []
    for team in teams:
        wld = find_wins_losses_draws(df, team)
        points = 3 * wld[0] + wld[2]
        league_table.append({'Team': team, 'Points': points})
    return pd.DataFrame(league_table).sort_values('Points', ascending=False)


def find_wins_losses_draws_by_round(df, team, round):
    df = df[df['Round'] < round]
    if df.empty:
        return 0, 0, 0
    else:
        games_played = sum(df['Home_Team'] == team) + \
            sum(df['Away_Team'] == team)
        games_won = sum(df['Winners'] == team)
        games_lost = sum(df['Losers'] == team)
        games_drawn = games_played - games_won - games_lost
        return games_won, games_lost, games_drawn


def calculate_league_table_by_round(league, year, round):
    df = clean_data(league, year)
    teams = df['Home_Team'].drop_duplicates()
    league_table = []
    for team in teams:
        wld = find_wins_losses_draws_by_round(df, team, round)
        points = 3 * wld[0] + wld[2]
        league_table.append({'Team': team, 'Points': points})
    return pd.DataFrame(league_table).sort_values('Points', ascending=False)


def wp_graph(league, years):
    df_wp = win_percentage_over_time(league, years)
    fig = px.bar(df_wp, x='Year', y='WP',
                 title='Top teams win percentage over time - {}'.format(
                     league), text_auto=True)
    fig.update_xaxes(categoryorder='category descending')
    fig.show()


def add_elo(league, years):
    df = clean_data(league, years)
    elo = pd.read_pickle('elo_dict.pkl')
    elo = pd.DataFrame.from_dict(elo, orient='index')
    elo.index.name = 'Link'
    return pd.merge(df, elo, on='Link')


def add_gf_thus_far(league, years):
    df = add_elo(league, years)
    teams = df['Home_Team'].drop_duplicates()
    df.insert(12, 'Home_Team_Goals_For_This_Far', [None] * len(df))
    df.insert(13, 'Away_Team_Goals_For_This_Far', [None] * len(df))
    for team in teams:
        goals = [0]
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
        for location, goal_tally in zip(mini_df.index.values, goals[:-1]):
            if df.loc[int(location)]['Home_Team'] == team:
                df.at[int(location), 'Home_Team_Goals_For_This_Far'
                      ] = goal_tally
            else:
                df.at[int(location), 'Away_Team_Goals_For_This_Far'
                      ] = goal_tally
    return df


def add_ga_thus_far(league, years):
    df = add_gf_thus_far(league, years)
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
        for location, goal_tally in zip(mini_df.index.values, goals[:-1]):
            if df.loc[int(location)]['Home_Team'] == team:
                df.at[int(location),
                      'Home_Team_Goals_Against_This_Far'] = goal_tally
            else:
                df.at[int(location),
                      'Away_Team_Goals_Against_This_Far'] = goal_tally
    return df


def add_u_streak(league, years):
    df = add_ga_thus_far(league, years)
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
        for location, streak_tally in zip(mini_df.index.values, streak[:-1]):
            if df.loc[int(location)]['Home_Team'] == team:
                df.at[int(location),
                      'Home_Team_Unbeaten_Streak'] = streak_tally
            else:
                df.at[int(location),
                      'Away_Team_Unbeaten_Streak'] = streak_tally
    return df


def add_winning_streak(league, years):
    df = add_u_streak(league, years)
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
        for location, streak_tally in zip(mini_df.index.values, streak[:-1]):
            if df.loc[int(location)]['Home_Team'] == team:
                df.at[int(location),
                      'Home_Team_Winning_Streak'] = streak_tally
            else:
                df.at[int(location),
                      'Away_Team_Winning_Streak'] = streak_tally
    return df


def add_losing_streak(league, years):
    df = add_winning_streak(league, years)
    teams = df['Home_Team'].drop_duplicates()
    df.insert(16, 'Home_Team_Losing_Streak', [None] * len(df))
    df.insert(17, 'Away_Team_Losing_Streak', [None] * len(df))
    for team in teams:
        streak = [0]
        team_games = (df['Home_Team'] == team) | (df['Away_Team'] == team)
        mini_df = df[team_games]
        for row, value in mini_df.iterrows():
            if value['Losers'] == team:
                streak.append(streak[-1] + 1)
            else:
                streak.append(0)
        for location, streak_tally in zip(mini_df.index.values, streak[:-1]):
            if df.loc[int(location)]['Home_Team'] == team:
                df.at[int(location),
                      'Home_Team_Losing_Streak'] = streak_tally
            else:
                df.at[int(location),
                      'Away_Team_Losing_Streak'] = streak_tally
    return df


def add_points(league, years):
    df = add_losing_streak(league, years)
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
        for location, points_tally in zip(mini_df.index.values, streak[:-1]):
            if df.loc[int(location)]['Home_Team'] == team:
                df.at[int(location),
                      'Home_Team_Points'] = points_tally
            else:
                df.at[int(location),
                      'Away_Team_Points'] = points_tally
    return df


def merge_data(leagues, years):
    team_info = pd.read_csv('Team_Info.csv')
    big_df = pd.DataFrame()
    for league in leagues:
        for year in years:
            df = add_points(league, year)
            big_df = pd.concat([big_df, df])
    big_df = pd.merge(big_df, team_info, on='Home_Team')
    return big_df


def normalise_data(leagues, years):
    df = merge_data(leagues, years)
    new_df = df.fillna(0)
    new_df.replace('Draw', 0, inplace=True)
    new_df.replace('Home_Team_Win', 1, inplace=True)
    new_df.replace('Away_Team_Win', -1, inplace=True)
    new_df = new_df[['Result', 'Season' 'Round', 'Teams_in_League',
                     'Home_Team_Goals_For_This_Far',
                     'Home_Team_Goals_Against_This_Far',
                     'Away_Team_Goals_For_This_Far',
                     'Away_Team_Goals_Against_This_Far',
                     'Home_Team_Points', 'Away_Team_Points',
                     'Home_Team_Losing_Streak', 'Away_Team_Losing_Streak',
                     'Home_Team_Winning_Streak',
                     'Away_Team_Winning_Streak',
                     'Home_Team_Unbeaten_Streak',
                     'Away_Team_Unbeaten_Streak',
                     'Elo_home', 'Elo_away', 'Capacity']]
    return new_df


# match_info = pd.read_csv('Match_Info.csv')
# match_info = clean_match_info(match_info)
if __name__ == '__main__':
    '''
    histogram('premier_league', 2003)
    bar_graph('premier_league', 2003)
    for league in leagues:
        wp_graph(league['Name'], years)
    '''
    league_names = [x['Name'] for x in leagues]
    x = normalise_data(league_names, years)
    x.to_csv('cleaned_dataset.csv', index=False)
