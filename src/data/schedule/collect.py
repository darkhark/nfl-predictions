import nfl_data_py as nfl
import pandas as pd

# TODO: Calculate days since previous game, cumulative average score, outdoors
# 'surface': ~12% null so not including for now
# 'temp' and 'wind': ~45% null so not including for now
COLS_TO_KEEP = [
    'game_id', 'season', 'game_type', 'week', 'gameday', 'away_team', 'away_score', 'home_team', 'home_score', 'result',
    'away_rest', 'home_rest', 'div_game', 'roof'
]


ODDS_COLS = [
    'away_moneyline', 'home_moneyline', 'spread_line', 'away_spread_odds', 'home_spread_odds', 'total_line',
    'under_odds', 'over_odds'
]


def get_schedule_data(years, keep_game_id=True, keep_odds=False):
    """
    Collects schedule data for the specified year(s).

    :param years: list of years to collect data for or a single year
    :param keep_game_id: boolean to keep the game_id column. Helpful for joining with other data
    :param keep_odds: boolean to keep the vegas type odds and money lines columns. May be useful like an ensemble model
    :return:
    """
    if isinstance(years, int):
        years = [years]
    elif not isinstance(years, list):
        raise TypeError('years must be a list or an integer')

    cols_to_import = COLS_TO_KEEP
    if not keep_game_id:
        cols_to_import.remove('game_id')
    if keep_odds:
        cols_to_import += ODDS_COLS

    schedule_df = None
    for year in years:
        df = nfl.import_schedules(years=[year])[cols_to_import]
        df['gameday'] = pd.to_datetime(df['gameday'], format='%Y-%m-%d')
        df = add_days_since_previous_game(df)
        if schedule_df is None:
            schedule_df = df
        else:
            schedule_df = pd.concat([schedule_df, df], ignore_index=True)

    return schedule_df


def add_days_since_previous_game(df):
    """
    Calculate the days since the previous game for the team in the row.

    :param df: The dataframe to calculate the days since the previous game for
    :return: The dataframe with the days since the previous game column added for
    both the home and away teams
    """
    df = df.sort_values(by=['season', 'week', 'gameday'])

    # Create a new dataframe that contains both the home and away games for each team
    home_df = df[['gameday', 'home_team']].rename(columns={'home_team': 'team'})
    away_df = df[['gameday', 'away_team']].rename(columns={'away_team': 'team'})
    team_df = pd.concat([home_df, away_df])

    # Sort this new dataframe by 'team' and 'gameday'
    team_df = team_df.sort_values(['team', 'gameday'])

    # Calculate the number of days since the previous game for each team
    team_df['days_since_previous_game'] = team_df.groupby('team')['gameday'].diff().dt.days

    # Merge this new dataframe back into the original dataframe
    df = df.merge(team_df, how='left', left_on=['gameday', 'home_team'], right_on=['gameday', 'team'])
    df = df.rename(columns={'days_since_previous_game': 'home_days_since_previous_game'})
    df = df.merge(team_df, how='left', left_on=['gameday', 'away_team'], right_on=['gameday', 'team'])
    df = df.rename(columns={'days_since_previous_game': 'away_days_since_previous_game'})

    # If week == 1, set the days since previous game to 8 * 30 to represent the 8 months since the last regular season
    # game
    df.loc[df['week'] == 1, 'away_days_since_previous_game'] = 8 * 30
    df.loc[df['week'] == 1, 'home_days_since_previous_game'] = 8 * 30

    return df

