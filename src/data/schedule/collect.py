import nfl_data_py as nfl
import pandas as pd

# TODO: outdoors
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
        df = add_calculated_values(df)
        if schedule_df is None:
            schedule_df = df
        else:
            schedule_df = pd.concat([schedule_df, df], ignore_index=True)

    return schedule_df


def add_calculated_values(df):
    """
    Add the following calculated values to the data frame:
        days since the previous game
        cumulative average score

    The cumulative average score only accounts for the games played in a single season.

    :param df: The dataframe to add calculated values to
    :return: The dataframe with calculated values added
    """
    df = df.sort_values(by=['season', 'week', 'gameday'])

    # Create a new dataframe that contains both the home and away games for each team
    home_df = df[['gameday', 'home_team', 'home_score', 'season']].rename(columns={
        'home_team': 'team',
        'home_score': 'score'
    })
    away_df = df[['gameday', 'away_team', 'away_score', 'season']].rename(columns={
        'away_team': 'team',
        'away_score': 'score'
    })
    team_df = pd.concat([home_df, away_df]).reset_index(drop=True)

    # Sort this new dataframe by 'team' and 'gameday'
    team_df = team_df.sort_values(['team', 'season', 'gameday'])

    # Calculate the number of days since the previous game for each team
    team_df['days_since_previous_game'] = team_df.groupby(['team', 'season'])['gameday'].diff().dt.days

    # Calculate the cumulative average score for each team, for example, the average score for the first game, the
    # average score for the first two games, the average score for the first three games, etc.
    team_df['cumulative_score'] = team_df.groupby(['team', 'season'])['score'].cumsum()
    team_df['cumulative_avg_score'] = team_df['cumulative_score'] / (team_df.groupby(['team', 'season']).cumcount() + 1)
    team_df['cumulative_avg_score_change'] = team_df.groupby(['team', 'season'])['cumulative_avg_score'].diff()
    # Replace the first game's cumulative average score with the first game's score
    team_df.loc[team_df.groupby(['team', 'season']).cumcount() == 0, 'cumulative_avg_score'] = team_df['score']

    df.reset_index(drop=True, inplace=True)
    # Merge this new dataframe back into the original dataframe
    df = df.merge(
        team_df,
        how='left',
        left_on=['gameday', 'home_team', 'season'],
        right_on=['gameday', 'team', 'season']
    )
    df = df.rename(columns={
        'days_since_previous_game': 'home_days_since_previous_game',
        'cumulative_score': 'home_cumulative_score',
        'cumulative_avg_score': 'home_cumulative_avg_score',
        'cumulative_avg_score_change': 'home_cumulative_avg_score_change'
    })
    df = df.merge(
        team_df,
        how='left',
        left_on=['gameday', 'away_team', 'season'],
        right_on=['gameday', 'team', 'season']
    )
    df = df.rename(columns={
        'days_since_previous_game': 'away_days_since_previous_game',
        'cumulative_score': 'away_cumulative_score',
        'cumulative_avg_score': 'away_cumulative_avg_score',
        'cumulative_avg_score_change': 'away_cumulative_avg_score_change'
    })

    # If week == 1, set the days since previous game to 8 * 30 to represent the 8 months since the last regular season
    # game
    df.loc[df['week'] == 1, 'away_days_since_previous_game'] = 8 * 30
    df.loc[df['week'] == 1, 'home_days_since_previous_game'] = 8 * 30

    return df

