import nfl_data_py as nfl
import pandas as pd

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
        df['indoor'] = df['roof'].apply(lambda x: 1 if x == 'dome' or 'closed' else 0)
        # if the result is positive, the home team won
        df['h_win'] = df['result'].apply(lambda x: 1 if x > 0 else 0)
        df.drop(
            columns=[
                'roof', 'gameday', 'result',
                'home_rest', 'away_rest',  # These are currently bugged or else we would keep
            ],
            inplace=True
        )
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
    df = _add_cumulative_columns(df)
    df = _add_cumulative_columns(df, offense=False)

    # If week == 1, set the days since previous game to 8 * 30 to represent the 8 months since the last regular season
    # game
    df.loc[df['week'] == 1, 'away_days_since_previous_game'] = 8 * 30
    df.loc[df['week'] == 1, 'home_days_since_previous_game'] = 8 * 30

    return df


def _add_cumulative_columns(df, offense=True):
    team_df = _create_team_df(df, offense=offense)

    if offense:
        # Calculate the number of days since the previous game for each team
        team_df['days_since_previous_game'] = team_df.groupby(['team', 'season'])['gameday'].diff().dt.days

    team_df = _calculate_cumulative_avg_score(team_df, offense=offense)

    df.reset_index(drop=True, inplace=True)
    df = _merge_team_df(df, team_df, 'home', offense=offense)
    return _merge_team_df(df, team_df, 'away', offense=offense)


def _create_team_df(df, offense=True):
    if offense:
        scores = ['home_score', 'away_score']
    else:
        scores = ['away_score', 'home_score']
    home_df = df[['gameday', 'home_team', scores[0], 'season']].rename(columns={
        'home_team': 'team',
        scores[0]: 'score'
    })
    away_df = df[['gameday', 'away_team', scores[1], 'season']].rename(columns={
        'away_team': 'team',
        scores[1]: 'score'
    })
    team_df = pd.concat([home_df, away_df]).reset_index(drop=True)
    team_df = team_df.sort_values(['team', 'season', 'gameday'])
    return team_df


def _calculate_cumulative_avg_score(team_df, offense=True):
    if offense:
        points = 'score'
    else:
        points = 'points_allowed'
    team_df[f'cumulative_{points}'] = team_df.groupby(['team', 'season'])['score'].cumsum()
    team_df[f'cumulative_avg_{points}'] = team_df[f'cumulative_{points}'] / (team_df.groupby(['team', 'season']).cumcount() + 1)
    team_df[f'cumulative_avg_{points}_change'] = team_df.groupby(['team', 'season'])[f'cumulative_avg_{points}'].diff()
    team_df.loc[team_df.groupby(['team', 'season']).cumcount() == 0, f'cumulative_avg_{points}'] = team_df['score']
    team_df.loc[team_df.groupby(['team', 'season']).cumcount() == 0, f'cumulative_avg_{points}_change'] = 0
    return team_df


def _merge_team_df(df, team_df, team_type, offense=True):
    assert team_type in ['home', 'away'], "team_type must be either 'home' or 'away'"

    side = 'off' if offense else 'def'
    score = 'score' if offense else 'points_allowed'
    df = df.merge(
        team_df,
        how='left',
        left_on=['gameday', f'{team_type}_team', 'season'],
        right_on=['gameday', 'team', 'season']
    )
    df.drop(columns=['team', 'score'], inplace=True)
    cols_rename_mapping = {
        'days_since_previous_game': f'{team_type}_days_since_previous_game',
        f'cumulative_{score}': f'{team_type}_{side}_cumulative_{score}',
        f'cumulative_avg_{score}': f'{team_type}_{side}_cumulative_avg_{score}',
        f'cumulative_avg_{score}_change': f'{team_type}_{side}_cumulative_avg_{score}_change'
    }
    if not offense:
        # remove the key value pair of days since previous game since it will exist for the offense which is called 1st
        cols_rename_mapping.pop('days_since_previous_game')
    df = df.rename(columns=cols_rename_mapping)
    return df
