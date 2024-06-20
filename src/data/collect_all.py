import pandas as pd

from src.data.next_gen_stats import collect as next_gen_collect
from src.data.play_by_play import collect as pbp_collect
from src.data.weekly import collect as weekly_collect
from src.data.schedule import collect as schedule_collect


def get_all_data(years):
    """
    Collects all data for the specified year(s). Merges the data from the following sources:
        - Next Gen Stats
        - Play by Play
        - Weekly
        - Schedule

    :param years: list of years to collect data for or a single year
    :return:
    """
    if isinstance(years, int):
        years = [years]
    elif not isinstance(years, list):
        raise TypeError('years must be a list or an integer')


def get_schedule_and_weekly_data(years):
    """
    Collects schedule and weekly data for the specified year(s). Merges the data from the following sources:
        - Weekly
        - Schedule

    :param years: list of years to collect data for or a single year
    :return:
    """
    if isinstance(years, int):
        years = [years]
    elif not isinstance(years, list):
        raise TypeError('years must be a list or an integer')

    weekly_data = weekly_collect.get_weekly_data(years).reset_index(drop=True)
    schedule_data = schedule_collect.get_schedule_data(years).reset_index(drop=True)

    home_team_is_team = _get_home_or_away_team_in_weekly_data(schedule_data, weekly_data, is_home=True)
    away_team_is_team = _get_home_or_away_team_in_weekly_data(schedule_data, weekly_data, is_home=False)
    combined_data = pd.merge(
        home_team_is_team,
        away_team_is_team,
        on=['season', 'week', 'home_team', 'away_team'],
        suffixes=('', '_y')
    )
    # drop the columns with _y suffix
    combined_data = combined_data[
        [col for col in combined_data.columns if not col.endswith('_y')]
    ]

    # duplicate each row, but change either the home or away team to the target team
    home_is_target = _convert_team_to_target(combined_data, is_home_target=True)
    away_is_target = _convert_team_to_target(combined_data, is_home_target=False)
    combined_data = pd.concat([home_is_target, away_is_target], ignore_index=True)

    combined_data = _shift_data(combined_data)
    return combined_data


def _get_home_or_away_team_in_weekly_data(schedule_data, weekly_data, is_home=True):
    if is_home:
        team_type = 'home'
        opp_team_type = 'away'
    else:
        team_type = 'away'
        opp_team_type = 'home'
    team_df = schedule_data.merge(
        weekly_data,
        left_on=[f'{team_type}_team', 'season', 'week'],
        right_on=['team', 'season', 'week'],
        how='inner'
    )
    # replace the 'team' column with 'home_team'
    team_df.drop(columns=['team', 'opp_team'], inplace=True)
    team_df = team_df.rename(
        columns=lambda x: x.replace('opp', opp_team_type) if 'opp' in x else x
    )
    team_df = team_df.rename(
        columns=lambda x: x.replace('off', f'off_{team_type}') if x.startswith('off') else x
    )
    team_df = team_df.rename(
        columns=lambda x: x.replace('team_', f'{team_type}_') if x.startswith('team_') else x
    )
    return team_df


def _convert_team_to_target(df, is_home_target: bool):
    if is_home_target:
        team_type = 'home'
        opp_team_type = 'away'
    else:
        team_type = 'away'
        opp_team_type = 'home'
    target_df = df.copy()
    target_df = target_df.rename(
        columns=lambda x: x.replace(team_type, 'target') if team_type in x else x
    )
    target_df = target_df.rename(
        columns=lambda x: x.replace(opp_team_type, 'opp') if opp_team_type in x else x
    )
    target_df['is_home_target'] = 1 if is_home_target else 0
    return target_df


def _get_stats_columns(df):
    stats_data_cols = [col for col in df.columns if 'target' in col or 'opp' in col]
    cols_not_stats = [
        'target_days_since_previous_game', 'opp_days_since_previous_game',
        'target_game_count', 'opp_game_count', 'target_team', 'opp_team',
        'target_score', 'opp_score', 'is_home_target'
    ]
    for col in cols_not_stats:
        stats_data_cols.remove(col)
    return stats_data_cols


def _shift_data(combined_data):
    # shift the weeks for each team to get the previous week's data. For example, the stats for week 1 should be in
    # week 2, and so on
    stats_columns = _get_stats_columns(combined_data)
    teams = combined_data['target_team'].unique()
    target_stat_cols = [col for col in stats_columns if 'target' in col]
    opp_stat_cols = [col for col in stats_columns if 'opp' in col]
    for team in teams:
        target_team_df = combined_data[(combined_data['target_team'] == team)].copy()
        opp_team_df = combined_data[(combined_data['opp_team'] == team)].copy()
        target_team_df = target_team_df.sort_values(['season', 'week'], ascending=True)
        opp_team_df = opp_team_df.sort_values(['season', 'week'], ascending=True)
        for col in target_stat_cols:
            target_team_df[col] = target_team_df[col].shift(1)
        for col in opp_stat_cols:
            opp_team_df[col] = opp_team_df[col].shift(1)
        combined_data.loc[target_team_df.index, target_stat_cols] = target_team_df[target_stat_cols]
        combined_data.loc[opp_team_df.index, opp_stat_cols] = opp_team_df[opp_stat_cols]
    return combined_data