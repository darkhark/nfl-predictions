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

    home_team_is_team = schedule_data.merge(
        weekly_data,
        left_on=['home_team', 'season', 'week'],
        right_on=['team', 'season', 'week'],
        how='inner'
    )
    # replace the 'team' column with 'home_team'
    home_team_is_team.drop(columns=['team', 'opp_team'], inplace=True)
    # in every column that contains 'opp', replace 'opp' with 'away'
    home_team_is_team = home_team_is_team.rename(
        columns=lambda x: x.replace('opp', 'away') if 'opp' in x else x
    )
    # in every column that contains 'off', replace 'off' with 'home'
    home_team_is_team = home_team_is_team.rename(
        columns=lambda x: x.replace('off', 'off_home') if 'off' in x else x
    )
    # in every column that starts with 'team_', replace 'team_' with 'home_'
    home_team_is_team = home_team_is_team.rename(
        columns=lambda x: x.replace('team_', 'home_') if 'team_' in x else x
    )

    away_team_is_team = schedule_data.merge(
        weekly_data,
        left_on=['away_team', 'season', 'week'],
        right_on=['team', 'season', 'week'],
        how='inner'
    )

    # replace the 'team' column with 'away_team'
    away_team_is_team.drop(columns=['team', 'opp_team'], inplace=True)
    # in every column that contains 'opp', replace 'opp' with 'home'
    away_team_is_team = away_team_is_team.rename(
        columns=lambda x: x.replace('opp', 'home') if 'opp' in x else x
    )
    # in every column that contains 'off', replace 'off' with 'away'
    away_team_is_team = away_team_is_team.rename(
        columns=lambda x: x.replace('off', 'off_away') if 'off' in x else x
    )
    # in every column that starts with 'team_', replace 'team_' with 'away_'
    away_team_is_team = away_team_is_team.rename(
        columns=lambda x: x.replace('team_', 'away_') if 'team_' in x else x
    )
    # merge the dataframes, but do not create any new columns where the names are the same
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



    return combined_data
