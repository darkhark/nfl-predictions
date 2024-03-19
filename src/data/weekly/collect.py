import nfl_data_py as nfl
import pandas as pd

# There are times when passing yards != receiving yards, but it's rare so we'll ignore receiving yards
# and tds since they are duplicated in the passing stats
ONLY_NON_IDENTIFIER_COLUMNS = [
    'recent_team', 'season', 'week',
    'season_type', 'opponent_team', 'completions', 'attempts',
    'passing_yards', 'passing_tds', 'interceptions', 'sacks', 'sack_yards',
    'sack_fumbles', 'sack_fumbles_lost', 'passing_air_yards',
    'passing_yards_after_catch', 'passing_first_downs', 'passing_epa',
    'pacr', 'dakota', 'carries', 'rushing_yards',
    'rushing_tds', 'rushing_fumbles', 'rushing_fumbles_lost',
    'rushing_first_downs', 'rushing_epa',
    'receiving_fumbles', 'receiving_fumbles_lost',
    'racr', 'wopr', 'special_teams_tds'
]

TEAM_COL = 'team'
OPPONENT_TEAM_COL = 'opponent_team'
SEASON_COL = 'season'
WEEK_COL = 'week'
SEASON_TYPE_COL = 'season_type'
TEAM_GAME_COUNT_COL = 'team_game_count'
OPP_GAME_COUNT_COL = 'opp_game_count'


def get_weekly_data(years):
    """
    Collects weekly data for the specified year(s) and week(s).

    :param years: list of years to collect data for or a single year
    :return:
    """
    if isinstance(years, int):
        years = [years]
    elif not isinstance(years, list):
        raise TypeError('years must be a list or an integer')

    weekly_df = None
    for year in years:
        df = nfl.import_weekly_data(years=[year], columns=ONLY_NON_IDENTIFIER_COLUMNS)
        df.rename(columns={'recent_team': TEAM_COL}, inplace=True)
        # Remove all week 0 records and records where the team and opponent team are the same
        df = df[(df[WEEK_COL] != 0) & (df[TEAM_COL] != df[OPPONENT_TEAM_COL])]
        # Collect the statistics in broad terms of passing yards, rushing yards, etc. by team
        # This reduces the memory footprint of the data while still allowing for analysis
        df = df.groupby(
            [TEAM_COL, SEASON_COL, WEEK_COL, SEASON_TYPE_COL, OPPONENT_TEAM_COL]
        ).sum().reset_index()
        df.sort_values(by=[TEAM_COL, SEASON_COL, WEEK_COL], inplace=True)
        # Add a game count column to the dataframe since week is not always a good indicator of games played
        df[TEAM_GAME_COUNT_COL] = df.groupby([TEAM_COL, SEASON_COL]).cumcount() + 1
        df.sort_values(by=[OPPONENT_TEAM_COL, SEASON_COL, WEEK_COL], inplace=True)
        df[OPP_GAME_COUNT_COL] = df.groupby([OPPONENT_TEAM_COL, SEASON_COL]).cumcount() + 1

        df = create_team_offense_columns(df)
        df = create_opponent_defensive_columns(df)
        # TODO: Fix the below functions
        # df = create_opponent_offensive_columns(df)
        # df = create_defensive_team_columns(df)
        if weekly_df is None:
            weekly_df = df
        else:
            weekly_df = weekly_df.concat(df)
    weekly_df.sort_values(by=[TEAM_COL, SEASON_COL, TEAM_GAME_COUNT_COL], inplace=True)
    return weekly_df


def create_cumulative_columns(df, groupby_columns, column_prefix, game_count_col, swap_team_and_opponent=False):
    """
    Create cumulative columns for the specified columns in the dataframe. The cumulative columns are
    the sum of the specified columns for the groupby_columns. The cumulative average is the cumulative
    sum divided by the week. The cumulative average change is the change in the cumulative average
    from the previous week.

    :param df: The dataframe to create the columns for
    :param groupby_columns: The columns to group by
    :param column_prefix: The prefix to add to the new columns, e.g. 'off' or 'def_opp'
    :return: The dataframe with the new columns
    """
    df = df.sort_values(by=groupby_columns + [game_count_col])
    # divisor_col = TEAM_GAME_COUNT_COL if TEAM_GAME_COUNT_COL in groupby_columns else OPP_GAME_COUNT_COL
    new_df = pd.DataFrame()
    for column in ONLY_NON_IDENTIFIER_COLUMNS[5:]:
        new_df[f'{column_prefix}_{column}_cumulative_sum'] = df.groupby(groupby_columns)[column].cumsum()
        # Need to use game count instead of week because bye weeks are not counted
        new_df[f'{column_prefix}_{column}_cumulative_average'] = new_df[f'{column_prefix}_{column}_cumulative_sum'] / df[game_count_col]
        new_df.drop(columns=[f'{column_prefix}_{column}_cumulative_sum'], inplace=True)
        new_df[f'{column_prefix}_{column}_cumulative_average_change'] = new_df[f'{column_prefix}_{column}_cumulative_average'].diff()
        new_df.fillna({f'{column_prefix}_{column}_cumulative_average_change': 0}, inplace=True)
    if swap_team_and_opponent:
        if TEAM_COL in groupby_columns:
            new_df[OPPONENT_TEAM_COL] = df[TEAM_COL]
        else:
            new_df[TEAM_COL] = df[OPPONENT_TEAM_COL]
    return pd.concat([df, new_df], axis=1)


def create_team_offense_columns(df):
    """
    Create cumulative columns for the team's offense.
    :param df: The dataframe to create the columns for
    :return: The dataframe with the new columns
    """
    return create_cumulative_columns(
        df,
        [TEAM_COL, SEASON_COL],
        'off',
        TEAM_GAME_COUNT_COL
    )

# TODO: Fix the below functions

# def create_opponent_offensive_columns(df):
#     """
#     Create cumulative columns for the team's opponent's offense.
#     :param df: The dataframe to create the columns for
#     :return: The dataframe with the new columns
#     """
#     return create_cumulative_columns(
#         df,
#         [TEAM_COL, SEASON_COL],
#         'off_opp',
#         swap_team_and_opponent=True
#     )
#
#
# def create_defensive_team_columns(df):
#     """
#     Create cumulative columns for the team's defense.
#     :param df: The dataframe to create the columns for
#     :return: The dataframe with the new columns
#     """
#     return create_cumulative_columns(
#         df,
#         [OPPONENT_TEAM_COL, SEASON_COL],
#         'def',
#         swap_team_and_opponent=True
#     )


def create_opponent_defensive_columns(df):
    """
    Create cumulative columns for the team's defense.
    :param df: The dataframe to create the columns for
    :return: The dataframe with the new columns
    """
    return create_cumulative_columns(
        df,
        [OPPONENT_TEAM_COL, SEASON_COL],
        'def_opp',
        OPP_GAME_COUNT_COL,
    )
