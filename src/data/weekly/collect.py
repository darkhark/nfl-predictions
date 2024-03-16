import nfl_data_py as nfl


ONLY_NON_IDENTIFIER_COLUMNS = [
    'recent_team', 'season', 'week',
    'season_type', 'opponent_team', 'completions', 'attempts',
    'passing_yards', 'passing_tds', 'interceptions', 'sacks', 'sack_yards',
    'sack_fumbles', 'sack_fumbles_lost', 'passing_air_yards',
    'passing_yards_after_catch', 'passing_first_downs', 'passing_epa',
    'pacr', 'dakota', 'carries', 'rushing_yards',
    'rushing_tds', 'rushing_fumbles', 'rushing_fumbles_lost',
    'rushing_first_downs', 'rushing_epa',
    'receptions', 'targets', 'receiving_yards', 'receiving_tds',
    'receiving_fumbles', 'receiving_fumbles_lost', 'receiving_air_yards',
    'receiving_yards_after_catch', 'receiving_first_downs', 'receiving_epa',
    'racr', 'wopr', 'special_teams_tds'
]


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
        # Collect the statistics in broad terms of passing yards, rushing yards, etc. by team
        # This reduces the memory footprint of the data while still allowing for analysis
        df = df.groupby(
            ['recent_team', 'season', 'week', 'season_type', 'opponent_team']
        ).sum().reset_index()
        if weekly_df is None:
            weekly_df = df
        else:
            weekly_df = weekly_df.concat(df)
    weekly_df.sort_values(by=['recent_team', 'season', 'week'], inplace=True)
    return weekly_df
