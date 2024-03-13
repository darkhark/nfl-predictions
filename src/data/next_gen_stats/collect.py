import nfl_data_py as nfl


def get_ngs_data(years, stat_type):
    """
    Collects NGS data for the specified year(s). When week == 0, the full season data (usually an average) is
    returned. When more than one year is specified, an additional column is added to the DataFrame to indicate the year

    :param years: list of years to collect data for. If multiple years are specified, the data will be aggregated
    :param stat_type: type of NGS data to collect. Options are 'passing', 'rushing', 'receiving'
    :return:
    """
    # remove hyphen from player_gsis_id and convert to int
    var_starts = ['avg', 'agg', 'max', 'passer_rating', 'comp', 'expe', 'rush', 'effi', 'perc', 'catch_per']
    ngs_passing_df = nfl.import_ngs_data(stat_type, years=years)
    ngs_passing_df['player_gsis_id'] = ngs_passing_df['player_gsis_id'].str.replace('-', '').astype(int)
    merge_cols = ['player_gsis_id', 'season'] if len(years) > 1 else ['player_gsis_id']
    ngs_passing_df = ngs_passing_df.merge(
        ngs_passing_df[ngs_passing_df['week'] == 0],
        on=merge_cols,
        suffixes=('', '_season')
    )
    ngs_passing_df = ngs_passing_df[ngs_passing_df['week'] != 0]
    ngs_passing_df.drop(columns=['week_season'], inplace=True)
    for col in ngs_passing_df.columns:
        col_to_get_diff = any([col.startswith(s) for s in var_starts])
        cols_not_helpful_diff = [
            # Rushing
            'expected_rush_yards',  # Week 0 is expected rush yards for the season while other weeks are per week
            'rush_yards',  # Week 0 is total rush yards for the season while other weeks are per week
            'rush_attempts',  # Week 0 is total rush attempts for the season while other weeks are per week
        ]
        if col_to_get_diff and not col.endswith('_season') and col not in cols_not_helpful_diff:
            ngs_passing_df[col + '_diff'] = ngs_passing_df[col] - ngs_passing_df[col + '_season']
    season_cols_to_drop = [col for col in ngs_passing_df.columns if col.endswith('_season')]
    ngs_passing_df.drop(columns=season_cols_to_drop, inplace=True)

    return ngs_passing_df


def _remove_identifier_columns_and_add_unique_index(df):
    """
    Removes columns that are used to identify players but are not useful for analysis. Adds a unique index to the
    DataFrame that consists of player_gsis_id_season_week

    :param df: DataFrame to remove columns from
    :return: DataFrame with identifier columns removed
    """
    id_cols = [
        'player_gsis_id', 'player_display_name', 'player_position', 'player_first_name', 'player_last_name',
        'player_jersey_number', 'player_short_name'
    ]

    df.drop(columns=id_cols, inplace=True)


# %%
# Collect data
passing_df = get_ngs_data([2018, 2019], 'passing')

# %%
rushing_df = get_ngs_data([2018, 2019], 'rushing')

# %%
receiving_df = get_ngs_data([2018, 2019], 'receiving')