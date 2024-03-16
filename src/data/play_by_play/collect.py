import nfl_data_py as nfl


# TODO: Wait to implement play by play until I can determine how to use it with weekly
# It'll likely require me to sum different portions of the data to get the weekly stats
# Something like avg_yards_right_tackle_gap, snaps_right_tackle_gap, etc. per week
def get_play_by_play_data(years):
    """
    Collects play by play data for the specified year(s).

    :param years: list of years to collect data for
    :return:
    """
    pbp_df = nfl.import_pbp_data(years=years)
    return pbp_df

