from src.data.next_gen_stats import collect as next_gen_collect
from src.data.play_by_play import collect as pbp_collect
from src.data.weekly import collect as weekly_collect


def get_all_data(years):
    """
    Collects all data for the specified year(s).

    :param years: list of years to collect data for or a single year
    :return:
    """
    if isinstance(years, int):
        years = [years]
    elif not isinstance(years, list):
        raise TypeError('years must be a list or an integer')

    weekly_data = weekly_collect.get_weekly_data(years)
    # pbp_data = pbp_collect.get_pbp_data(years)
    # next_gen_data = next_gen_collect.get_next_gen_data(years)

    return weekly_data, pbp_data, next_gen_data