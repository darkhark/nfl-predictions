import unittest
from src.data.weekly import collect

TEAM = 'team'
WEEK = 'week'
OPP_TEAM = 'opp_team'
PASSING_YARDS_OFF = 'off_passing_yards'
PASSING_YARDS_OFF_CUMULATIVE = 'off_passing_yards_cumulative_average'
PASSING_YARDS_DEF_OPP_CUMULATIVE = 'def_opp_passing_yards_cumulative_average'


class TestCollect(unittest.TestCase):

    def setUp(self):
        self.years = [2023]
        self.weekly_data = collect.get_weekly_data(self.years)

    def test_stats_are_cumulative_averages(self):
        cumulative_passing_yards_sum = 0
        bye_week = 100
        for week in range(1, 18):
            try:
                passing_yards = self.weekly_data[
                    (self.weekly_data[WEEK] == week) & (self.weekly_data[TEAM] == 'LA')
                ][PASSING_YARDS_OFF].values[0]
                cumulative_passing_yards_sum += passing_yards
                if week < bye_week:
                    actual_cumulative_passing_yards_average = cumulative_passing_yards_sum / week
                elif week > bye_week:
                    actual_cumulative_passing_yards_average = cumulative_passing_yards_sum / (week - 1)
                df_cumulative_passing_yards = self.weekly_data[
                    (self.weekly_data[WEEK] == week) & (self.weekly_data[TEAM] == 'LA')
                ][PASSING_YARDS_OFF_CUMULATIVE].values[0]
                # actual_cumulative_passing_yards_average cannot be unassigned because the IndexError will be raised
                # when the week is a bye week
                self.assertEqual(actual_cumulative_passing_yards_average, df_cumulative_passing_yards)
            except IndexError:
                print("Bye week is week ", week, " for LA. Continuing...")
                bye_week = week
                continue

    def test_weekly_data_defense_stats_match_opposing_offense_week_one(self):
        # Assert that the row where team is ARI and week == 1 has the same value in
        # passing_yards the row where opposing team is ARI and week == 1 value in
        # opp_def_passing_yards_cumulative_avg
        passing_yards = self.weekly_data[
            (self.weekly_data[TEAM] == 'ARI') & (self.weekly_data[WEEK] == 1)
        ][PASSING_YARDS_OFF].values[0]
        opponent = self.weekly_data[
            (self.weekly_data[TEAM] == 'ARI') & (self.weekly_data[WEEK] == 1)
        ][OPP_TEAM].values[0]
        opp_def_passing_yards_cumulative_avg = self.weekly_data[
            (self.weekly_data[OPP_TEAM] == opponent) & (self.weekly_data[WEEK] == 1)
            ][PASSING_YARDS_DEF_OPP_CUMULATIVE].values[0]
        self.assertEqual(passing_yards, opp_def_passing_yards_cumulative_avg)

    def test_opp_def_stats_are_cumulative_averages(self):
        cumulative_passing_yards_given_sum = 0
        bye_week = 100
        for week in range(1, 18):
            try:
                print("Week ", week)
                passing_yards_given = self.weekly_data[
                    (self.weekly_data[WEEK] == week) & (self.weekly_data[OPP_TEAM] == 'LA')
                ][PASSING_YARDS_OFF].values[0]
                cumulative_passing_yards_given_sum += passing_yards_given
                if week < bye_week:
                    actual_cumulative_passing_yards_average = cumulative_passing_yards_given_sum / week
                elif week > bye_week:
                    actual_cumulative_passing_yards_average = cumulative_passing_yards_given_sum / (week - 1)
                df_cumulative_passing_yards = self.weekly_data[
                    (self.weekly_data[WEEK] == week) & (self.weekly_data[OPP_TEAM] == 'LA')
                ][PASSING_YARDS_DEF_OPP_CUMULATIVE].values[0]
                # actual_cumulative_passing_yards_average cannot be unassigned because the IndexError will be raised
                # when the week is a bye week
                print("Expected", actual_cumulative_passing_yards_average)
                print("Actual", df_cumulative_passing_yards)
                self.assertEqual(actual_cumulative_passing_yards_average, df_cumulative_passing_yards)
            except IndexError:
                print("Bye week is week ", week, " for LA. Continuing...")
                bye_week = week
                continue


if __name__ == '__main__':
    unittest.main()
