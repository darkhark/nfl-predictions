import unittest
from unittest import mock

import pandas as pd

from src.data.weekly import collect

TEAM = 'team'
WEEK = 'week'
SEASON = 'season'
OPP_TEAM = 'opp_team'
TEAM_GAME_COUNT = 'team_game_count'
OPP_GAME_COUNT = 'opp_game_count'
PASSING_YARDS_OFF = 'off_passing_yards'
PASSING_YARDS_OFF_CUMULATIVE = 'off_passing_yards_cumulative_average'
PASSING_YARDS_DEF_OPP_CUMULATIVE = 'def_opp_passing_yards_cumulative_average'
PASSING_YARDS_OFF_RANK = 'off_passing_yards_cumulative_average_rank'
PASSING_YARDS_OFF_RANK_CHANGE = 'off_passing_yards_cumulative_average_rank_change'
PASSING_YARDS_DEF_OPP_RANK = 'def_opp_passing_yards_cumulative_average_rank'
PASSING_YARDS_DEF_OPP_RANK_CHANGE = 'def_opp_passing_yards_cumulative_average_rank_change'


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
    def test_offense_rank_one_holds_the_max_cumulative_average(self):
        week_eight = self.weekly_data[self.weekly_data[WEEK] == 8]
        max_cumulative_average = week_eight[PASSING_YARDS_OFF_CUMULATIVE].max()
        rank_one_rows = week_eight[week_eight[PASSING_YARDS_OFF_RANK] == 1]
        self.assertFalse(rank_one_rows.empty)
        for value in rank_one_rows[PASSING_YARDS_OFF_CUMULATIVE]:
            self.assertEqual(value, max_cumulative_average)

    def test_defense_rank_one_holds_the_min_cumulative_average(self):
        week_eight = self.weekly_data[self.weekly_data[WEEK] == 8]
        min_cumulative_average = week_eight[PASSING_YARDS_DEF_OPP_CUMULATIVE].min()
        rank_one_rows = week_eight[week_eight[PASSING_YARDS_DEF_OPP_RANK] == 1]
        self.assertFalse(rank_one_rows.empty)
        for value in rank_one_rows[PASSING_YARDS_DEF_OPP_CUMULATIVE]:
            self.assertEqual(value, min_cumulative_average)

    def test_ranks_span_one_to_number_of_teams_each_week(self):
        for week in range(1, 18):
            week_data = self.weekly_data[self.weekly_data[WEEK] == week]
            number_of_teams = week_data[TEAM].nunique()
            ranks = week_data[PASSING_YARDS_OFF_RANK]
            self.assertEqual(ranks.min(), 1)
            self.assertLessEqual(ranks.max(), number_of_teams)
            self.assertGreaterEqual(ranks.min(), 1)

    def test_offense_rank_change_equals_consecutive_game_rank_diff(self):
        la_games = self.weekly_data[self.weekly_data[TEAM] == 'LA'].sort_values(TEAM_GAME_COUNT)
        previous_rank = None
        for _, row in la_games.iterrows():
            if previous_rank is not None:
                expected_change = row[PASSING_YARDS_OFF_RANK] - previous_rank
                self.assertEqual(row[PASSING_YARDS_OFF_RANK_CHANGE], expected_change)
            previous_rank = row[PASSING_YARDS_OFF_RANK]

    def test_defense_rank_change_equals_consecutive_opp_game_rank_diff(self):
        la_games = self.weekly_data[self.weekly_data[OPP_TEAM] == 'LA'].sort_values(OPP_GAME_COUNT)
        previous_rank = None
        for _, row in la_games.iterrows():
            if previous_rank is not None:
                expected_change = row[PASSING_YARDS_DEF_OPP_RANK] - previous_rank
                self.assertEqual(row[PASSING_YARDS_DEF_OPP_RANK_CHANGE], expected_change)
            previous_rank = row[PASSING_YARDS_DEF_OPP_RANK]

    def test_offense_rank_change_is_zero_for_first_game(self):
        first_games = self.weekly_data[self.weekly_data[TEAM_GAME_COUNT] == 1]
        self.assertTrue((first_games[PASSING_YARDS_OFF_RANK_CHANGE] == 0).all())

    def test_defense_rank_change_is_zero_for_first_game(self):
        first_games = self.weekly_data[self.weekly_data[OPP_GAME_COUNT] == 1]
        self.assertTrue((first_games[PASSING_YARDS_DEF_OPP_RANK_CHANGE] == 0).all())

    def test_offense_rank_order_matches_descending_cumulative_average(self):
        week_eight = self.weekly_data[self.weekly_data[WEEK] == 8]
        values_sorted_by_rank = week_eight.sort_values(
            PASSING_YARDS_OFF_RANK
        )[PASSING_YARDS_OFF_CUMULATIVE].tolist()
        self.assertEqual(values_sorted_by_rank, sorted(values_sorted_by_rank, reverse=True))


class TestLoadWeeklyYear(unittest.TestCase):
    """Unit tests for the source mapping from nflverse's 'stats_player' release to the
    canonical column names the pipeline expects. The network read is mocked out."""

    def _fake_new_release_row(self):
        # A single-row frame using the new release's column names, including the renamed
        # columns and an extra column (dakota was removed upstream and must not be required).
        return pd.DataFrame([{
            'team': 'LA', 'season': 2025, 'week': 1, 'season_type': 'REG',
            'opponent_team': 'SEA', 'completions': 20, 'attempts': 30,
            'passing_yards': 250, 'passing_tds': 2, 'passing_interceptions': 1,
            'sacks_suffered': 3, 'sack_yards_lost': -21,
            'sack_fumbles': 0, 'sack_fumbles_lost': 0, 'passing_air_yards': 300,
            'passing_yards_after_catch': 120, 'passing_first_downs': 12, 'passing_epa': 5.5,
            'pacr': 0.8, 'carries': 25, 'rushing_yards': 110, 'rushing_tds': 1,
            'rushing_fumbles': 0, 'rushing_fumbles_lost': 0, 'rushing_first_downs': 6,
            'rushing_epa': 2.1, 'receiving_fumbles': 0, 'receiving_fumbles_lost': 0,
            'racr': 0.9, 'wopr': 0.0, 'special_teams_tds': 0,
            'some_new_unused_column': 99,
        }])

    def test_returns_only_canonical_columns(self):
        with mock.patch.object(collect.pd, 'read_parquet', return_value=self._fake_new_release_row()):
            result = collect._load_weekly_year(2025)
        self.assertEqual(list(result.columns), collect.ONLY_NON_IDENTIFIER_COLUMNS)

    def test_renames_new_release_columns_to_canonical_names(self):
        with mock.patch.object(collect.pd, 'read_parquet', return_value=self._fake_new_release_row()):
            result = collect._load_weekly_year(2025)
        row = result.iloc[0]
        self.assertEqual(row['recent_team'], 'LA')
        self.assertEqual(row['interceptions'], 1)
        self.assertEqual(row['sacks'], 3)

    def test_sack_yards_restored_to_positive_convention(self):
        with mock.patch.object(collect.pd, 'read_parquet', return_value=self._fake_new_release_row()):
            result = collect._load_weekly_year(2025)
        # New release stores -21; pipeline expects the positive legacy convention.
        self.assertEqual(result.iloc[0]['sack_yards'], 21)

    def test_dakota_is_not_part_of_the_feature_set(self):
        self.assertNotIn('dakota', collect.ONLY_NON_IDENTIFIER_COLUMNS)


if __name__ == '__main__':
    unittest.main()
