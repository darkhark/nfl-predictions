import unittest

import pandas as pd

from src.data import transformations


def make_frame():
    """Two teams, two weeks. Offense column where higher is better, defense column
    where lower is better. AAA is better on both in week 1; they swap in week 2."""
    return pd.DataFrame([
        {'team': 'AAA', 'opp_team': 'BBB', 'season': 2023, 'week': 1,
         'team_game_count': 1, 'opp_game_count': 1,
         'off_metric_cumulative_average': 30.0, 'def_opp_metric_cumulative_average': 10.0},
        {'team': 'BBB', 'opp_team': 'AAA', 'season': 2023, 'week': 1,
         'team_game_count': 1, 'opp_game_count': 1,
         'off_metric_cumulative_average': 20.0, 'def_opp_metric_cumulative_average': 15.0},
        {'team': 'AAA', 'opp_team': 'BBB', 'season': 2023, 'week': 2,
         'team_game_count': 2, 'opp_game_count': 2,
         'off_metric_cumulative_average': 18.0, 'def_opp_metric_cumulative_average': 16.0},
        {'team': 'BBB', 'opp_team': 'AAA', 'season': 2023, 'week': 2,
         'team_game_count': 2, 'opp_game_count': 2,
         'off_metric_cumulative_average': 25.0, 'def_opp_metric_cumulative_average': 11.0},
    ])


class TestAddRankAndRankChangeColumns(unittest.TestCase):

    def setUp(self):
        self.result = transformations.add_rank_and_rank_change_columns(
            make_frame(),
            off_cols=['off_metric_cumulative_average'],
            def_cols=['def_opp_metric_cumulative_average'],
        )

    def _value(self, team, week, col):
        row = self.result[(self.result['team'] == team) & (self.result['week'] == week)]
        return row[col].values[0]

    def test_offense_rank_one_is_highest_value(self):
        self.assertEqual(self._value('AAA', 1, 'off_metric_cumulative_average_rank'), 1)
        self.assertEqual(self._value('BBB', 1, 'off_metric_cumulative_average_rank'), 2)

    def test_defense_rank_one_is_lowest_value(self):
        self.assertEqual(self._value('AAA', 1, 'def_opp_metric_cumulative_average_rank'), 1)
        self.assertEqual(self._value('BBB', 1, 'def_opp_metric_cumulative_average_rank'), 2)

    def test_first_game_rank_change_is_zero(self):
        self.assertEqual(self._value('AAA', 1, 'off_metric_cumulative_average_rank_change'), 0)
        self.assertEqual(self._value('AAA', 1, 'def_opp_metric_cumulative_average_rank_change'), 0)

    def test_offense_rank_change_is_week_over_week_diff(self):
        # AAA falls from rank 1 to rank 2 -> change +1; BBB rises -> change -1
        self.assertEqual(self._value('AAA', 2, 'off_metric_cumulative_average_rank_change'), 1)
        self.assertEqual(self._value('BBB', 2, 'off_metric_cumulative_average_rank_change'), -1)

    def test_defense_rank_change_is_grouped_by_opponent(self):
        # def_opp columns describe the OPPONENT's defense, so changes group by opp_team.
        # On AAA's rows the opponent is BBB: BBB's defense allowed 10.0 then 16.0, so its
        # rank goes 1 -> 2 and the change is +1.
        self.assertEqual(self._value('AAA', 2, 'def_opp_metric_cumulative_average_rank_change'), 1)

    def test_nan_values_get_nan_rank(self):
        frame = make_frame()
        frame.loc[0, 'off_metric_cumulative_average'] = float('nan')
        result = transformations.add_rank_and_rank_change_columns(
            frame, off_cols=['off_metric_cumulative_average'],
            def_cols=['def_opp_metric_cumulative_average'],
        )
        week1 = result[result['week'] == 1]
        self.assertTrue(pd.isna(
            week1[week1['team'] == 'AAA']['off_metric_cumulative_average_rank'].values[0]
        ))
        # The remaining team still ranks 1 among teams with values
        self.assertEqual(
            week1[week1['team'] == 'BBB']['off_metric_cumulative_average_rank'].values[0], 1
        )


class TestTeamAbbrMappings(unittest.TestCase):

    def test_relocated_franchises_map_to_current_abbreviations(self):
        self.assertEqual(transformations.TEAM_ABBR_MAPPINGS['STL'], 'LA')
        self.assertEqual(transformations.TEAM_ABBR_MAPPINGS['SD'], 'LAC')
        self.assertEqual(transformations.TEAM_ABBR_MAPPINGS['OAK'], 'LV')


if __name__ == '__main__':
    unittest.main()
