import unittest
from unittest import mock

import pandas as pd

from src.data.play_by_play import collect


def make_play(posteam='AAA', defteam='BBB', season=2023, week=1, season_type='REG',
              play_id=1, game_id='2023_01_AAA_BBB', is_pass=0, is_rush=0,
              down=1, yardline_100=75.0, third_down_converted=0.0, success=0.0,
              epa=0.0, wp=0.5, xpass=None, fixed_drive=1, fixed_drive_result='Punt'):
    """One synthetic nflfastR play row. 'pass'/'rush' are reserved words as kwargs,
    hence is_pass/is_rush."""
    return {
        'posteam': posteam, 'defteam': defteam, 'season': season, 'week': week,
        'season_type': season_type, 'play_id': play_id, 'game_id': game_id,
        'pass': is_pass, 'rush': is_rush, 'down': down, 'yardline_100': yardline_100,
        'third_down_converted': third_down_converted, 'success': success, 'epa': epa,
        'wp': wp, 'xpass': xpass, 'fixed_drive': fixed_drive,
        'fixed_drive_result': fixed_drive_result,
    }


class TestAssignWpContext(unittest.TestCase):

    def test_thresholds_and_boundaries(self):
        wp = pd.Series([0.5, 0.05, 0.95, 0.951, 0.049, 0.0, 1.0, float('nan')])
        result = collect._assign_wp_context(wp)
        expected = [
            collect.COMPETITIVE,        # 0.5
            collect.COMPETITIVE,        # 0.05 boundary is inclusive
            collect.COMPETITIVE,        # 0.95 boundary is inclusive
            collect.GARBAGE_LEADING,    # 0.951
            collect.GARBAGE_TRAILING,   # 0.049
            collect.GARBAGE_TRAILING,   # 0.0
            collect.GARBAGE_LEADING,    # 1.0
            collect.COMPETITIVE,        # NaN wp defaults to competitive
        ]
        self.assertEqual(list(result), expected)


class TestAggregatePlayComponents(unittest.TestCase):

    def setUp(self):
        plays = pd.DataFrame([
            # competitive: one successful pass, one failed rush (both early downs)
            make_play(play_id=1, is_pass=1, down=1, success=1.0, epa=0.5, wp=0.5, xpass=0.6),
            make_play(play_id=2, is_rush=1, down=2, success=0.0, epa=-0.2, wp=0.5, xpass=0.3),
            # garbage_leading: converted third-down pass, no xpass value
            make_play(play_id=3, is_pass=1, down=3, third_down_converted=1.0,
                      success=1.0, epa=1.0, wp=0.96, xpass=None),
            # not a pass or rush play (e.g. kickoff): must be excluded entirely
            make_play(play_id=4, is_pass=0, is_rush=0, epa=2.0, wp=0.5),
        ])
        self.result = collect._aggregate_play_components(plays)

    def _row(self, context):
        return self.result[self.result[collect.CONTEXT_COL] == context].iloc[0]

    def test_competitive_components(self):
        row = self._row(collect.COMPETITIVE)
        self.assertEqual(row['play_count'], 2)
        self.assertAlmostEqual(row['epa_sum'], 0.3)
        self.assertEqual(row['success_sum'], 1)
        self.assertEqual(row['dropback_count'], 1)
        self.assertAlmostEqual(row['dropback_epa_sum'], 0.5)
        self.assertEqual(row['dropback_success_sum'], 1)
        self.assertEqual(row['rush_count'], 1)
        self.assertAlmostEqual(row['rush_epa_sum'], -0.2)
        self.assertEqual(row['rush_success_sum'], 0)
        self.assertEqual(row['early_down_count'], 2)
        self.assertEqual(row['early_down_success_sum'], 1)
        self.assertEqual(row['third_down_count'], 0)
        self.assertEqual(row['third_down_conversion_sum'], 0)
        self.assertEqual(row['xpass_play_count'], 2)
        # (1 - 0.6) + (0 - 0.3) = 0.1
        self.assertAlmostEqual(row['pass_minus_xpass_sum'], 0.1)

    def test_garbage_leading_components(self):
        row = self._row(collect.GARBAGE_LEADING)
        self.assertEqual(row['play_count'], 1)
        self.assertEqual(row['third_down_count'], 1)
        self.assertEqual(row['third_down_conversion_sum'], 1)
        # xpass was NaN: play contributes to neither PROE component
        self.assertEqual(row['xpass_play_count'], 0)
        self.assertEqual(row['pass_minus_xpass_sum'], 0)

    def test_non_pass_rush_plays_are_excluded(self):
        self.assertEqual(self.result['play_count'].sum(), 3)

    def test_grain_is_team_week_context(self):
        expected_keys = collect.AGGREGATION_KEY_COLUMNS + [collect.CONTEXT_COL]
        for key in expected_keys:
            self.assertIn(key, self.result.columns)
        self.assertFalse(self.result.duplicated(subset=expected_keys).any())


if __name__ == '__main__':
    unittest.main()
