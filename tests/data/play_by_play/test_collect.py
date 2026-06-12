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


if __name__ == '__main__':
    unittest.main()
