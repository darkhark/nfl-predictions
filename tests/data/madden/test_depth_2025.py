import unittest
import pandas as pd
from src.data.madden import depth_2025


def _schedule():
    # BAL plays week 1 on 2025-09-07, week 2 on 2025-09-14.
    return pd.DataFrame([
        {'season': 2025, 'week': 1, 'game_type': 'REG', 'gameday': '2025-09-07',
         'gametime': '13:00', 'home_team': 'BAL', 'away_team': 'CLE'},
        {'season': 2025, 'week': 2, 'game_type': 'REG', 'gameday': '2025-09-14',
         'gametime': '13:00', 'home_team': 'CIN', 'away_team': 'BAL'},
        {'season': 2025, 'week': 1, 'game_type': 'PRE', 'gameday': '2025-08-10',
         'gametime': '13:00', 'home_team': 'BAL', 'away_team': 'IND'},
    ])


def _depth():
    # Two snapshots: one before wk1 (09-03) and a fresher one also before wk1 (09-05);
    # a post-wk1 snapshot (09-10) must NOT leak into week 1 but feeds week 2.
    rows = []
    for dt, qb in [('2025-09-03T10:00:00Z', 'OLD_QB'),
                   ('2025-09-05T10:00:00Z', 'NEW_QB'),
                   ('2025-09-10T10:00:00Z', 'WK2_QB')]:
        rows += [
            {'dt': dt, 'team': 'BAL', 'pos_abb': 'QB', 'pos_rank': 1, 'gsis_id': qb},
            {'dt': dt, 'team': 'BAL', 'pos_abb': 'WR', 'pos_rank': 1, 'gsis_id': 'WR1'},
            {'dt': dt, 'team': 'BAL', 'pos_abb': 'WR', 'pos_rank': 2, 'gsis_id': 'WR2'},
            {'dt': dt, 'team': 'BAL', 'pos_abb': 'LT', 'pos_rank': 1, 'gsis_id': 'LTACK'},
            {'dt': dt, 'team': 'BAL', 'pos_abb': 'KR', 'pos_rank': 1, 'gsis_id': 'RETURNER'},
        ]
    return pd.DataFrame(rows)


class TestNormalize2025Depth(unittest.TestCase):
    def test_picks_latest_pre_gameday_snapshot(self):
        out = depth_2025.normalize_2025_depth(_depth(), _schedule(), 2025)
        wk1 = out[(out['week'] == 1) & (out['position'] == 'QB')]
        self.assertEqual(list(wk1['gsis_id']), ['NEW_QB'])  # 09-05, not 09-03, not 09-10

    def test_coarsens_positions_and_keeps_rank(self):
        out = depth_2025.normalize_2025_depth(_depth(), _schedule(), 2025)
        wk1 = out[out['week'] == 1]
        self.assertEqual(set(wk1['position']), {'QB', 'WR', 'T'})  # LT->T, KR dropped
        lt = wk1[wk1['position'] == 'T'].iloc[0]
        self.assertEqual(lt['depth_team'], 1)

    def test_reg_only_and_contract_columns(self):
        out = depth_2025.normalize_2025_depth(_depth(), _schedule(), 2025)
        self.assertEqual(set(out['game_type']), {'REG'})
        self.assertEqual(list(out.columns),
                         ['season', 'week', 'club_code', 'game_type',
                          'position', 'depth_team', 'gsis_id'])
        self.assertEqual(set(out['week']), {1, 2})

    def test_empty_depth_returns_empty_contract(self):
        out = depth_2025.normalize_2025_depth(pd.DataFrame(), _schedule(), 2025)
        self.assertEqual(len(out), 0)
        self.assertIn('depth_team', out.columns)
