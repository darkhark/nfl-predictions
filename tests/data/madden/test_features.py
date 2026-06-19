import unittest
import pandas as pd
from src.data.madden import features


class TestBuildTeamWeekOveralls(unittest.TestCase):
    def _starters(self):
        # one team-week: QB, two tackles, three WRs, a left edge
        return pd.DataFrame([
            {'season': 2023, 'week': 1, 'team': 'KC', 'gsis_id': g, 'position': p}
            for g, p in [('QB1', 'QB'), ('T1', 'T'), ('T2', 'T'),
                         ('W1', 'WR'), ('W2', 'WR'), ('W3', 'WR'), ('E1', 'OLB')]
        ])

    def _players(self):
        return pd.DataFrame([
            {'gsis_id': 'QB1', 'overall': 96, 'position': 'QB', 'role': 'qb', 'side': 'none'},
            {'gsis_id': 'T1', 'overall': 90, 'position': 'LT', 'role': 'exterior_ol', 'side': 'left'},
            {'gsis_id': 'T2', 'overall': 78, 'position': 'RT', 'role': 'exterior_ol', 'side': 'right'},
            {'gsis_id': 'W1', 'overall': 99, 'position': 'WR', 'role': 'receiver', 'side': 'none'},
            {'gsis_id': 'W2', 'overall': 84, 'position': 'WR', 'role': 'receiver', 'side': 'none'},
            {'gsis_id': 'W3', 'overall': 72, 'position': 'WR', 'role': 'receiver', 'side': 'none'},
            {'gsis_id': 'E1', 'overall': 88, 'position': 'LOLB', 'role': 'edge', 'side': 'left'},
        ])

    def test_granular_slots_and_groups(self):
        out = features.build_team_week_overalls(self._starters(), self._players())
        row = out.iloc[0]
        self.assertEqual(row['madden_qb_ovr'], 96)
        self.assertEqual(row['madden_lt_ovr'], 90)    # side from Madden position
        self.assertEqual(row['madden_rt_ovr'], 78)
        self.assertEqual(row['madden_wr1_ovr'], 99)   # ranked by overall desc
        self.assertEqual(row['madden_wr2_ovr'], 84)
        self.assertEqual(row['madden_wr3_ovr'], 72)
        self.assertEqual(row['madden_edge_left_ovr'], 88)
        self.assertEqual(row['madden_receivers_ovr'], (99 + 84 + 72) / 3)
        self.assertEqual(row['madden_exterior_ol_ovr'], (90 + 78) / 2)


class TestWithinSeasonDiffs(unittest.TestCase):
    def test_prev_and_4g_diffs(self):
        df = pd.DataFrame([
            {'season': 2023, 'week': w, 'team': 'KC', 'madden_qb_ovr': ovr}
            for w, ovr in [(1, 96), (2, 96), (3, 70), (4, 70), (5, 96)]
        ])
        out = features.add_within_season_diffs(df).sort_values('week')
        diffs = out['madden_qb_ovr_diff_prev'].tolist()
        self.assertTrue(pd.isna(diffs[0]))            # first game -> NA
        self.assertEqual(diffs[2], -26)               # 70 - 96 (starter went down)
        self.assertEqual(diffs[4], 26)                # 96 - 70 (starter returned)
        fourg = out['madden_qb_ovr_diff_4g'].tolist()
        self.assertTrue(pd.isna(fourg[3]))            # game 4 -> still NA
        self.assertEqual(fourg[4], 0)                 # week5 96 vs week1 96


class TestPrevSeasonDiff(unittest.TestCase):
    def test_prev_season_diff(self):
        cur = pd.DataFrame([{'season': 2024, 'week': 1, 'team': 'KC', 'madden_qb_ovr': 99}])
        prev = pd.DataFrame([{'season': 2023, 'week': 1, 'team': 'KC', 'madden_qb_ovr': 96}])
        out = features.add_prev_season_diff(cur, prev)
        self.assertEqual(out.iloc[0]['madden_qb_ovr_diff_prev_season'], 3)


class TestMatchupColumns(unittest.TestCase):
    def test_matchup_deltas(self):
        df = pd.DataFrame([{
            'target_madden_exterior_ol_ovr': 90, 'opp_madden_edge_ovr': 80,
            'target_madden_interior_ol_ovr': 85, 'opp_madden_interior_dl_ovr': 75,
            'target_madden_receivers_ovr': 88, 'opp_madden_cornerback_ovr': 70,
            'opp_madden_safety_ovr': 72, 'opp_madden_exterior_ol_ovr': 60,
            'target_madden_edge_ovr': 90}])
        out = features.add_madden_matchup_columns(df).iloc[0]
        self.assertEqual(out['madden_matchup_pass_pro'], 10)        # 90-80
        self.assertEqual(out['madden_matchup_interior'], 10)        # 85-75
        self.assertEqual(out['madden_matchup_skill_cover'], 88 - 71)  # 88-(70+72)/2
        self.assertEqual(out['madden_matchup_pass_rush'], -30)      # 60-90


class TestMatchupColumnsNoCoverage(unittest.TestCase):
    def test_missing_source_cols_produce_float_nan(self):
        """add_madden_matchup_columns on a frame with no madden source columns must
        produce float64 (not object) columns so XGBoost accepts them."""
        df = pd.DataFrame([
            {'team': 'KC', 'season': 2025, 'week': 1},
            {'team': 'SF', 'season': 2025, 'week': 1},
        ])
        out = features.add_madden_matchup_columns(df)
        expected_cols = [
            'madden_matchup_pass_pro',
            'madden_matchup_interior',
            'madden_matchup_skill_cover',
            'madden_matchup_pass_rush',
        ]
        for col in expected_cols:
            self.assertIn(col, out.columns, msg=f'{col} missing from output')
            self.assertTrue(
                pd.api.types.is_float_dtype(out[col]),
                msg=f'{col} has dtype {out[col].dtype}, expected float',
            )
            self.assertTrue(out[col].isna().all(), msg=f'{col} should be all NaN')


if __name__ == '__main__':
    unittest.main()
