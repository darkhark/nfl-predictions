# tests/experiments/test_madden_internal_ablation.py
"""Unit tests for madden_internal_ablation pure helpers.

All tests are fast (no training, no parquet I/O).
"""
import unittest

from scripts.experiments import madden_internal_ablation as mia


class TestMaddenUnit(unittest.TestCase):
    """madden_unit: maps column name -> unit family string."""

    def test_bare_qb_ovr(self):
        self.assertEqual(mia.madden_unit('target_madden_qb_ovr'), 'qb')

    def test_target_madden_qb_ovr(self):
        self.assertEqual(mia.madden_unit('target_madden_qb_ovr'), 'qb')

    def test_opp_madden_edge_left_ovr_diff_4g(self):
        self.assertEqual(mia.madden_unit('opp_madden_edge_left_ovr_diff_4g'), 'dfront')

    def test_madden_matchup_pass_pro(self):
        self.assertEqual(mia.madden_unit('madden_matchup_pass_pro'), 'matchup')

    def test_target_madden_cornerback_ovr(self):
        self.assertEqual(mia.madden_unit('target_madden_cornerback_ovr'), 'coverage')

    def test_opp_madden_interior_ol_ovr(self):
        self.assertEqual(mia.madden_unit('opp_madden_interior_ol_ovr'), 'oline')

    def test_target_madden_wr1_ovr(self):
        self.assertEqual(mia.madden_unit('target_madden_wr1_ovr'), 'offense_skill')

    def test_target_madden_rb_ovr_diff_prev(self):
        self.assertEqual(mia.madden_unit('target_madden_rb_ovr_diff_prev'), 'offense_skill')

    def test_opp_madden_lt_ovr(self):
        self.assertEqual(mia.madden_unit('opp_madden_lt_ovr'), 'oline')

    def test_target_madden_edge_ovr_diff_prev_season(self):
        self.assertEqual(mia.madden_unit('target_madden_edge_ovr_diff_prev_season'), 'dfront')

    def test_opp_madden_linebacker_ovr(self):
        self.assertEqual(mia.madden_unit('opp_madden_linebacker_ovr'), 'coverage')

    def test_madden_matchup_pass_rush(self):
        self.assertEqual(mia.madden_unit('madden_matchup_pass_rush'), 'matchup')

    def test_madden_matchup_interior(self):
        self.assertEqual(mia.madden_unit('madden_matchup_interior'), 'matchup')

    def test_madden_matchup_skill_cover(self):
        self.assertEqual(mia.madden_unit('madden_matchup_skill_cover'), 'matchup')


class TestMaddenMeasure(unittest.TestCase):
    """madden_measure: maps column name -> 'level' or 'diff'."""

    def test_target_madden_qb_ovr_is_level(self):
        self.assertEqual(mia.madden_measure('target_madden_qb_ovr'), 'level')

    def test_target_madden_qb_ovr_diff_prev_is_diff(self):
        self.assertEqual(mia.madden_measure('target_madden_qb_ovr_diff_prev'), 'diff')

    def test_target_madden_qb_ovr_diff_4g_is_diff(self):
        self.assertEqual(mia.madden_measure('target_madden_qb_ovr_diff_4g'), 'diff')

    def test_opp_madden_qb_ovr_diff_prev_season_is_diff(self):
        self.assertEqual(mia.madden_measure('opp_madden_qb_ovr_diff_prev_season'), 'diff')

    def test_madden_matchup_pass_pro_is_level(self):
        self.assertEqual(mia.madden_measure('madden_matchup_pass_pro'), 'level')

    def test_opp_madden_safety_ovr_is_level(self):
        self.assertEqual(mia.madden_measure('opp_madden_safety_ovr'), 'level')

    def test_target_madden_wr2_ovr_diff_prev_season_is_diff(self):
        self.assertEqual(mia.madden_measure('target_madden_wr2_ovr_diff_prev_season'), 'diff')


class TestMaddenUnitGroups(unittest.TestCase):
    """madden_unit_groups: each input col lands in exactly one group."""

    SAMPLE_COLS = [
        'target_madden_qb_ovr',
        'opp_madden_qb_ovr_diff_prev',
        'target_madden_rb_ovr',
        'opp_madden_wr1_ovr_diff_4g',
        'target_madden_lt_ovr',
        'opp_madden_interior_dl_ovr_diff_prev_season',
        'target_madden_cornerback_ovr',
        'madden_matchup_pass_pro',
    ]

    def test_correct_unit_grouping(self):
        g = mia.madden_unit_groups(self.SAMPLE_COLS)
        self.assertIn('qb', g)
        self.assertIn('offense_skill', g)
        self.assertIn('oline', g)
        self.assertIn('dfront', g)
        self.assertIn('coverage', g)
        self.assertIn('matchup', g)

        self.assertCountEqual(g['qb'], ['target_madden_qb_ovr', 'opp_madden_qb_ovr_diff_prev'])
        self.assertCountEqual(g['offense_skill'], ['target_madden_rb_ovr', 'opp_madden_wr1_ovr_diff_4g'])
        self.assertCountEqual(g['oline'], ['target_madden_lt_ovr'])
        self.assertCountEqual(g['dfront'], ['opp_madden_interior_dl_ovr_diff_prev_season'])
        self.assertCountEqual(g['coverage'], ['target_madden_cornerback_ovr'])
        self.assertCountEqual(g['matchup'], ['madden_matchup_pass_pro'])

    def test_every_col_in_exactly_one_group(self):
        g = mia.madden_unit_groups(self.SAMPLE_COLS)
        all_grouped = [c for cols in g.values() for c in cols]
        self.assertEqual(sorted(all_grouped), sorted(self.SAMPLE_COLS))


class TestMaddenMeasureGroups(unittest.TestCase):
    """madden_measure_groups: each input col lands in exactly one group."""

    SAMPLE_COLS = [
        'target_madden_qb_ovr',             # level
        'opp_madden_qb_ovr_diff_prev',       # diff
        'target_madden_rb_ovr_diff_4g',      # diff
        'opp_madden_wr1_ovr',               # level
        'target_madden_lt_ovr_diff_prev_season',  # diff
        'madden_matchup_pass_pro',          # level (matchup delta = level by convention)
    ]

    def test_correct_measure_grouping(self):
        g = mia.madden_measure_groups(self.SAMPLE_COLS)
        self.assertIn('level', g)
        self.assertIn('diff', g)

        self.assertCountEqual(g['level'], [
            'target_madden_qb_ovr',
            'opp_madden_wr1_ovr',
            'madden_matchup_pass_pro',
        ])
        self.assertCountEqual(g['diff'], [
            'opp_madden_qb_ovr_diff_prev',
            'target_madden_rb_ovr_diff_4g',
            'target_madden_lt_ovr_diff_prev_season',
        ])

    def test_every_col_in_exactly_one_group(self):
        g = mia.madden_measure_groups(self.SAMPLE_COLS)
        all_grouped = [c for cols in g.values() for c in cols]
        self.assertEqual(sorted(all_grouped), sorted(self.SAMPLE_COLS))


if __name__ == '__main__':
    unittest.main()
