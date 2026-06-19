import unittest
from unittest import mock
import pandas as pd
from src.data.madden import starters
import nfl_data_py as nfl


def _depth(rows):
    return pd.DataFrame(rows)


class TestAvailableStarters(unittest.TestCase):
    def _two_qbs(self):
        return _depth([
            {'club_code': 'KC', 'season': 2023, 'week': 1, 'position': 'QB',
             'depth_team': '1', 'gsis_id': 'STARTER'},
            {'club_code': 'KC', 'season': 2023, 'week': 1, 'position': 'QB',
             'depth_team': '2', 'gsis_id': 'BACKUP'},
        ])

    def test_healthy_starter_chosen(self):
        out = starters.available_starters(self._two_qbs(), pd.DataFrame(
            columns=['gsis_id', 'report_status']))
        qb = out[out['position'] == 'QB'].iloc[0]
        self.assertEqual(qb['gsis_id'], 'STARTER')

    def test_out_starter_promotes_backup(self):
        inj = pd.DataFrame([{'gsis_id': 'STARTER', 'report_status': 'Out'}])
        out = starters.available_starters(self._two_qbs(), inj)
        qb = out[out['position'] == 'QB'].iloc[0]
        self.assertEqual(qb['gsis_id'], 'BACKUP')

    def test_questionable_starter_still_starts(self):
        inj = pd.DataFrame([{'gsis_id': 'STARTER', 'report_status': 'Questionable'}])
        out = starters.available_starters(self._two_qbs(), inj)
        self.assertEqual(out[out['position'] == 'QB'].iloc[0]['gsis_id'], 'STARTER')


class TestGetWeeklyStartersResilience(unittest.TestCase):
    """get_weekly_starters must return an empty frame rather than raising when
    nfl_data_py raises (e.g. injuries not available before 2009) or when the
    depth-chart frame is missing expected columns (e.g. 2025 new schema)."""

    _EXPECTED_COLS = ['season', 'week', 'team', 'gsis_id', 'position']

    def test_proceeds_with_healthy_starters_when_injuries_unavailable(self):
        """Pre-2009 seasons: import_injuries raises ValueError; injuries fall back to
        empty (all players treated as healthy) and valid starters are still returned."""
        good_depth = pd.DataFrame([{
            'game_type': 'REG', 'club_code': 'KC', 'season': 2008, 'week': 1,
            'gsis_id': 'G1', 'position': 'QB', 'depth_team': '1',
        }])
        with mock.patch.object(nfl, 'import_depth_charts', return_value=good_depth), \
             mock.patch.object(nfl, 'import_injuries',
                               side_effect=ValueError('Data not available before 2009.')):
            out = starters.get_weekly_starters([2008])
        # Depth charts are valid so starters should be returned (non-empty)
        self.assertListEqual(list(out.columns), self._EXPECTED_COLS)
        self.assertFalse(out.empty)
        self.assertIn('G1', out['gsis_id'].values)

    def test_returns_empty_when_depth_chart_raises(self):
        """If import_depth_charts itself raises, an empty frame with correct columns is returned."""
        with mock.patch.object(nfl, 'import_depth_charts',
                               side_effect=Exception('network error')):
            out = starters.get_weekly_starters([2003])
        self.assertTrue(out.empty)
        self.assertListEqual(list(out.columns), self._EXPECTED_COLS)

    def test_returns_empty_when_depth_chart_missing_schema_columns(self):
        """2025-style depth charts have a different schema; the function returns empty."""
        new_schema_depth = pd.DataFrame([{
            'dt': '2025-01-01', 'team': 'KC', 'player_name': 'Patrick Mahomes',
            'gsis_id': 'G1', 'pos_grp': 'QB',
        }])
        with mock.patch.object(nfl, 'import_depth_charts', return_value=new_schema_depth):
            out = starters.get_weekly_starters([2025])
        self.assertTrue(out.empty)
        self.assertListEqual(list(out.columns), self._EXPECTED_COLS)


class TestLiveStarters(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = starters.get_weekly_starters([2023])

    def test_one_qb_per_team_week(self):
        qbs = self.df[self.df['position'] == 'QB']
        counts = qbs.groupby(['team', 'week']).size()
        self.assertTrue((counts == 1).all())

    def test_has_keys(self):
        for col in ['season', 'week', 'team', 'gsis_id', 'position']:
            self.assertIn(col, self.df.columns)
