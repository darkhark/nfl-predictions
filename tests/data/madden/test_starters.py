import unittest
from unittest import mock
import pandas as pd
from src.data.madden import starters
from src.data.madden import depth_2025  # noqa: E402  (top of file with the others)
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


class TestGetWeeklyStarters2025Dispatch(unittest.TestCase):
    def _new_depth(self):
        return pd.DataFrame([
            {'dt': '2025-09-03T10:00:00Z', 'team': 'BAL', 'pos_abb': 'QB',
             'pos_rank': 1, 'gsis_id': 'LAMAR'},
            {'dt': '2025-09-03T10:00:00Z', 'team': 'BAL', 'pos_abb': 'QB',
             'pos_rank': 2, 'gsis_id': 'BACKUP_QB'},
        ])

    def _schedule(self):
        return pd.DataFrame([
            {'season': 2025, 'week': 1, 'game_type': 'REG', 'gameday': '2025-09-07',
             'gametime': '13:00', 'home_team': 'BAL', 'away_team': 'CLE'},
        ])

    def test_2025_routes_through_new_schema(self):
        with mock.patch.object(starters.nfl, 'import_depth_charts',
                               return_value=self._new_depth()), \
             mock.patch.object(starters.nfl, 'import_schedules',
                               return_value=self._schedule()), \
             mock.patch.object(starters.nfl, 'import_injuries',
                               return_value=pd.DataFrame(
                                   columns=['gsis_id', 'report_status', 'team',
                                            'season', 'week'])):
            out = starters.get_weekly_starters([2025])
        qb = out[(out['team'] == 'BAL') & (out['position'] == 'QB')].iloc[0]
        self.assertEqual(qb['gsis_id'], 'LAMAR')
        self.assertEqual(qb['week'], 1)

    def test_2025_quarantines_on_unexpected_schema(self):
        # An old-shaped frame for a 2025 request -> no pos_abb -> empty (quarantine).
        old_shape = pd.DataFrame([
            {'season': 2025, 'week': 1, 'game_type': 'REG', 'club_code': 'BAL',
             'depth_team': '1', 'position': 'QB', 'gsis_id': 'LAMAR'}])
        with mock.patch.object(starters.nfl, 'import_depth_charts',
                               return_value=old_shape), \
             mock.patch.object(starters.nfl, 'import_schedules',
                               return_value=self._schedule()), \
             mock.patch.object(starters.nfl, 'import_injuries',
                               return_value=pd.DataFrame(
                                   columns=['gsis_id', 'report_status'])):
            out = starters.get_weekly_starters([2025])
        self.assertEqual(len(out), 0)
        self.assertEqual(list(out.columns),
                         ['season', 'week', 'team', 'gsis_id', 'position'])
