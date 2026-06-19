import unittest
from unittest import mock
import pandas as pd
from src.data.madden import starters


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
