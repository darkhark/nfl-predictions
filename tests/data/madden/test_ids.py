# tests/data/madden/test_ids.py
import unittest
import pandas as pd
from src.data.madden import ids


class TestNormalizeName(unittest.TestCase):
    def test_strips_case_punct_suffix(self):
        self.assertEqual(ids.normalize_name('A.J. Brown'), 'aj brown')
        self.assertEqual(ids.normalize_name('Michael Pittman Jr.'), 'michael pittman')


class TestAttachGsisId(unittest.TestCase):
    def _madden(self):
        return pd.DataFrame([
            {'season': 2023, 'full_name': 'Patrick Mahomes', 'team': 'KC',
             'position': 'QB'},
            {'season': 2023, 'full_name': 'Nobody Here', 'team': 'KC',
             'position': 'QB'},
        ])

    def _processed(self):
        return pd.DataFrame([
            {'fullname': 'Patrick Mahomes', 'team': 'KC', 'player_id': '00-0033873'},
        ])

    def test_matches_on_name_and_team(self):
        out = ids.attach_gsis_id(self._madden(), 2023, processed=self._processed())
        self.assertEqual(out.iloc[0]['gsis_id'], '00-0033873')

    def test_unmatched_is_na(self):
        out = ids.attach_gsis_id(self._madden(), 2023, processed=self._processed())
        self.assertTrue(pd.isna(out.iloc[1]['gsis_id']))
