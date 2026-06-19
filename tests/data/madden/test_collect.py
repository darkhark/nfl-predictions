# tests/data/madden/test_collect.py
import unittest
from unittest import mock
import pandas as pd
from src.data.madden import collect


class TestGetMaddenData(unittest.TestCase):
    def test_assembles_team_week_with_madden_columns(self):
        players_2023 = pd.DataFrame([
            {'season': 2023, 'full_name': 'A', 'team': 'KC', 'position': 'QB',
             'overall': 96, 'weight': 220, 'power_moves': 0, 'finesse_moves': 0,
             'gsis_id': 'QB1', 'role': 'qb', 'side': 'none'}])
        starters = pd.DataFrame([
            {'season': 2023, 'week': 1, 'team': 'KC', 'gsis_id': 'QB1', 'position': 'QB'}])
        with mock.patch.object(collect, '_player_season', return_value=players_2023), \
             mock.patch.object(collect.starters_mod, 'get_weekly_starters',
                               return_value=starters):
            out = collect.get_madden_data([2023])
        self.assertEqual(set(['team', 'season', 'week']).issubset(out.columns), True)
        self.assertIn('madden_qb_ovr', out.columns)
        self.assertIn('madden_qb_ovr_diff_prev', out.columns)
        self.assertEqual(out.iloc[0]['madden_qb_ovr'], 96)
        # every feature column carries the 'madden' token (partition.py contract)
        feat = [c for c in out.columns if c not in ('team', 'season', 'week')]
        self.assertTrue(all('madden' in c for c in feat))
