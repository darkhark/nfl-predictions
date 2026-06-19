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
        # single team in synthetic fixture → within-season std is NaN → z-score = 0.0
        self.assertEqual(out.iloc[0]['madden_qb_ovr'], 0.0)
        # every feature column carries the 'madden' token (partition.py contract)
        feat = [c for c in out.columns if c not in ('team', 'season', 'week')]
        self.assertTrue(all('madden' in c for c in feat))

    def test_prev_season_diff_two_seasons(self):
        def _make_players(season, overall):
            return pd.DataFrame([{
                'season': season, 'full_name': 'QB1', 'team': 'KC',
                'position': 'QB', 'overall': overall, 'weight': 220,
                'power_moves': 0, 'finesse_moves': 0,
                'gsis_id': 'QB1', 'role': 'qb', 'side': 'none',
            }])

        def _player_season_side_effect(season):
            if season == 2023:
                return _make_players(2023, 90)
            if season == 2024:
                return _make_players(2024, 96)
            raise ValueError(f"Unexpected season: {season}")

        starters = pd.DataFrame([
            {'season': 2023, 'week': 1, 'team': 'KC', 'gsis_id': 'QB1', 'position': 'QB'},
            {'season': 2024, 'week': 1, 'team': 'KC', 'gsis_id': 'QB1', 'position': 'QB'},
        ])

        with mock.patch.object(collect, '_player_season',
                               side_effect=_player_season_side_effect), \
             mock.patch.object(collect.starters_mod, 'get_weekly_starters',
                               return_value=starters):
            out = collect.get_madden_data([2023, 2024])

        row_2023 = out[(out['season'] == 2023) & (out['team'] == 'KC')].iloc[0]
        row_2024 = out[(out['season'] == 2024) & (out['team'] == 'KC')].iloc[0]

        # single team per season → within-season std is NaN → z-score = 0.0
        self.assertEqual(row_2023['madden_qb_ovr'], 0.0)
        self.assertEqual(row_2024['madden_qb_ovr'], 0.0)
        self.assertTrue(pd.isna(row_2023['madden_qb_ovr_diff_prev_season']),
                        "2023 row should have NaN diff (no 2022 season loaded)")
        self.assertEqual(row_2024['madden_qb_ovr_diff_prev_season'], 6,
                         "2024 diff should be 96 - 90 = 6")
