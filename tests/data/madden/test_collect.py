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


class TestSeasonRouting(unittest.TestCase):
    def test_game_version_formula(self):
        self.assertEqual(collect.season_to_game_version(2025), 'madden-26')
        self.assertEqual(collect.season_to_game_version(2026), 'madden-27')

    def test_2025_routes_to_launch_and_rosters(self):
        launch_df = pd.DataFrame([{
            'season': 2025, 'full_name': 'Lamar Jackson', 'team': 'BAL',
            'position': 'QB', 'overall': 94, 'weight': 215,
            'power_moves': 40, 'finesse_moves': 35}])
        with mock.patch.object(collect.launch, 'load_madden_launch',
                               return_value=launch_df) as mlaunch, \
             mock.patch.object(collect.ids, 'attach_gsis_id_from_rosters',
                               side_effect=lambda d, s: d.assign(gsis_id='G')) as mbridge, \
             mock.patch.object(collect.ingest, 'load_madden_season') as mold:
            out = collect._player_season(2025)
        mlaunch.assert_called_once_with('madden-26', 2025)
        mbridge.assert_called_once()
        mold.assert_not_called()
        self.assertEqual(out.iloc[0]['gsis_id'], 'G')
        self.assertIn('role', out.columns)

    def test_pre2025_uses_theedgepredictor_path(self):
        old_df = pd.DataFrame([{
            'season': 2023, 'full_name': 'X Y', 'team': 'KC', 'position': 'QB',
            'overall': 99, 'weight': 230, 'power_moves': 20, 'finesse_moves': 20}])
        with mock.patch.object(collect.ingest, 'load_madden_season',
                               return_value=old_df) as mold, \
             mock.patch.object(collect.ids, 'attach_gsis_id',
                               side_effect=lambda d, s: d.assign(gsis_id='G')) as mbridge, \
             mock.patch.object(collect.launch, 'load_madden_launch') as mlaunch:
            collect._player_season(2023)
        mold.assert_called_once_with(2023)
        mbridge.assert_called_once()
        mlaunch.assert_not_called()
