import unittest
import pandas as pd
from src.data.madden import launch
from src.data.madden.ingest import OUTPUT_COLUMNS


def _players():
    return [
        {'first_name': 'Lamar', 'last_name': 'Jackson', 'position': 'QB',
         'team_id': 26, 'rating_overall': 94, 'weight': 215,
         'rating_power_moves': 40, 'rating_finesse_moves': 35},
        {'first_name': 'Kyle', 'last_name': 'Juszczyk', 'position': 'FB',
         'team_id': 25, 'rating_overall': 88, 'weight': 235,
         'rating_power_moves': 30, 'rating_finesse_moves': 35},
    ]


def _teams():
    return [
        {'id': 26, 'name': 'Ravens', 'acronym': 'BAL'},
        {'id': 25, 'name': '49ers', 'acronym': 'SF'},
    ]


class TestParseLaunchRatings(unittest.TestCase):
    def test_maps_to_output_columns(self):
        out = launch.parse_launch_ratings(_players(), _teams(), 2025)
        self.assertEqual(list(out.columns), OUTPUT_COLUMNS)
        lamar = out[out['full_name'] == 'Lamar Jackson'].iloc[0]
        self.assertEqual(lamar['team'], 'BAL')
        self.assertEqual(lamar['position'], 'QB')
        self.assertEqual(lamar['overall'], 94)
        self.assertEqual(lamar['power_moves'], 40)
        self.assertEqual(lamar['finesse_moves'], 35)
        self.assertEqual(lamar['season'], 2025)

    def test_empty_players_returns_empty_contract_frame(self):
        out = launch.parse_launch_ratings([], _teams(), 2025)
        self.assertEqual(list(out.columns), OUTPUT_COLUMNS)
        self.assertEqual(len(out), 0)

    def test_unknown_team_id_yields_na_team(self):
        players = [{'first_name': 'A', 'last_name': 'B', 'position': 'WR',
                    'team_id': 999, 'rating_overall': 70, 'weight': 190,
                    'rating_power_moves': 10, 'rating_finesse_moves': 60}]
        out = launch.parse_launch_ratings(players, _teams(), 2025)
        self.assertTrue(pd.isna(out.iloc[0]['team']))


class TestSelectLaunchIteration(unittest.TestCase):
    def _iterations(self):
        return [
            {'id': 0, 'label': 'Launch', 'release_date': '2025-08-14', 'active': True},
            {'id': 1, 'label': 'Week 1', 'release_date': '2025-09-04', 'active': True},
        ]

    def test_selects_launch(self):
        self.assertEqual(launch.select_launch_iteration(self._iterations())['id'], 0)

    def test_raises_when_no_launch(self):
        with self.assertRaises(ValueError):
            launch.select_launch_iteration([{'id': 1, 'label': 'Week 1'}])

    def test_raises_when_multiple_launch(self):
        dup = [{'id': 0, 'label': 'Launch'}, {'id': 5, 'label': 'launch'}]
        with self.assertRaises(ValueError):
            launch.select_launch_iteration(dup)
