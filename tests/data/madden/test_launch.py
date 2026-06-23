import unittest
import pandas as pd
from unittest import mock
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


class TestLoadMaddenLaunch(unittest.TestCase):
    def _fake_cdn(self):
        store = {
            'https://cdn.test/madden-26/json/iterations.json': [
                {'id': 0, 'label': 'Launch', 'active': True},
                {'id': 1, 'label': 'Week 1', 'active': True},
            ],
            'https://cdn.test/madden-26/json/iterations/0/players.json': _players(),
            'https://cdn.test/madden-26/json/iterations/0/teams.json': _teams(),
        }
        return lambda url: store[url]

    def test_loads_launch_iteration(self):
        with mock.patch.object(launch, '_cdn_json', side_effect=self._fake_cdn()):
            out = launch.load_madden_launch('madden-26', 2025, cdn_base='https://cdn.test')
        self.assertEqual(set(out['full_name']), {'Lamar Jackson', 'Kyle Juszczyk'})
        self.assertEqual(out[out['full_name'] == 'Lamar Jackson'].iloc[0]['team'], 'BAL')

    def test_network_failure_returns_empty(self):
        with mock.patch.object(launch, '_cdn_json', side_effect=OSError('boom')):
            out = launch.load_madden_launch('madden-26', 2025, cdn_base='https://cdn.test')
        self.assertEqual(list(out.columns), OUTPUT_COLUMNS)
        self.assertEqual(len(out), 0)

    def test_missing_base_returns_empty(self):
        with mock.patch.dict('os.environ', {}, clear=True):
            out = launch.load_madden_launch('madden-26', 2025, cdn_base=None)
        self.assertEqual(len(out), 0)
