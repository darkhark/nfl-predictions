# tests/data/madden/test_ingest.py
import unittest
from unittest import mock
import pandas as pd
from src.data.madden import ingest


class TestIsNestedSchema(unittest.TestCase):
    def test_nested_detected_by_stats_prefix(self):
        self.assertTrue(ingest.is_nested_schema(
            ['firstName', 'lastName', 'stats/overall/value', 'weight']))

    def test_classic_not_nested(self):
        self.assertFalse(ingest.is_nested_schema(
            ['Team', 'Name', 'Position', 'Overall', 'Power Moves']))


class TestNormalizeClassic(unittest.TestCase):
    def _classic(self):
        # Column order mirrors raw/2018.csv (Name col, Overall, Power/Finesse Moves, Weight)
        return pd.DataFrame([
            {'Team': 'OAK', 'Name': 'Khalil Mack', 'Position': 'LOLB',
             'Overall': 96, 'Power Moves': 90, 'Finesse Moves': 91, 'Weight': 252},
        ])

    def test_classic_maps_and_normalizes_team(self):
        out = ingest.normalize_madden_frame(self._classic(), 2018)
        row = out.iloc[0]
        self.assertEqual(list(out.columns), ['season', 'full_name', 'team', 'position',
                                             'overall', 'weight', 'power_moves',
                                             'finesse_moves'])
        self.assertEqual(row['team'], 'LV')          # OAK -> LV via TEAM_ABBR_MAPPINGS
        self.assertEqual(row['full_name'], 'Khalil Mack')
        self.assertEqual(row['position'], 'LOLB')
        self.assertEqual(row['overall'], 96)
        self.assertEqual(row['power_moves'], 90)
        self.assertEqual(row['season'], 2018)


class TestNormalizeNested(unittest.TestCase):
    def _nested(self):
        return pd.DataFrame([
            {'firstName': 'Maxx', 'lastName': 'Crosby', 'team': 'Raiders',
             'Position': 'RE', 'stats/overall/value': 89, 'weight': 255,
             'stats/powerMoves/value': 88, 'stats/finesseMoves/value': 90},
        ])

    def test_nested_builds_full_name_and_maps_stats(self):
        out = ingest.normalize_madden_frame(self._nested(), 2024)
        row = out.iloc[0]
        self.assertEqual(row['full_name'], 'Maxx Crosby')
        self.assertEqual(row['overall'], 89)
        self.assertEqual(row['finesse_moves'], 90)
        self.assertEqual(row['position'], 'RE')


class TestPositionQualityFlag(unittest.TestCase):
    def test_normalize_tolerates_missing_position(self):
        # 2025 raw has `position_short_label == "False"` for ~88% of rows; once renamed
        # to `position`, normalization must not crash and must leave those as-is for the
        # downstream depth-chart backfill (Plan 2).
        df = pd.DataFrame([{'Team': 'KC', 'Name': 'Patrick Mahomes',
                            'Position': 'False', 'Overall': 99, 'Weight': 225,
                            'Power Moves': 50, 'Finesse Moves': 60}])
        out = ingest.normalize_madden_frame(df, 2025)
        self.assertEqual(out.iloc[0]['position'], 'False')
        self.assertEqual(out.iloc[0]['overall'], 99)
