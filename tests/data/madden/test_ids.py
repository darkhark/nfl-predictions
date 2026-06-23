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


class TestFallbackByPosition(unittest.TestCase):
    """Fallback: when name+team misses, try name+coarse_position (no team)."""

    def _madden_traded(self):
        """Player has traded teams in Madden vs processed — team mismatch."""
        return pd.DataFrame([
            {'season': 2023, 'full_name': 'La\'el Collins', 'team': 'CIN',
             'position': 'RT'},
        ])

    def _processed_old_team(self):
        """processed still has player on old team."""
        return pd.DataFrame([
            {'fullname': 'La\'el Collins', 'team': 'DAL', 'player_id': '00-0031534',
             'position': 'OL'},
        ])

    def test_fallback_resolves_traded_player(self):
        """Name+team misses but name+coarse_position hits → gsis_id resolved."""
        out = ids.attach_gsis_id(self._madden_traded(), 2023,
                                 processed=self._processed_old_team())
        self.assertEqual(out.iloc[0]['gsis_id'], '00-0031534')

    def test_ambiguous_fallback_yields_na(self):
        """If name+coarse_position maps to >1 distinct gsis, result must be NA."""
        madden = pd.DataFrame([
            {'season': 2023, 'full_name': 'Lamar Jackson', 'team': 'KC',
             'position': 'CB'},
        ])
        # Two different players named Lamar Jackson at DB in processed
        processed = pd.DataFrame([
            {'fullname': 'Lamar Jackson', 'team': 'BAL', 'player_id': '00-0034796',
             'position': 'DB'},
            {'fullname': 'Lamar Jackson', 'team': 'CAR', 'player_id': '00-0036152',
             'position': 'DB'},
        ])
        out = ids.attach_gsis_id(madden, 2023, processed=processed)
        self.assertTrue(pd.isna(out.iloc[0]['gsis_id']),
                        "Ambiguous name+position must not produce a false match")

    def test_primary_match_takes_precedence(self):
        """Primary (name+team) match is used even if fallback would also hit."""
        madden = pd.DataFrame([
            {'season': 2023, 'full_name': 'Patrick Mahomes', 'team': 'KC',
             'position': 'QB'},
        ])
        processed = pd.DataFrame([
            {'fullname': 'Patrick Mahomes', 'team': 'KC', 'player_id': 'primary-id',
             'position': 'QB'},
            {'fullname': 'Patrick Mahomes', 'team': 'DAL', 'player_id': 'fallback-id',
             'position': 'QB'},
        ])
        out = ids.attach_gsis_id(madden, 2023, processed=processed)
        self.assertEqual(out.iloc[0]['gsis_id'], 'primary-id')

    def test_no_gsis_id_col_left_after_drop(self):
        """Internal _norm_name column must not leak into output."""
        madden = pd.DataFrame([
            {'season': 2023, 'full_name': 'Patrick Mahomes', 'team': 'KC',
             'position': 'QB'},
        ])
        processed = pd.DataFrame([
            {'fullname': 'Patrick Mahomes', 'team': 'KC', 'player_id': '00-0033873',
             'position': 'QB'},
        ])
        out = ids.attach_gsis_id(madden, 2023, processed=processed)
        self.assertNotIn('_norm_name', out.columns)


class TestAttachGsisFromRosters(unittest.TestCase):
    def _rosters(self):
        return pd.DataFrame([
            {'player_name': 'Lamar Jackson', 'player_id': '00-0034796',
             'team': 'BAL', 'position': 'QB'},
            {'player_name': 'Josh Allen', 'player_id': '00-0034857',
             'team': 'BUF', 'position': 'QB'},
            {'player_name': 'Josh Allen', 'player_id': '00-0035000',
             'team': 'JAX', 'position': 'LB'},  # name collision -> ambiguous
        ])

    def test_primary_name_team_match(self):
        df = pd.DataFrame([{'full_name': 'Lamar Jackson', 'team': 'BAL', 'position': 'QB'}])
        out = ids.attach_gsis_id_from_rosters(df, 2025, rosters=self._rosters())
        self.assertEqual(out.iloc[0]['gsis_id'], '00-0034796')

    def test_team_match_disambiguates_name_collision(self):
        df = pd.DataFrame([{'full_name': 'Josh Allen', 'team': 'BUF', 'position': 'QB'}])
        out = ids.attach_gsis_id_from_rosters(df, 2025, rosters=self._rosters())
        self.assertEqual(out.iloc[0]['gsis_id'], '00-0034857')

    def test_unmatched_team_falls_back_to_name_only_when_unambiguous(self):
        df = pd.DataFrame([{'full_name': 'Lamar Jackson', 'team': 'XXX', 'position': 'QB'}])
        out = ids.attach_gsis_id_from_rosters(df, 2025, rosters=self._rosters())
        self.assertEqual(out.iloc[0]['gsis_id'], '00-0034796')

    def test_ambiguous_name_only_yields_na(self):
        df = pd.DataFrame([{'full_name': 'Josh Allen', 'team': 'XXX', 'position': 'QB'}])
        out = ids.attach_gsis_id_from_rosters(df, 2025, rosters=self._rosters())
        self.assertTrue(pd.isna(out.iloc[0]['gsis_id']))
