import os
import unittest

import pandas as pd

from data_science_utilities.feature_groups.partition import (
    CONTENT_FAMILIES,
    META_ID_COLUMNS,
    TARGET,
    content_family,
    form_tags,
    is_rank_only_kept,
    partition_features,
)

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')
)
XGB_FEATURES_CSV = os.path.join(
    REPO_ROOT, 'data', 'predict_games', 'model_features_in', 'xgb_features_list.csv'
)


class TestContentFamily(unittest.TestCase):
    """One representative column per family; the bare-metric rule must place each
    in exactly the family documented by the data-assembly source."""

    def test_context_rest_members(self):
        for col in ['week', 'div_game', 'indoor', 'is_home_target',
                    'target_days_since_previous_game', 'opp_game_count']:
            self.assertEqual(content_family(col), 'context_rest')

    def test_market_members(self):
        for col in ['away_moneyline', 'home_moneyline', 'spread_line', 'over_odds']:
            self.assertEqual(content_family(col), 'market')

    def test_schedule_points_uses_reversed_prefix(self):
        self.assertEqual(
            content_family('target_off_cumulative_avg_score_rank'), 'schedule_points')
        self.assertEqual(
            content_family('opp_def_cumulative_avg_points_allowed_rank'),
            'schedule_points')

    def test_box_score_has_no_wp_context(self):
        for col in ['off_target_passing_yards',
                    'off_target_rushing_yards_cumulative_average',
                    'def_opp_passing_epa_cumulative_average_rank']:
            self.assertEqual(content_family(col), 'box_score')

    def test_pbp_phase1_efficiency(self):
        self.assertEqual(
            content_family('off_target_epa_per_play_competitive_cumulative_average_rank'),
            'pbp_phase1')
        # pass_success_rate starts with 'pass' but is a Phase-1 efficiency token,
        # NOT a Phase-2 directional bucket
        self.assertEqual(
            content_family(
                'off_target_pass_success_rate_competitive_cumulative_average_rank'),
            'pbp_phase1')

    def test_pbp_phase2_requires_directional_suffix(self):
        self.assertEqual(
            content_family(
                'off_target_run_left_tackle_explosive_rate_competitive'
                '_cumulative_average_rank'),
            'pbp_phase2_directional')
        self.assertEqual(
            content_family(
                'def_opp_pass_short_left_yards_per_attempt_garbage_trailing'
                '_cumulative_average_rank'),
            'pbp_phase2_directional')

    def test_pbp_phase3_trench_luck_pace(self):
        for col in ['off_target_sack_rate_competitive_cumulative_average_rank',
                    'off_opp_cpoe_competitive_cumulative_average_rank',
                    'off_target_seconds_per_play_competitive_ewma_average_rank']:
            self.assertEqual(content_family(col), 'pbp_phase3')

    def test_situational_playcall(self):
        for col in [
            'off_target_down1_pass_rate_competitive_cumulative_average_rank',
            'off_target_down3_short_conversion_rate_pass_competitive'
            '_cumulative_average_rank',
            'off_opp_goalToGo_pass_rate_competitive_cumulative_average_rank',
        ]:
            self.assertEqual(content_family(col), 'situational_playcall')

    def test_snap_share_classified_before_box(self):
        self.assertEqual(
            content_family('off_target_snap_share_competitive_cumulative_average_rank'),
            'snap_share')

    def test_unclassifiable_column_raises(self):
        with self.assertRaises(ValueError):
            content_family('some_totally_unknown_column')

    def test_madden_columns_classify_as_madden_ratings(self):
        for col in ['target_madden_qb_ovr', 'opp_madden_edge_ovr',
                    'target_madden_lt_ovr_diff_prev', 'madden_matchup_pass_pro',
                    'opp_madden_safety_ovr_diff_prev_season']:
            self.assertEqual(content_family(col), 'madden_ratings')


class TestFormTags(unittest.TestCase):

    def test_cumulative_rank(self):
        tags = form_tags('def_opp_passing_epa_cumulative_average_rank')
        self.assertEqual(tags['temporal'], 'cumulative')
        self.assertEqual(tags['state'], 'rank')
        self.assertFalse(tags['is_recency'])
        self.assertFalse(tags['is_aggregated_value'])

    def test_rank_change_checked_before_rank(self):
        tags = form_tags(
            'def_target_snap_share_competitive_ewma_average_rank_change')
        self.assertEqual(tags['state'], 'rank_change')

    def test_ewma_value_is_recency_and_aggregated(self):
        tags = form_tags('off_target_passing_yards_ewma_average')
        self.assertEqual(tags['temporal'], 'ewma')
        self.assertEqual(tags['state'], 'value')
        self.assertTrue(tags['is_recency'])
        self.assertTrue(tags['is_aggregated_value'])

    def test_rolling_rank_is_recency_but_not_aggregated_value(self):
        tags = form_tags(
            'off_target_run_middle_yards_per_attempt_competitive_rolling_average_rank')
        self.assertTrue(tags['is_recency'])
        self.assertFalse(tags['is_aggregated_value'])

    def test_raw_per_game_value(self):
        tags = form_tags('off_target_passing_yards')
        self.assertEqual(tags['temporal'], 'raw')
        self.assertEqual(tags['state'], 'value')
        self.assertFalse(tags['is_recency'])
        self.assertFalse(tags['is_aggregated_value'])


class TestRankOnlyFilter(unittest.TestCase):
    """Mirrors rfe.ipynb: drop any column containing cumulative/ewma/rolling
    UNLESS it ends in _rank or _rank_change."""

    def test_keeps_ranks_and_rank_changes(self):
        self.assertTrue(is_rank_only_kept('def_opp_passing_epa_cumulative_average_rank'))
        self.assertTrue(is_rank_only_kept(
            'def_target_snap_share_competitive_ewma_average_rank_change'))

    def test_drops_aggregated_values(self):
        self.assertFalse(is_rank_only_kept('off_target_passing_yards_ewma_average'))
        self.assertFalse(is_rank_only_kept('target_off_cumulative_avg_score'))

    def test_keeps_raw_values_without_markers(self):
        self.assertTrue(is_rank_only_kept('off_target_passing_yards'))
        self.assertTrue(is_rank_only_kept('week'))


class TestPartitionFeatures(unittest.TestCase):

    def test_groups_by_content_family_and_excludes_meta_and_target(self):
        features = [
            'off_target_passing_yards',
            'def_opp_passing_epa_cumulative_average_rank',
            'off_target_sack_rate_competitive_cumulative_average_rank',
            'week',
            'game_id',          # meta -> excluded
            'target_win',       # target -> excluded
        ]
        groups = partition_features(features)
        self.assertNotIn('game_id', [c for cols in groups.values() for c in cols])
        self.assertNotIn('target_win', [c for cols in groups.values() for c in cols])
        self.assertEqual(sorted(groups['box_score']),
                         ['def_opp_passing_epa_cumulative_average_rank',
                          'off_target_passing_yards'])
        self.assertEqual(groups['pbp_phase3'],
                         ['off_target_sack_rate_competitive_cumulative_average_rank'])
        self.assertEqual(groups['context_rest'], ['week'])

    def test_rank_only_drops_aggregated_values(self):
        features = [
            'off_target_passing_yards_cumulative_average',        # dropped
            'off_target_passing_yards_cumulative_average_rank',   # kept
        ]
        groups = partition_features(features, rank_only=True)
        self.assertEqual(groups['box_score'],
                         ['off_target_passing_yards_cumulative_average_rank'])

    def test_empty_families_are_omitted(self):
        groups = partition_features(['week'])
        self.assertNotIn('market', groups)
        self.assertEqual(list(groups), ['context_rest'])

    def test_target_constant_and_meta_set(self):
        self.assertEqual(TARGET, 'target_win')
        self.assertIn('game_id', META_ID_COLUMNS)
        self.assertIn('season', META_ID_COLUMNS)


@unittest.skipUnless(os.path.exists(XGB_FEATURES_CSV),
                     'xgb_features_list.csv (real candidate pool) not present')
class TestPartitionOverRealFeatureList(unittest.TestCase):
    """The strongest correctness anchor: every one of the ~9,297 real candidate
    features must classify into exactly one content family, with no unknowns and
    no collisions."""

    @classmethod
    def setUpClass(cls):
        # The raw candidate CSV includes the target_win row (the assembly EXCLUDE
        # list omits the target); real model features are everything else.
        cls.rows = list(pd.read_csv(XGB_FEATURES_CSV)['feature'])
        cls.features = [c for c in cls.rows
                        if c != TARGET and c not in META_ID_COLUMNS]

    def test_only_known_meta_rows_are_unclassifiable(self):
        # Every row that is NOT a model feature classifies; the only non-features
        # present in the candidate CSV are the target (and, defensively, meta ids).
        unknown = []
        for col in self.features:
            try:
                content_family(col)
            except ValueError:
                unknown.append(col)
        self.assertEqual(unknown, [], f'{len(unknown)} unclassified, e.g. {unknown[:5]}')

    def test_partition_is_disjoint_and_covers_all_features(self):
        groups = partition_features(self.rows)
        flat = [c for cols in groups.values() for c in cols]
        self.assertEqual(len(flat), len(set(flat)), 'a feature landed in 2+ families')
        self.assertEqual(set(flat), set(self.features))

    def test_core_families_are_non_empty(self):
        groups = partition_features(self.features)
        for family in ['box_score', 'pbp_phase1', 'pbp_phase2_directional',
                       'pbp_phase3', 'situational_playcall', 'snap_share',
                       'schedule_points', 'context_rest']:
            self.assertGreater(len(groups.get(family, [])), 0,
                               f'{family} unexpectedly empty')

    def test_content_families_constant_lists_known_groups(self):
        self.assertIn('box_score', CONTENT_FAMILIES)
        self.assertIn('pbp_phase2_directional', CONTENT_FAMILIES)


if __name__ == '__main__':
    unittest.main()
