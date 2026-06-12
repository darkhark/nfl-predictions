import tempfile
import unittest
from unittest import mock

import pandas as pd

from src.data.play_by_play import collect


def make_play(posteam='AAA', defteam='BBB', season=2023, week=1, season_type='REG',
              play_id=1, game_id='2023_01_AAA_BBB', is_pass=0, is_rush=0,
              down=1, yardline_100=75.0, third_down_converted=0.0, success=0.0,
              epa=0.0, wp=0.5, xpass=None, fixed_drive=1, fixed_drive_result='Punt',
              yards_gained=0.0, run_location=None, run_gap=None,
              pass_location=None, pass_length=None):
    """One synthetic nflfastR play row. 'pass'/'rush' are reserved words as kwargs,
    hence is_pass/is_rush."""
    return {
        'posteam': posteam, 'defteam': defteam, 'season': season, 'week': week,
        'season_type': season_type, 'play_id': play_id, 'game_id': game_id,
        'pass': is_pass, 'rush': is_rush, 'down': down, 'yardline_100': yardline_100,
        'third_down_converted': third_down_converted, 'success': success, 'epa': epa,
        # float('nan') rather than None so the xpass column is float64 like real
        # nflfastR data, not object dtype
        'wp': wp, 'xpass': float('nan') if xpass is None else xpass,
        'fixed_drive': fixed_drive, 'fixed_drive_result': fixed_drive_result,
        'yards_gained': yards_gained, 'run_location': run_location,
        'run_gap': run_gap, 'pass_location': pass_location,
        'pass_length': pass_length,
    }


class TestAssignWpContext(unittest.TestCase):

    def test_thresholds_and_boundaries(self):
        wp = pd.Series([0.5, 0.05, 0.95, 0.951, 0.049, 0.0, 1.0, float('nan')])
        result = collect._assign_wp_context(wp)
        expected = [
            collect.COMPETITIVE,        # 0.5
            collect.COMPETITIVE,        # 0.05 boundary is inclusive
            collect.COMPETITIVE,        # 0.95 boundary is inclusive
            collect.GARBAGE_LEADING,    # 0.951
            collect.GARBAGE_TRAILING,   # 0.049
            collect.GARBAGE_TRAILING,   # 0.0
            collect.GARBAGE_LEADING,    # 1.0
            collect.COMPETITIVE,        # NaN wp defaults to competitive
        ]
        self.assertEqual(list(result), expected)


class TestAggregatePlayComponents(unittest.TestCase):

    def setUp(self):
        plays = pd.DataFrame([
            # competitive: one successful pass, one failed rush (both early downs)
            make_play(play_id=1, is_pass=1, down=1, success=1.0, epa=0.5, wp=0.5, xpass=0.6),
            make_play(play_id=2, is_rush=1, down=2, success=0.0, epa=-0.2, wp=0.5, xpass=0.3),
            # garbage_leading: converted third-down pass, no xpass value
            make_play(play_id=3, is_pass=1, down=3, third_down_converted=1.0,
                      success=1.0, epa=1.0, wp=0.96, xpass=None),
            # not a pass or rush play (e.g. kickoff): must be excluded entirely
            make_play(play_id=4, is_pass=0, is_rush=0, epa=2.0, wp=0.5),
        ])
        self.result = collect._aggregate_play_components(plays)

    def _row(self, context):
        return self.result[self.result[collect.CONTEXT_COL] == context].iloc[0]

    def test_competitive_components(self):
        row = self._row(collect.COMPETITIVE)
        self.assertEqual(row['play_count'], 2)
        self.assertAlmostEqual(row['epa_sum'], 0.3)
        self.assertEqual(row['success_sum'], 1)
        self.assertEqual(row['dropback_count'], 1)
        self.assertAlmostEqual(row['dropback_epa_sum'], 0.5)
        self.assertEqual(row['dropback_success_sum'], 1)
        self.assertEqual(row['rush_count'], 1)
        self.assertAlmostEqual(row['rush_epa_sum'], -0.2)
        self.assertEqual(row['rush_success_sum'], 0)
        self.assertEqual(row['early_down_count'], 2)
        self.assertEqual(row['early_down_success_sum'], 1)
        self.assertEqual(row['third_down_count'], 0)
        self.assertEqual(row['third_down_conversion_sum'], 0)
        self.assertEqual(row['xpass_play_count'], 2)
        # (1 - 0.6) + (0 - 0.3) = 0.1
        self.assertAlmostEqual(row['pass_minus_xpass_sum'], 0.1)

    def test_garbage_leading_components(self):
        row = self._row(collect.GARBAGE_LEADING)
        self.assertEqual(row['play_count'], 1)
        self.assertEqual(row['third_down_count'], 1)
        self.assertEqual(row['third_down_conversion_sum'], 1)
        # xpass was NaN: play contributes to neither PROE component
        self.assertEqual(row['xpass_play_count'], 0)
        self.assertEqual(row['pass_minus_xpass_sum'], 0)

    def test_non_pass_rush_plays_are_excluded(self):
        self.assertEqual(self.result['play_count'].sum(), 3)

    def test_grain_is_team_week_context(self):
        expected_keys = collect.AGGREGATION_KEY_COLUMNS + [collect.CONTEXT_COL]
        for key in expected_keys:
            self.assertIn(key, self.result.columns)
        self.assertFalse(self.result.duplicated(subset=expected_keys).any())


class TestAggregateRedZoneComponents(unittest.TestCase):

    def setUp(self):
        plays = pd.DataFrame([
            # Drive 1: reaches the red zone (min yardline 18), ends in a TD.
            # First play wp 0.5 -> the whole drive counts as competitive even though
            # a later play is in garbage range.
            make_play(play_id=1, fixed_drive=1, yardline_100=45.0, wp=0.50,
                      fixed_drive_result='Touchdown', is_pass=1),
            make_play(play_id=2, fixed_drive=1, yardline_100=18.0, wp=0.96,
                      fixed_drive_result='Touchdown', is_rush=1),
            # Drive 2: never reaches the red zone -> excluded
            make_play(play_id=3, fixed_drive=2, yardline_100=60.0, wp=0.5,
                      fixed_drive_result='Punt', is_pass=1),
            # Drive 3: red zone field goal in garbage_leading (first play wp 0.97)
            make_play(play_id=4, fixed_drive=3, yardline_100=15.0, wp=0.97,
                      fixed_drive_result='Field goal', is_rush=1),
        ])
        self.result = collect._aggregate_red_zone_components(plays)

    def _row(self, context):
        return self.result[self.result[collect.CONTEXT_COL] == context].iloc[0]

    def test_red_zone_drive_counts_by_context(self):
        competitive = self._row(collect.COMPETITIVE)
        self.assertEqual(competitive['red_zone_drive_count'], 1)
        self.assertEqual(competitive['red_zone_td_drive_count'], 1)
        leading = self._row(collect.GARBAGE_LEADING)
        self.assertEqual(leading['red_zone_drive_count'], 1)
        self.assertEqual(leading['red_zone_td_drive_count'], 0)

    def test_non_red_zone_drives_are_excluded(self):
        self.assertEqual(self.result['red_zone_drive_count'].sum(), 2)

    def test_drive_context_uses_first_play_in_play_id_order(self):
        # Same drive 1 rows but shuffled: context must still come from play_id 1 (wp 0.5)
        plays = pd.DataFrame([
            make_play(play_id=2, fixed_drive=1, yardline_100=18.0, wp=0.96,
                      fixed_drive_result='Touchdown', is_rush=1),
            make_play(play_id=1, fixed_drive=1, yardline_100=45.0, wp=0.50,
                      fixed_drive_result='Touchdown', is_pass=1),
        ])
        result = collect._aggregate_red_zone_components(plays)
        self.assertEqual(result[collect.CONTEXT_COL].iloc[0], collect.COMPETITIVE)

    def test_non_scrimmage_rows_do_not_create_red_zone_trips(self):
        # A long touchdown: scrimmage play scores from the 30, then the PAT row sits at
        # the 15 with the same fixed_drive. The PAT must not fake a red-zone trip.
        plays = pd.DataFrame([
            make_play(play_id=1, fixed_drive=1, yardline_100=30.0, wp=0.5,
                      fixed_drive_result='Touchdown', is_pass=1),
            make_play(play_id=2, fixed_drive=1, yardline_100=15.0, wp=0.5,
                      fixed_drive_result='Touchdown'),  # PAT: neither pass nor rush
        ])
        result = collect._aggregate_red_zone_components(plays)
        self.assertTrue(result.empty)

    def test_red_zone_trip_counted_once_despite_special_teams_rows(self):
        # A genuine red-zone touchdown drive still counts exactly once when the PAT row
        # tags along in the same fixed_drive.
        plays = pd.DataFrame([
            make_play(play_id=1, fixed_drive=1, yardline_100=18.0, wp=0.5,
                      fixed_drive_result='Touchdown', is_rush=1),
            make_play(play_id=2, fixed_drive=1, yardline_100=15.0, wp=0.5,
                      fixed_drive_result='Touchdown'),  # PAT: neither pass nor rush
        ])
        result = collect._aggregate_red_zone_components(plays)
        self.assertEqual(result['red_zone_drive_count'].sum(), 1)
        self.assertEqual(result['red_zone_td_drive_count'].sum(), 1)


class TestAggregateSeason(unittest.TestCase):

    def setUp(self):
        plays = pd.DataFrame([
            # 'SD' must come out as 'LAC'; opponent 'OAK' as 'LV'
            make_play(posteam='SD', defteam='OAK', play_id=1, is_pass=1, epa=0.5,
                      success=1.0, wp=0.5, yardline_100=15.0, fixed_drive=1,
                      fixed_drive_result='Touchdown'),
            make_play(posteam='OAK', defteam='SD', play_id=2, is_rush=1, epa=-0.1,
                      wp=0.4, fixed_drive=2),
        ])
        self.result = collect._aggregate_season(plays)

    def test_team_abbreviations_are_normalized(self):
        self.assertEqual(set(self.result['team']), {'LAC', 'LV'})
        self.assertEqual(set(self.result['opp_team']), {'LAC', 'LV'})

    def test_one_row_per_team_game(self):
        self.assertEqual(len(self.result), 2)

    def test_all_component_context_columns_exist_and_missing_contexts_are_zero(self):
        for component in collect.COMPONENT_COLUMNS:
            for context in collect.WP_CONTEXTS:
                self.assertIn(f'{component}_{context}', self.result.columns)
        lac = self.result[self.result['team'] == 'LAC'].iloc[0]
        # LAC had no garbage-time plays: those component cells are 0 (no plays), not NaN
        self.assertEqual(lac['play_count_garbage_leading'], 0)
        self.assertEqual(lac['play_count_garbage_trailing'], 0)

    def test_play_and_drive_components_land_on_the_same_row(self):
        lac = self.result[self.result['team'] == 'LAC'].iloc[0]
        self.assertEqual(lac['play_count_competitive'], 1)
        self.assertEqual(lac['red_zone_drive_count_competitive'], 1)
        self.assertEqual(lac['red_zone_td_drive_count_competitive'], 1)

    def test_missing_required_column_raises(self):
        plays = pd.DataFrame([make_play()]).drop(columns=['epa'])
        with self.assertRaises(ValueError):
            collect._aggregate_season(plays)

    def test_column_order_is_deterministic_across_context_presence(self):
        # pivot_table's column order depends on which contexts appear in the data; the
        # per-season parquet caches must share one schema regardless.
        competitive_only = pd.DataFrame([make_play(is_pass=1, epa=0.1, wp=0.5)])
        with_garbage = pd.DataFrame([
            make_play(play_id=1, is_pass=1, epa=0.1, wp=0.5),
            make_play(play_id=2, is_rush=1, epa=0.2, wp=0.97),
        ])
        self.assertEqual(
            list(collect._aggregate_season(competitive_only).columns),
            list(collect._aggregate_season(with_garbage).columns),
        )


class TestGetPlayByPlayDataCaching(unittest.TestCase):

    def _synthetic_pbp(self):
        return pd.DataFrame([
            make_play(posteam='AAA', defteam='BBB', play_id=1, is_pass=1, epa=0.5, wp=0.5),
            make_play(posteam='BBB', defteam='AAA', play_id=2, is_rush=1, epa=0.1, wp=0.5),
        ])

    def test_download_happens_once_then_cache_is_read(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with mock.patch.object(collect, 'CACHE_DIR', tmp_dir), \
                    mock.patch.object(collect.nfl, 'import_pbp_data',
                                      return_value=self._synthetic_pbp()) as import_mock:
                first = collect.get_play_by_play_data([2023])
                second = collect.get_play_by_play_data([2023])
            import_mock.assert_called_once()
            pd.testing.assert_frame_equal(
                first.sort_index(axis=1), second.sort_index(axis=1)
            )

    def test_refresh_forces_redownload(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with mock.patch.object(collect, 'CACHE_DIR', tmp_dir), \
                    mock.patch.object(collect.nfl, 'import_pbp_data',
                                      return_value=self._synthetic_pbp()) as import_mock:
                collect.get_play_by_play_data([2023])
                collect.get_play_by_play_data([2023], refresh=True)
            self.assertEqual(import_mock.call_count, 2)

    def test_years_type_validation(self):
        with self.assertRaises(TypeError):
            collect.get_play_by_play_data('2023')


def make_component_row(team, opp_team, season=2023, week=1, season_type='REG', **overrides):
    row = {'team': team, 'opp_team': opp_team, 'season': season, 'week': week,
           'season_type': season_type}
    for component in collect.COMPONENT_COLUMNS:
        for context in collect.WP_CONTEXTS:
            row[f'{component}_{context}'] = 0
    row.update(overrides)
    return row


class TestCumulativeRateColumns(unittest.TestCase):

    def setUp(self):
        components = pd.DataFrame([
            # AAA week 1 vs BBB: 10 competitive plays / 5 EPA; 2 leading-garbage plays / 1 EPA
            make_component_row('AAA', 'BBB', week=1,
                               play_count_competitive=10, epa_sum_competitive=5.0,
                               play_count_garbage_leading=2, epa_sum_garbage_leading=1.0),
            make_component_row('BBB', 'AAA', week=1,
                               play_count_competitive=8, epa_sum_competitive=-2.0),
            # AAA week 2 vs CCC: efficiency drops; first trailing-garbage snaps appear.
            # Play count differs from week 1 on purpose so ratio-of-cumsums and
            # mean-of-weekly-rates give different answers and the test can tell them apart.
            make_component_row('AAA', 'CCC', week=2,
                               play_count_competitive=20, epa_sum_competitive=1.0,
                               play_count_garbage_trailing=4, epa_sum_garbage_trailing=-1.0),
            make_component_row('CCC', 'AAA', week=2,
                               play_count_competitive=12, epa_sum_competitive=0.0),
        ]).reset_index(drop=True)
        components = collect._add_game_count_columns(components).reset_index(drop=True)
        self.result = collect._add_cumulative_rate_columns(components)

    def _value(self, team, week, col):
        row = self.result[(self.result['team'] == team) & (self.result['week'] == week)]
        return row[col].values[0]

    def test_cumulative_rate_is_ratio_of_cumulative_sums(self):
        # Week 1: 5/10 = 0.5. Week 2: (5+1)/(10+20) = 0.2.
        # A wrong mean-of-weekly-rates implementation would give (0.5 + 0.05)/2 = 0.275.
        self.assertAlmostEqual(
            self._value('AAA', 1, 'off_epa_per_play_competitive_cumulative_average'), 0.5)
        self.assertAlmostEqual(
            self._value('AAA', 2, 'off_epa_per_play_competitive_cumulative_average'), 0.2)

    def test_zero_denominator_is_nan_then_recovers(self):
        # AAA had no trailing-garbage snaps in week 1 -> NaN, not 0
        self.assertTrue(pd.isna(
            self._value('AAA', 1, 'off_epa_per_play_garbage_trailing_cumulative_average')))
        # Week 2: cumulative = (0 + -1.0) / (0 + 4) = -0.25
        self.assertAlmostEqual(
            self._value('AAA', 2, 'off_epa_per_play_garbage_trailing_cumulative_average'), -0.25)

    def test_defense_mirrors_opponent_offense_on_first_game(self):
        # def_opp_* on AAA's week-1 row is BBB's defense; BBB's only game so far is this
        # one, so it equals AAA's own offense value from the same game.
        self.assertAlmostEqual(
            self._value('AAA', 1, 'def_opp_epa_per_play_competitive_cumulative_average'), 0.5)

    def test_defense_context_labels_are_swapped(self):
        # AAA's leading-garbage offense (1.0 EPA / 2 plays) is BBB's trailing-garbage
        # defense: the defense was on the field while ITS team was likely losing.
        self.assertAlmostEqual(
            self._value('AAA', 1, 'def_opp_epa_per_play_garbage_trailing_cumulative_average'), 0.5)
        self.assertTrue(pd.isna(
            self._value('AAA', 1, 'def_opp_epa_per_play_garbage_leading_cumulative_average')))

    def test_defense_accumulates_across_opponent_games(self):
        # CCC's week-2 def_opp row tracks AAA's defense. AAA defended BBB week 1
        # (8 plays, -2 EPA) and CCC week 2 (12 plays, 0 EPA): (-2+0)/(8+12) = -0.1
        self.assertAlmostEqual(
            self._value('CCC', 2, 'def_opp_epa_per_play_competitive_cumulative_average'), -0.1)

    def test_all_rate_metric_columns_exist(self):
        for metric, _, _ in collect.RATE_METRICS:
            for context in collect.WP_CONTEXTS:
                self.assertIn(f'off_{metric}_{context}_cumulative_average', self.result.columns)
                self.assertIn(f'def_opp_{metric}_{context}_cumulative_average', self.result.columns)

    def test_non_unique_index_is_rejected(self):
        # A duplicated index would silently scramble def_opp values during the
        # index-aligned concat; the guard must fail fast instead.
        components = pd.DataFrame([
            make_component_row('AAA', 'BBB', week=1),
            make_component_row('BBB', 'AAA', week=1),
        ])
        components = collect._add_game_count_columns(components).reset_index(drop=True)
        components.index = [0, 0]
        with self.assertRaises(ValueError):
            collect._add_cumulative_rate_columns(components)


class TestGetPlayByPlayFeatures(unittest.TestCase):

    def _synthetic_pbp(self):
        # Two games in one week: four teams so ranks span 1..4
        return pd.DataFrame([
            make_play(posteam='AAA', defteam='BBB', game_id='g1', play_id=1,
                      is_pass=1, epa=1.0, success=1.0, wp=0.5),
            make_play(posteam='BBB', defteam='AAA', game_id='g1', play_id=2,
                      is_rush=1, epa=0.5, success=1.0, wp=0.5),
            make_play(posteam='CCC', defteam='DDD', game_id='g2', play_id=1,
                      is_pass=1, epa=-0.5, success=0.0, wp=0.5),
            make_play(posteam='DDD', defteam='CCC', game_id='g2', play_id=2,
                      is_rush=1, epa=-1.0, success=0.0, wp=0.5),
        ])

    def _features(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with mock.patch.object(collect, 'CACHE_DIR', tmp_dir), \
                    mock.patch.object(collect.nfl, 'import_pbp_data',
                                      return_value=self._synthetic_pbp()):
                return collect.get_play_by_play_features([2023])

    def test_feature_count_and_naming(self):
        features = self._features()
        feature_cols = [col for col in features.columns
                        if col.startswith('off_') or col.startswith('def_opp_')]
        # 10 metrics x 3 contexts x 2 sides x 3 column kinds (rate, rank, rank_change)
        self.assertEqual(len(feature_cols), 180)
        self.assertEqual(list(features.columns[:3]), ['team', 'season', 'week'])

    def test_offense_rank_one_is_best_epa(self):
        features = self._features()
        rank_col = 'off_epa_per_play_competitive_cumulative_average_rank'
        best = features[features[rank_col] == 1]
        self.assertEqual(best['team'].values[0], 'AAA')
        worst = features[features[rank_col] == 4]
        self.assertEqual(worst['team'].values[0], 'DDD')

    def test_defense_rank_one_allows_least_epa(self):
        features = self._features()
        # def_opp on a row describes that row's opponent. CCC's defense allowed DDD's
        # -1.0 EPA/play (least allowed -> rank 1) and CCC is the opponent on DDD's row.
        rank_col = 'def_opp_epa_per_play_competitive_cumulative_average_rank'
        best = features[features[rank_col] == 1]
        self.assertEqual(best['team'].values[0], 'DDD')

    def test_no_component_columns_leak_into_output(self):
        features = self._features()
        for component in collect.COMPONENT_COLUMNS:
            for context in collect.WP_CONTEXTS:
                self.assertNotIn(f'{component}_{context}', features.columns)


class TestDirectionalConstants(unittest.TestCase):

    def test_thirteen_buckets(self):
        self.assertEqual(len(collect.RUN_BUCKETS), 7)
        self.assertEqual(len(collect.PASS_BUCKETS), 6)
        self.assertEqual(
            collect.DIRECTIONAL_BUCKETS, collect.RUN_BUCKETS + collect.PASS_BUCKETS)

    def test_directional_generated_lists(self):
        # 3 components per bucket and 2 metrics per bucket, generated from the
        # bucket lists; wired into the aggregate lists in the next task
        self.assertEqual(len(collect.DIRECTIONAL_COMPONENT_COLUMNS), 39)
        self.assertEqual(len(collect.DIRECTIONAL_RATE_METRICS), 26)
        self.assertIn('run_left_end_attempt_count', collect.DIRECTIONAL_COMPONENT_COLUMNS)
        self.assertIn(
            ('pass_deep_right_explosive_rate', 'pass_deep_right_explosive_count',
             'pass_deep_right_attempt_count'),
            collect.DIRECTIONAL_RATE_METRICS)

    def test_directional_required_columns(self):
        for column in ('yards_gained', 'run_location', 'run_gap',
                       'pass_location', 'pass_length'):
            self.assertIn(column, collect.REQUIRED_PBP_COLUMNS)


if __name__ == '__main__':
    unittest.main()
