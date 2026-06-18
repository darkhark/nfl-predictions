import unittest

import numpy as np
import pandas as pd

from data_science_utilities.feature_groups.scorers import (
    DEFAULT_XGB_PARAMS,
    make_bart_fold_scorer,
    make_xgb_fold_scorer,
    score_predictions,
    season_rolling_origin_folds,
    to_model_matrix,
)


class TestScorePredictions(unittest.TestCase):

    def test_roc_auc_perfect_separation(self):
        y = np.array([0, 0, 1, 1])
        proba = np.array([0.1, 0.2, 0.8, 0.9])
        self.assertAlmostEqual(score_predictions(y, proba, 'roc_auc'), 1.0)

    def test_brier_is_mean_squared_error(self):
        y = np.array([1, 0])
        proba = np.array([0.75, 0.25])
        self.assertAlmostEqual(score_predictions(y, proba, 'brier'), 0.0625)

    def test_unknown_metric_raises(self):
        with self.assertRaises(ValueError):
            score_predictions(np.array([0, 1]), np.array([0.4, 0.6]), 'nonsense')


class TestToModelMatrix(unittest.TestCase):

    def test_fills_nan_with_sentinel_and_selects_columns_in_order(self):
        df = pd.DataFrame({'a': [1.0, np.nan], 'b': [3.0, 4.0], 'c': [5.0, 6.0]})
        matrix = to_model_matrix(df, ['b', 'a'], sentinel=-100.0)
        np.testing.assert_array_equal(matrix, np.array([[3.0, 1.0], [4.0, -100.0]]))

    def test_raises_if_nan_remains(self):
        df = pd.DataFrame({'a': [np.nan]})
        # a NaN sentinel would leave NaN -> must fail loudly (BART would crash)
        with self.assertRaises(ValueError):
            to_model_matrix(df, ['a'], sentinel=np.nan)

    def test_raises_if_a_real_value_collides_with_the_sentinel(self):
        # a genuine (non-NaN) -100 must not be confused with the missing-marker, or
        # the trees can no longer split the missing region off cleanly
        df = pd.DataFrame({'a': [1.0, -100.0]})
        with self.assertRaises(ValueError):
            to_model_matrix(df, ['a'], sentinel=-100.0)


class TestSeasonRollingOriginFolds(unittest.TestCase):

    def _frame(self):
        # two perspective rows per game, both sharing the same season
        rows = []
        for season in range(2010, 2021):
            for game in range(4):
                gid = f'{season}_{game}'
                rows.append({'season': season, 'game_id': gid, 'perspective': 0})
                rows.append({'season': season, 'game_id': gid, 'perspective': 1})
        return pd.DataFrame(rows)

    def test_test_block_is_exactly_the_test_season(self):
        df = self._frame()
        folds = season_rolling_origin_folds(
            df, first_test_season=2015, last_test_season=2018, valid_years=2)
        test_seasons = [f['test_season'] for f in folds]
        self.assertEqual(test_seasons, [2015, 2016, 2017, 2018])
        for fold in folds:
            self.assertEqual(set(df.loc[fold['test'], 'season']), {fold['test_season']})

    def test_validation_window_precedes_test_and_train_precedes_validation(self):
        df = self._frame()
        fold = season_rolling_origin_folds(
            df, first_test_season=2016, last_test_season=2016, valid_years=2)[0]
        self.assertEqual(set(df.loc[fold['valid'], 'season']), {2014, 2015})
        self.assertTrue((df.loc[fold['train'], 'season'] < 2014).all())

    def test_no_game_leaks_across_train_valid_test(self):
        df = self._frame()
        for fold in season_rolling_origin_folds(
                df, first_test_season=2015, last_test_season=2018, valid_years=2):
            train_games = set(df.loc[fold['train'], 'game_id'])
            valid_games = set(df.loc[fold['valid'], 'game_id'])
            test_games = set(df.loc[fold['test'], 'game_id'])
            self.assertTrue(train_games.isdisjoint(test_games))
            self.assertTrue(valid_games.isdisjoint(test_games))
            self.assertTrue(train_games.isdisjoint(valid_games))

    def test_skips_folds_without_enough_training_history(self):
        df = self._frame()  # seasons start at 2010
        folds = season_rolling_origin_folds(
            df, first_test_season=2010, last_test_season=2016,
            valid_years=2, min_train_seasons=3)
        # season s needs >=3 seasons before its valid window [s-2, s-1]; with data
        # starting 2010 the first usable test season is 2015 (valid 2013-2014,
        # train 2010-2012). Assert the exact boundary, not a vacuous superset.
        self.assertEqual([f['test_season'] for f in folds], [2015, 2016])


class TestMakeXgbFoldScorer(unittest.TestCase):
    """A fast real XGBoost smoke test: a subset containing the informative feature
    must out-score the base-only (noise) subset."""

    def _split(self):
        rng = np.random.default_rng(32)
        n = 900
        signal = rng.normal(size=n)
        noise = rng.normal(size=n)
        target = (signal + 0.25 * rng.normal(size=n) > 0).astype(int)
        df = pd.DataFrame({'signal': signal, 'noise': noise, 'target_win': target,
                           'season': np.repeat([1, 2, 3], n // 3)})
        return df[df.season == 1], df[df.season == 2], df[df.season == 3]

    def test_informative_feature_beats_base_only(self):
        train_df, valid_df, test_df = self._split()
        scorer = make_xgb_fold_scorer(
            train_df, valid_df, test_df, target='target_win', metric='roc_auc',
            xgb_params={'n_estimators': 60, 'early_stopping_rounds': 10})
        base_only = scorer(['noise'])
        with_signal = scorer(['noise', 'signal'])
        self.assertGreater(with_signal, base_only + 0.1)
        self.assertTrue(0.4 <= base_only <= 0.6, base_only)

    def test_empty_feature_list_raises(self):
        train_df, valid_df, test_df = self._split()
        scorer = make_xgb_fold_scorer(train_df, valid_df, test_df,
                                      target='target_win')
        with self.assertRaises(ValueError):
            scorer([])

    def test_default_params_enforce_the_decoupling_regime(self):
        # the sweep deliberately uses a FIXED, shallow, regularised, seeded model so
        # the *group* effect is not confounded with per-subset tuning
        self.assertEqual(DEFAULT_XGB_PARAMS['random_state'], 32)
        self.assertLessEqual(DEFAULT_XGB_PARAMS['max_depth'], 3)
        self.assertGreaterEqual(DEFAULT_XGB_PARAMS['min_child_weight'], 1)
        self.assertIn('early_stopping_rounds', DEFAULT_XGB_PARAMS)

    def test_scorer_is_deterministic(self):
        train_df, valid_df, test_df = self._split()
        scorer = make_xgb_fold_scorer(
            train_df, valid_df, test_df, target='target_win', metric='brier',
            xgb_params={'n_estimators': 60, 'early_stopping_rounds': 10})
        self.assertEqual(scorer(['noise', 'signal']), scorer(['noise', 'signal']))


class TestMakeBartFoldScorer(unittest.TestCase):

    def test_empty_feature_list_raises_before_sampling(self):
        # the empty-subset guard must trip without importing/sampling pymc-bart
        df = pd.DataFrame({'a': [1.0, 2.0, 3.0], 'target_win': [0, 1, 0]})
        scorer = make_bart_fold_scorer(df, df, target='target_win')
        with self.assertRaises(ValueError):
            scorer([])


if __name__ == '__main__':
    unittest.main()
