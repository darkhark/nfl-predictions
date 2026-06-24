import unittest

import numpy as np
import pandas as pd

from data_science_utilities.models.xgb.feature_selection.recursive.classifier_cross_validation import (
    ClassifierCrossValidationRecursiveFeatureSelection,
)

FAST_XGB_PARAMS = {
    'n_estimators': 5,
    'max_depth': 2,
    'eval_metric': 'logloss',
    'random_state': 32,
}


def make_synthetic_frame(num_features=12, rows=80):
    rng = np.random.default_rng(32)
    X = pd.DataFrame(
        rng.normal(size=(rows, num_features)),
        columns=[f'feat_{i}' for i in range(num_features)],
    )
    # give a couple of features real signal so importances are non-degenerate
    y = pd.Series(((X['feat_0'] + 0.5 * X['feat_1'] + rng.normal(scale=0.5, size=rows)) > 0).astype(int))
    return X, y


class TestMinFeaturesFloor(unittest.TestCase):

    def test_feature_count_never_falls_below_min_features(self):
        X, y = make_synthetic_frame()
        rfe = ClassifierCrossValidationRecursiveFeatureSelection(X, y, dict(FAST_XGB_PARAMS))
        rfe.get_optimal_features_no_grouped_records(
            drop_rate=0.3, max_iter=20, n_folds=3, min_features=5
        )
        evaluated_sizes = sorted(rfe.all_features.keys())
        self.assertEqual(min(evaluated_sizes), 5)

    def test_loop_stops_at_floor_instead_of_exhausting_max_iter(self):
        X, y = make_synthetic_frame()
        rfe = ClassifierCrossValidationRecursiveFeatureSelection(X, y, dict(FAST_XGB_PARAMS))
        rfe.get_optimal_features_no_grouped_records(
            drop_rate=0.3, max_iter=20, n_folds=3, min_features=5
        )
        # 12 features at 30% drops reaches 5 in a handful of iterations; with the floor
        # the loop must stop there rather than re-evaluating 5 for all 20 iterations
        self.assertLess(len(rfe.all_model_scores), 20)
        self.assertEqual(len(rfe.all_model_scores), len(rfe.all_features))

    def test_default_behavior_unchanged_without_min_features(self):
        X, y = make_synthetic_frame()
        rfe = ClassifierCrossValidationRecursiveFeatureSelection(X, y, dict(FAST_XGB_PARAMS))
        rfe.get_optimal_features_no_grouped_records(drop_rate=0.3, max_iter=4, n_folds=3)
        # no floor: runs all max_iter iterations exactly as before
        self.assertEqual(len(rfe.all_model_scores), 4)

    def test_on_iteration_callback_streams_progress(self):
        X, y = make_synthetic_frame()
        rfe = ClassifierCrossValidationRecursiveFeatureSelection(X, y, dict(FAST_XGB_PARAMS))
        seen = []
        rfe.get_optimal_features_no_grouped_records(
            drop_rate=0.3, max_iter=3, n_folds=3, on_iteration=seen.append
        )
        self.assertEqual(len(seen), 3)
        self.assertEqual(seen[0]['iteration'], 1)
        self.assertEqual(seen[0]['max_iter'], 3)
        self.assertIn('score', seen[0])
        self.assertIn('num_features', seen[0])


class TestBrierMetric(unittest.TestCase):

    def test_brier_metric_runs_and_scores_are_probabilities(self):
        X, y = make_synthetic_frame()
        rfe = ClassifierCrossValidationRecursiveFeatureSelection(
            X, y, dict(FAST_XGB_PARAMS), model_score_metric='brier'
        )
        rfe.get_optimal_features_no_grouped_records(drop_rate=0.3, max_iter=3, n_folds=3)
        self.assertTrue(all(0.0 <= s <= 1.0 for s in rfe.all_model_scores))
        self.assertEqual(len(rfe.all_model_scores), 3)

    def test_brier_is_treated_as_lower_is_better(self):
        rfe = ClassifierCrossValidationRecursiveFeatureSelection(
            *make_synthetic_frame(), dict(FAST_XGB_PARAMS), model_score_metric='brier'
        )
        self.assertNotIn('brier', rfe.HIGHER_IS_BETTER_METRICS)


class TestPerFoldStorage(unittest.TestCase):

    def test_per_fold_scores_stored_and_consistent_with_means(self):
        X, y = make_synthetic_frame()
        rfe = ClassifierCrossValidationRecursiveFeatureSelection(X, y, dict(FAST_XGB_PARAMS))
        rfe.get_optimal_features_no_grouped_records(drop_rate=0.3, max_iter=3, n_folds=3)
        self.assertEqual(len(rfe.all_model_score_folds), len(rfe.all_model_scores))
        for folds, mean in zip(rfe.all_model_score_folds, rfe.all_model_scores):
            self.assertEqual(len(folds), 3)
            self.assertAlmostEqual(sum(folds) / len(folds), mean)


class TestOneSESelection(unittest.TestCase):

    def _rfe(self, metric):
        X, y = make_synthetic_frame()
        return ClassifierCrossValidationRecursiveFeatureSelection(
            X, y, dict(FAST_XGB_PARAMS), model_score_metric=metric
        )

    def test_lower_is_better_picks_parsimonious_within_band(self):
        rfe = self._rfe('brier')
        rfe.all_features = {10: ['f'] * 10, 7: ['f'] * 7, 5: ['f'] * 5, 3: ['f'] * 3}
        rfe.all_model_scores = [0.200, 0.180, 0.181, 0.230]
        rfe.all_model_score_folds = [
            [0.200, 0.200, 0.200],
            [0.178, 0.180, 0.182],
            [0.181, 0.181, 0.181],
            [0.230, 0.230, 0.230],
        ]
        self.assertEqual(rfe.get_best_num_features_1se(), 5)

    def test_tight_band_returns_the_optimum(self):
        rfe = self._rfe('brier')
        rfe.all_features = {10: ['f'] * 10, 7: ['f'] * 7, 5: ['f'] * 5}
        rfe.all_model_scores = [0.200, 0.180, 0.190]
        rfe.all_model_score_folds = [
            [0.200, 0.200, 0.200],
            [0.180, 0.180, 0.180],
            [0.190, 0.190, 0.190],
        ]
        self.assertEqual(rfe.get_best_num_features_1se(), 7)

    def test_higher_is_better_direction(self):
        rfe = self._rfe('roc_auc')
        rfe.all_features = {10: ['f'] * 10, 7: ['f'] * 7, 5: ['f'] * 5, 3: ['f'] * 3}
        rfe.all_model_scores = [0.690, 0.710, 0.709, 0.660]
        rfe.all_model_score_folds = [
            [0.690, 0.690, 0.690],
            [0.708, 0.710, 0.712],
            [0.709, 0.709, 0.709],
            [0.660, 0.660, 0.660],
        ]
        self.assertEqual(rfe.get_best_num_features_1se(), 5)


class TestStalledPruning(unittest.TestCase):

    def test_stalled_pruning_terminates_without_crashing(self):
        # A tiny all-informative pool makes iteration 0's importance ranking drop nothing,
        # which previously re-evaluated the same feature count and crashed. The convergence
        # guard must let it finish cleanly with aligned bookkeeping.
        rng = np.random.default_rng(32)
        X = pd.DataFrame(rng.normal(size=(80, 3)), columns=['a', 'b', 'c'])
        y = pd.Series(((X['a'] + X['b'] + X['c'] + rng.normal(scale=0.3, size=80)) > 0).astype(int))
        rfe = ClassifierCrossValidationRecursiveFeatureSelection(X, y, dict(FAST_XGB_PARAMS))
        rfe.get_optimal_features_no_grouped_records(drop_rate=0.3, max_iter=4, n_folds=3)
        self.assertEqual(len(rfe.all_model_scores), len(rfe.all_features))  # alignment intact
        self.assertEqual(len(set(rfe.all_features.keys())), len(rfe.all_features))  # unique counts
        self.assertGreaterEqual(len(rfe.all_features), 1)


if __name__ == '__main__':
    unittest.main()
