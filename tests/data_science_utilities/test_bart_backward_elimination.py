import threading
import unittest

import pandas as pd

from data_science_utilities.models.bart.feature_selection.backward_elimination import (
    BartBackwardElimination,
)


def make_fit_fn(score_by_size=None):
    """A deterministic synthetic fit_fn. Importance is alphabetical: features earlier
    in the alphabet get HIGHER inclusion, so elimination order is fully predictable
    (z drops first). validation_score defaults to 0.6 + 0.001 * num_features unless
    score_by_size overrides a specific size. The seed nudges inclusion by a tiny,
    rank-preserving epsilon so replicate averaging is exercised without changing
    the elimination order."""
    calls = []

    def fit_fn(features, seed):
        calls.append((tuple(features), seed))
        ordered = sorted(features)  # alphabetical
        inclusion = pd.Series(
            {f: (len(ordered) - i) + seed * 1e-6 for i, f in enumerate(ordered)}
        )
        size = len(features)
        score = (score_by_size or {}).get(size, 0.6 + 0.001 * size)
        return {'validation_score': score, 'variable_inclusion': inclusion}

    fit_fn.calls = calls
    return fit_fn


FEATURES_8 = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']


class TestBartBackwardElimination(unittest.TestCase):

    def test_drop_schedule_and_history(self):
        fit_fn = make_fit_fn()
        rfe = BartBackwardElimination(fit_fn, drop_rate=0.25, min_features=3,
                                      replicates=2, max_workers=1)
        history = rfe.run(FEATURES_8)
        # 8 -> drop 2 -> 6 -> drop 1 (25% of 6 floored to 1, min 1) -> wait: 25% of 6
        # is 1.5 -> floor 1? The spec: drop max(1, floor(drop_rate * n)). 8->6->5->4->3.
        self.assertEqual(list(history['num_features']), [8, 6, 5, 4, 3])
        # alphabetical importance means the LAST letters drop first
        self.assertEqual(sorted(history.iloc[1]['features']),
                         ['a', 'b', 'c', 'd', 'e', 'f'])
        self.assertEqual(sorted(history.iloc[-1]['features']), ['a', 'b', 'c'])

    def test_stops_at_min_features(self):
        fit_fn = make_fit_fn()
        rfe = BartBackwardElimination(fit_fn, drop_rate=0.5, min_features=4,
                                      replicates=1, max_workers=1)
        history = rfe.run(FEATURES_8)
        # 8 -> 4, then stop (4 == min_features is evaluated, no further drop)
        self.assertEqual(list(history['num_features']), [8, 4])

    def test_replicates_get_distinct_seeds_and_scores_average(self):
        fit_fn = make_fit_fn(score_by_size={8: 0.7})
        rfe = BartBackwardElimination(fit_fn, drop_rate=0.5, min_features=4,
                                      replicates=3, max_workers=1, base_seed=100)
        history = rfe.run(FEATURES_8)
        first_iter_calls = [c for c in fit_fn.calls if len(c[0]) == 8]
        self.assertEqual(len(first_iter_calls), 3)
        self.assertEqual(len({seed for _, seed in first_iter_calls}), 3)
        self.assertAlmostEqual(history.iloc[0]['validation_score'], 0.7)
        self.assertEqual(len(history.iloc[0]['replicate_scores']), 3)

    def test_parallel_workers_produce_same_result_as_sequential(self):
        sequential = BartBackwardElimination(make_fit_fn(), drop_rate=0.25,
                                             min_features=3, replicates=2,
                                             max_workers=1).run(FEATURES_8)
        parallel = BartBackwardElimination(make_fit_fn(), drop_rate=0.25,
                                           min_features=3, replicates=2,
                                           max_workers=2).run(FEATURES_8)
        self.assertEqual(list(sequential['num_features']),
                         list(parallel['num_features']))
        for s_feats, p_feats in zip(sequential['features'], parallel['features']):
            self.assertEqual(sorted(s_feats), sorted(p_feats))

    def test_parallel_actually_runs_concurrently(self):
        # two replicates that block until both have started prove real concurrency
        barrier = threading.Barrier(2, timeout=10)

        def blocking_fit(features, seed):
            barrier.wait()  # deadlocks (then Barrier raises) unless 2 run at once
            ordered = sorted(features)
            return {'validation_score': 0.6,
                    'variable_inclusion': pd.Series(
                        {f: len(ordered) - i for i, f in enumerate(ordered)})}

        rfe = BartBackwardElimination(blocking_fit, drop_rate=0.5, min_features=4,
                                      replicates=2, max_workers=2)
        history = rfe.run(FEATURES_8)  # raises BrokenBarrierError if sequential
        self.assertEqual(list(history['num_features']), [8, 4])

    def test_rank_averaged_inclusion_decides_drops(self):
        # replicate seeds disagree on raw scale but agree on order -> order wins;
        # engineered case: one replicate's raw inclusion would mislead a raw average
        def fit_fn(features, seed):
            ordered = sorted(features)
            if seed % 2 == 0:
                # huge scale, alphabetical order
                inclusion = {f: (len(ordered) - i) * 1000 for i, f in enumerate(ordered)}
            else:
                # tiny scale, alphabetical order
                inclusion = {f: (len(ordered) - i) * 0.001 for i, f in enumerate(ordered)}
            return {'validation_score': 0.6,
                    'variable_inclusion': pd.Series(inclusion)}

        rfe = BartBackwardElimination(fit_fn, drop_rate=0.25, min_features=6,
                                      replicates=2, max_workers=1)
        history = rfe.run(FEATURES_8)
        # rank-averaging keeps the alphabetical order regardless of scale
        self.assertEqual(sorted(history.iloc[-1]['features']),
                         ['a', 'b', 'c', 'd', 'e', 'f'])

    def test_best_features_at_validation_peak(self):
        # run-13 lesson: select the PEAK of the validation curve, not the smallest
        # set within a tolerance
        fit_fn = make_fit_fn(score_by_size={8: 0.60, 6: 0.65, 5: 0.64, 4: 0.61, 3: 0.58})
        rfe = BartBackwardElimination(fit_fn, drop_rate=0.25, min_features=3,
                                      replicates=1, max_workers=1)
        rfe.run(FEATURES_8)
        best = rfe.get_best_features()
        self.assertEqual(len(best), 6)
        self.assertEqual(sorted(best), ['a', 'b', 'c', 'd', 'e', 'f'])

    def test_run_before_best_raises(self):
        rfe = BartBackwardElimination(make_fit_fn())
        with self.assertRaises(RuntimeError):
            rfe.get_best_features()

    def test_partial_history_survives_fit_failure(self):
        # A real BART run takes ~20 minutes; a crash in a late iteration must leave
        # the completed iterations queryable on .history.
        state = {'calls': 0}

        def flaky_fit(features, seed):
            state['calls'] += 1
            if len(features) < 8:
                raise RuntimeError('sampler crashed')
            ordered = sorted(features)
            return {'validation_score': 0.6,
                    'variable_inclusion': pd.Series(
                        {f: len(ordered) - i for i, f in enumerate(ordered)})}

        rfe = BartBackwardElimination(flaky_fit, drop_rate=0.5, min_features=2,
                                      replicates=1, max_workers=1)
        with self.assertRaises(RuntimeError):
            rfe.run(FEATURES_8)
        self.assertIsNotNone(rfe.history)
        self.assertEqual(list(rfe.history['num_features']), [8])

    def test_incomplete_inclusion_series_raises(self):
        def partial_fit(features, seed):
            ordered = sorted(features)[:-1]  # drops one candidate from the Series
            return {'validation_score': 0.6,
                    'variable_inclusion': pd.Series(
                        {f: len(ordered) - i for i, f in enumerate(ordered)})}

        rfe = BartBackwardElimination(partial_fit, drop_rate=0.25, min_features=3,
                                      replicates=1, max_workers=1)
        with self.assertRaises(ValueError):
            rfe.run(FEATURES_8)

    def test_on_iteration_callback_receives_each_completed_row(self):
        rows_seen = []
        rfe = BartBackwardElimination(make_fit_fn(), drop_rate=0.5, min_features=4,
                                      replicates=1, max_workers=1,
                                      on_iteration=rows_seen.append)
        rfe.run(FEATURES_8)
        self.assertEqual([row['num_features'] for row in rows_seen], [8, 4])
        self.assertIn('validation_score', rows_seen[0])


class TestBartOneSE(unittest.TestCase):

    def _history(self):
        return pd.DataFrame([
            {'num_features': 8, 'validation_score': 0.200,
             'replicate_scores': [0.200, 0.200, 0.200],
             'features': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']},
            {'num_features': 6, 'validation_score': 0.180,
             'replicate_scores': [0.178, 0.180, 0.182],
             'features': ['a', 'b', 'c', 'd', 'e', 'f']},
            {'num_features': 5, 'validation_score': 0.181,
             'replicate_scores': [0.181, 0.181, 0.181],
             'features': ['a', 'b', 'c', 'd', 'e']},
            {'num_features': 3, 'validation_score': 0.230,
             'replicate_scores': [0.230, 0.230, 0.230],
             'features': ['a', 'b', 'c']},
        ])

    def test_lower_is_better_picks_parsimonious_within_band(self):
        rfe = BartBackwardElimination(make_fit_fn())
        rfe.history = self._history()
        best = rfe.get_best_features_1se(higher_is_better=False)
        self.assertEqual(sorted(best), ['a', 'b', 'c', 'd', 'e'])

    def test_tight_band_returns_optimum(self):
        rfe = BartBackwardElimination(make_fit_fn())
        df = self._history()
        df.at[1, 'replicate_scores'] = [0.180, 0.180, 0.180]
        rfe.history = df
        best = rfe.get_best_features_1se(higher_is_better=False)
        self.assertEqual(sorted(best), ['a', 'b', 'c', 'd', 'e', 'f'])

    def test_higher_is_better_direction(self):
        rfe = BartBackwardElimination(make_fit_fn())
        rfe.history = pd.DataFrame([
            {'num_features': 8, 'validation_score': 0.690,
             'replicate_scores': [0.690, 0.690, 0.690], 'features': list('abcdefgh')},
            {'num_features': 6, 'validation_score': 0.710,
             'replicate_scores': [0.708, 0.710, 0.712], 'features': list('abcdef')},
            {'num_features': 5, 'validation_score': 0.709,
             'replicate_scores': [0.709, 0.709, 0.709], 'features': list('abcde')},
        ])
        best = rfe.get_best_features_1se(higher_is_better=True)
        self.assertEqual(sorted(best), list('abcde'))

    def test_run_before_1se_raises(self):
        rfe = BartBackwardElimination(make_fit_fn())
        with self.assertRaises(RuntimeError):
            rfe.get_best_features_1se()


if __name__ == '__main__':
    unittest.main()
