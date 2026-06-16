import math
from concurrent.futures import ThreadPoolExecutor

import pandas as pd


class BartBackwardElimination:
    """
    Iterative backward elimination driven by an injected fit function — built for BART,
    whose variable_inclusion importances (like any importance measure) are conditional
    on the candidate set and must be RE-MEASURED after every drop.

    fit_fn(features: list[str], seed: int) -> dict with:
        'validation_score': float (higher is better; e.g. ROC-AUC on a validation slice)
        'variable_inclusion': pd.Series indexed by feature name (higher = more used)

    Each iteration runs `replicates` fits with distinct seeds — in parallel threads when
    max_workers > 1 (each PyMC fit spawns its own chain subprocesses, so threads only
    coordinate; max_workers=1 degrades to sequential with identical results). Validation
    scores are averaged across replicates; importances are RANK-averaged (each
    replicate's inclusion converted to ranks before averaging) so replicates with
    different inclusion scales weight equally. The bottom max(1, floor(drop_rate * n))
    features by averaged rank are dropped each iteration until min_features is reached.

    Selection follows the validation-curve PEAK, not smallest-within-tolerance: on flat
    curves the tolerance rule over-shrinks, and small sets carry hold-out variance the
    validation score does not price (see the run-13 experiment-log entry).
    """

    def __init__(self, fit_fn, drop_rate=0.2, min_features=10, replicates=2,
                 max_workers=2, base_seed=32, on_iteration=None):
        self.fit_fn = fit_fn
        self.drop_rate = drop_rate
        self.min_features = min_features
        self.replicates = replicates
        self.max_workers = max_workers
        self.base_seed = base_seed
        # Called with each completed history row (dict). Headless nbconvert buffers a
        # cell's stdout until the cell finishes, so callers that want LIVE progress
        # should write to a sidecar file here rather than print.
        self.on_iteration = on_iteration
        self.history = None

    def run(self, features):
        """
        Run the elimination from the given starting feature list. Returns (and stores
        on self.history) a frame with one row per evaluated set: num_features,
        validation_score (replicate mean), replicate_scores (list), features (list).
        """
        features = list(features)
        rows = []
        iteration = 0
        while True:
            mean_score, replicate_scores, mean_ranks = self._evaluate(features, iteration)
            rows.append({
                'num_features': len(features),
                'validation_score': mean_score,
                'replicate_scores': replicate_scores,
                'features': list(features),
            })
            # Persist after every iteration so a mid-run fit failure (real BART runs
            # take ~20 minutes) leaves the completed iterations queryable instead of
            # discarding them with the exception.
            self.history = pd.DataFrame(rows)
            if self.on_iteration is not None:
                self.on_iteration(rows[-1])
            if len(features) <= self.min_features:
                break
            drop_count = max(1, math.floor(self.drop_rate * len(features)))
            drop_count = min(drop_count, len(features) - self.min_features)
            # mean_ranks: higher rank value = more important; drop the lowest
            features = list(mean_ranks.sort_values(ascending=False)
                            .head(len(features) - drop_count).index)
            iteration += 1
        return self.history

    def get_best_features(self):
        """The feature list at the peak of the replicate-mean validation curve."""
        if self.history is None:
            raise RuntimeError('call run() before get_best_features()')
        best_row = self.history.loc[self.history['validation_score'].idxmax()]
        return list(best_row['features'])

    def _evaluate(self, features, iteration):
        seeds = [self.base_seed + iteration * self.replicates + r
                 for r in range(self.replicates)]
        if self.max_workers > 1 and self.replicates > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                results = list(pool.map(
                    lambda seed: self.fit_fn(list(features), seed), seeds))
        else:
            results = [self.fit_fn(list(features), seed) for seed in seeds]

        scores = [r['validation_score'] for r in results]
        candidate_index = pd.Index(features)
        rank_frames = []
        for result in results:
            inclusion = result['variable_inclusion']
            missing = candidate_index.difference(inclusion.index)
            if len(missing) > 0:
                raise ValueError(
                    f'variable_inclusion is missing {len(missing)} candidate '
                    f'features (e.g. {list(missing[:3])}); a partial Series would '
                    f'silently mis-rank under NaN-skipping averaging'
                )
            rank_frames.append(inclusion.reindex(candidate_index).rank())
        mean_ranks = pd.concat(rank_frames, axis=1).mean(axis=1)
        return sum(scores) / len(scores), scores, mean_ranks
