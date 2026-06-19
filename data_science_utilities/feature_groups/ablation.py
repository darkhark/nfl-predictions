"""Group-level feature ablation: score every subset of feature groups, then decompose
the result into per-group Shapley main effects and pairwise interactions.

This is the harness that answers "which feature groups work together?" without the
confound of the per-feature RFE selector. ``GroupAblation`` runs the full 2^G subset
sweep against an injected ``fit_score_fn`` (so the engine is estimator-agnostic and
unit-testable with a synthetic scorer), and ``cross_validated_group_ablation`` repeats
the sweep across season-blocked folds and reports each effect as mean +/- standard
error -- so an interaction only counts if it clears the evaluation noise that a single
544-game hold-out cannot resolve.
"""

import itertools

import numpy as np
import pandas as pd

from data_science_utilities.feature_groups.shapley import (
    group_shapley_values,
    leave_one_out_deltas,
    pairwise_interaction_index,
    standalone_deltas,
)

HIGHER_IS_BETTER_METRICS = ['roc_auc', 'f1', 'precision', 'recall', 'accuracy']


def feature_subset(base_features, groups, selected_group_names):
    """Build the feature list for a subset: the always-on base features followed by
    the columns of each selected group, in ``groups`` key order."""
    columns = list(base_features)
    for name in groups:
        if name in selected_group_names:
            columns.extend(groups[name])
    return columns


class GroupAblation:
    """Score every subset of feature groups with an injected fit function.

    fit_score_fn(feature_subset: list[str]) -> float validation score on a single
    evaluation block (e.g. one rolling-origin test season).

    :param groups: dict group name -> list of its feature columns.
    :param base_features: columns included in EVERY subset (e.g. rest/context).
    :param metric: scoring metric name; direction read from HIGHER_IS_BETTER_METRICS.
    :param on_iteration: optional callback(dict) invoked per scored subset (headless
        nbconvert buffers stdout, so write progress to a sidecar file here).
    """

    HIGHER_IS_BETTER_METRICS = HIGHER_IS_BETTER_METRICS

    def __init__(self, fit_score_fn, groups, base_features=(), metric='roc_auc',
                 on_iteration=None):
        self.fit_score_fn = fit_score_fn
        self.groups = dict(groups)
        self.base_features = list(base_features)
        self.metric = metric
        self.on_iteration = on_iteration
        self.scores = None
        self.history = None

    @property
    def higher_is_better(self):
        return self.metric in self.HIGHER_IS_BETTER_METRICS

    def run(self):
        """Score the full power set of groups. Returns (and stores on self.history) a
        frame with one row per subset; also populates self.scores (frozenset -> score)."""
        names = list(self.groups)
        scores = {}
        rows = []
        for size in range(len(names) + 1):
            for combo in itertools.combinations(names, size):
                selected = frozenset(combo)
                feats = feature_subset(self.base_features, self.groups, selected)
                score = self.fit_score_fn(feats)
                scores[selected] = score
                row = {
                    'groups': sorted(combo),
                    'num_groups': len(combo),
                    'num_features': len(feats),
                    'score': score,
                }
                rows.append(row)
                if self.on_iteration is not None:
                    self.on_iteration(row)
        self.scores = scores
        self.history = pd.DataFrame(rows)
        return self.history

    def _require_run(self):
        if self.scores is None:
            raise RuntimeError('call run() before decomposing the sweep')

    def score_table(self):
        self._require_run()
        return self.scores

    def shapley_values(self):
        self._require_run()
        return group_shapley_values(self.scores, list(self.groups),
                                    higher_is_better=self.higher_is_better)

    def interaction_index(self):
        self._require_run()
        return pairwise_interaction_index(self.scores, list(self.groups),
                                          higher_is_better=self.higher_is_better)

    def standalone_deltas(self):
        self._require_run()
        return standalone_deltas(self.scores, list(self.groups),
                                 higher_is_better=self.higher_is_better)

    def leave_one_out_deltas(self):
        self._require_run()
        return leave_one_out_deltas(self.scores, list(self.groups),
                                    higher_is_better=self.higher_is_better)


def _mean_se(values):
    """Mean and standard error across folds.

    IMPORTANT: ``se = std / sqrt(k)`` is the SE of the mean only if the per-fold
    estimates are independent. With ``season_rolling_origin_folds`` they are NOT --
    training sets are strictly nested (each fold reuses almost all of the previous
    fold's train data) and the early-stopping validation windows overlap between
    consecutive folds -- so the estimates are positively correlated and this SE is a
    LOWER bound on the true SE. Treat it as a descriptive spread, not a calibrated
    significance test; do not over-trust a |mean/se| just above 2."""
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    se = float(array.std(ddof=1) / np.sqrt(len(array))) if len(array) > 1 else 0.0
    return {'mean': mean, 'se': se}


def cross_validated_group_ablation(groups, base_features, fold_scorers,
                                   metric='roc_auc', on_progress=None):
    """Run the group-ablation sweep once per fold and aggregate each effect across
    folds as mean +/- standard error.

    :param fold_scorers: iterable of fit_score_fn, one per evaluation fold (e.g. one
        per rolling-origin test season). Each is fit_score_fn(feature_subset)->float.
    :param on_progress: optional callback(dict) invoked after each fold completes.
    :returns: dict with per-group ``shapley``/``standalone``/``leave_one_out`` and
        pairwise ``interactions``, each {name -> {'mean', 'se'}}, plus ``mean_scores``
        (subset -> mean score across folds), ``per_fold`` records and ``num_folds``.
        See ``_mean_se``: the reported SE is a LOWER bound (folds are positively
        correlated), so use it as a descriptive spread, not a calibrated significance
        test. ``per_fold`` is retained so callers can inspect the raw per-fold range.
    """
    group_names = list(groups)
    per_fold = []
    for fold_index, fit_score_fn in enumerate(fold_scorers):
        ablation = GroupAblation(fit_score_fn, groups, base_features, metric=metric)
        ablation.run()
        record = {
            'fold': fold_index,
            'scores': ablation.score_table(),
            'shapley': ablation.shapley_values(),
            'interactions': ablation.interaction_index(),
            'standalone': ablation.standalone_deltas(),
            'leave_one_out': ablation.leave_one_out_deltas(),
        }
        per_fold.append(record)
        if on_progress is not None:
            on_progress(record)

    def aggregate_groups(key):
        return {name: _mean_se([fold[key][name] for fold in per_fold])
                for name in group_names}

    pairs = [frozenset(pair) for pair in itertools.combinations(group_names, 2)]
    interactions = {
        pair: _mean_se([fold['interactions'][pair] for fold in per_fold])
        for pair in pairs
    }
    all_subsets = per_fold[0]['scores'].keys() if per_fold else []
    mean_scores = {
        subset: float(np.mean([fold['scores'][subset] for fold in per_fold]))
        for subset in all_subsets
    }
    return {
        'group_names': group_names,
        'num_folds': len(per_fold),
        'shapley': aggregate_groups('shapley'),
        'standalone': aggregate_groups('standalone'),
        'leave_one_out': aggregate_groups('leave_one_out'),
        'interactions': interactions,
        'mean_scores': mean_scores,
        'per_fold': per_fold,
    }
