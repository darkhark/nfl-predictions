"""Group-level cooperative-game decomposition of a feature-ablation sweep.

Given a COMPLETE table of validation scores -- one per subset of feature groups --
this module answers two questions the per-feature RFE selector cannot:

* **Main effect** (``group_shapley_values``): how much does each group add, averaged
  fairly over every context of the other groups? Unlike a single add-one-family run,
  the Shapley value is symmetric to ordering and satisfies efficiency (the values sum
  to the full-set gain).
* **Pairwise interaction** (``pairwise_interaction_index``): do two groups *work
  together*? The Shapley interaction index is the averaged second difference
  ``u(S+i+j) - u(S+i) - u(S+j) + u(S)``. **Positive = super-additive (the pair beats
  the sum of its parts); negative = sub-additive (the pair adds less than its parts).**

  Caveat on NEGATIVE interactions: sub-additivity has TWO causes this index cannot
  separate -- (a) genuine informational *redundancy* (correlated substitutes, the
  crowding-out the experiment log keeps inferring), and (b) *metric saturation*: when
  the score is near a ceiling/floor (this model sits on a documented ~0.705 ROC-AUC /
  ~0.219 Brier plateau), two families that each push the model toward the ceiling MUST
  interact negatively because the ceiling caps their joint score, even if their
  information is independent. A negative interaction is therefore evidence of
  sub-additivity, NOT proof of redundancy on its own -- corroborate with feature
  correlations or a less ceiling-bound target (logit / log-loss space).

Scores are keyed by ``frozenset`` of group names and the FULL power set must be
present. Pass ``higher_is_better=False`` for lower-is-better metrics (Brier, log loss):
scores are converted to a utility (negated) so a positive Shapley value / interaction
always means "beneficial" / "super-additive" regardless of metric direction.
"""

import itertools
from math import factorial


def _utility(scores, higher_is_better):
    sign = 1.0 if higher_is_better else -1.0
    return {subset: sign * value for subset, value in scores.items()}


def _get(utility, subset):
    key = frozenset(subset)
    if key not in utility:
        raise KeyError(
            f'subset {sorted(key)} missing from the score table; the full power set '
            f'of all groups must be scored before decomposing')
    return utility[key]


def group_shapley_values(scores, groups, higher_is_better=True):
    """Shapley value of each group: its average marginal contribution over all
    orderings of the other groups.

    :param scores: dict frozenset(group subset) -> score, for the COMPLETE power set.
    :param groups: the full set of toggleable group names.
    :param higher_is_better: metric direction (False for Brier/log-loss).
    :returns: dict group -> Shapley value (positive == beneficial)."""
    groups = list(groups)
    utility = _utility(scores, higher_is_better)
    n = len(groups)
    values = {}
    for group in groups:
        others = [g for g in groups if g != group]
        total = 0.0
        for size in range(len(others) + 1):
            weight = factorial(size) * factorial(n - size - 1) / factorial(n)
            for combo in itertools.combinations(others, size):
                base = frozenset(combo)
                total += weight * (_get(utility, base | {group}) - _get(utility, base))
        values[group] = total
    return values


def pairwise_interaction_index(scores, groups, higher_is_better=True):
    """Shapley interaction index for every unordered pair of groups.

    :returns: dict frozenset({i, j}) -> interaction. Positive == super-additive (the
        pair beats the sum of its parts); negative == sub-additive (the pair adds less
        than its parts -- from redundancy OR metric saturation; see the module
        docstring caveat before reading a negative value as redundancy)."""
    groups = list(groups)
    utility = _utility(scores, higher_is_better)
    n = len(groups)
    interactions = {}
    for group_i, group_j in itertools.combinations(groups, 2):
        others = [g for g in groups if g not in (group_i, group_j)]
        total = 0.0
        for size in range(len(others) + 1):
            weight = factorial(size) * factorial(n - size - 2) / factorial(n - 1)
            for combo in itertools.combinations(others, size):
                base = frozenset(combo)
                total += weight * (
                    _get(utility, base | {group_i, group_j})
                    - _get(utility, base | {group_i})
                    - _get(utility, base | {group_j})
                    + _get(utility, base)
                )
        interactions[frozenset([group_i, group_j])] = total
    return interactions


def standalone_deltas(scores, groups, higher_is_better=True):
    """Each group's value ON ITS OWN over the base-only model: u({g}) - u(emptyset)."""
    utility = _utility(scores, higher_is_better)
    base = _get(utility, frozenset())
    return {group: _get(utility, frozenset([group])) - base for group in groups}


def leave_one_out_deltas(scores, groups, higher_is_better=True):
    """Each group's MARGINAL value on top of every other group:
    u(all) - u(all - {g}). Near-zero LOO with a large standalone delta is the
    fingerprint of a redundant substitute."""
    groups = list(groups)
    utility = _utility(scores, higher_is_better)
    grand = _get(utility, frozenset(groups))
    return {group: grand - _get(utility, frozenset(g for g in groups if g != group))
            for group in groups}
