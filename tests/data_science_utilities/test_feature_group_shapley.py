import itertools
import unittest

from data_science_utilities.feature_groups.shapley import (
    group_shapley_values,
    leave_one_out_deltas,
    pairwise_interaction_index,
    standalone_deltas,
)


def powerset_scores(groups, value_fn):
    """Build a complete subset->score table from a characteristic function."""
    scores = {}
    for r in range(len(groups) + 1):
        for combo in itertools.combinations(groups, r):
            scores[frozenset(combo)] = value_fn(frozenset(combo))
    return scores


class TestGroupShapleyValues(unittest.TestCase):

    def test_additive_game_attributes_each_groups_own_value(self):
        # u(S) = sum of member values -> Shapley value == own value, no interaction
        value = {'a': 0.03, 'b': 0.05, 'c': 0.01}
        scores = powerset_scores(['a', 'b', 'c'],
                                 lambda s: sum(value[g] for g in s))
        shap = group_shapley_values(scores, ['a', 'b', 'c'])
        self.assertAlmostEqual(shap['a'], 0.03)
        self.assertAlmostEqual(shap['b'], 0.05)
        self.assertAlmostEqual(shap['c'], 0.01)

    def test_two_redundant_groups_split_the_shared_value(self):
        # perfect substitutes: u = 0.1 if either present (OR), 0 if neither
        scores = powerset_scores(
            ['a', 'b'], lambda s: 0.1 if len(s) >= 1 else 0.0)
        shap = group_shapley_values(scores, ['a', 'b'])
        self.assertAlmostEqual(shap['a'], 0.05)
        self.assertAlmostEqual(shap['b'], 0.05)

    def test_efficiency_property_values_sum_to_grand_coalition_gain(self):
        # Sum of Shapley values == u(all) - u(none) for ANY game.
        groups = ['a', 'b', 'c']
        scores = powerset_scores(groups, lambda s: 0.7 + 0.013 * len(s) ** 2)
        shap = group_shapley_values(scores, groups)
        grand = scores[frozenset(groups)] - scores[frozenset()]
        self.assertAlmostEqual(sum(shap.values()), grand)

    def test_lower_is_better_metric_flips_direction(self):
        # Brier-like: adding 'a' LOWERS the score by 0.02 (an improvement).
        scores = {frozenset(): 0.25, frozenset(['a']): 0.23}
        shap = group_shapley_values(scores, ['a'], higher_is_better=False)
        self.assertAlmostEqual(shap['a'], 0.02)  # positive == beneficial

    def test_single_group_value_is_its_marginal_over_empty(self):
        scores = {frozenset(): 0.6, frozenset(['a']): 0.71}
        shap = group_shapley_values(scores, ['a'])
        self.assertAlmostEqual(shap['a'], 0.11)

    def test_missing_subset_raises(self):
        scores = {frozenset(): 0.6, frozenset(['a']): 0.7}  # missing {b}, {a,b}
        with self.assertRaises(KeyError):
            group_shapley_values(scores, ['a', 'b'])


class TestPairwiseInteractionIndex(unittest.TestCase):

    def test_additive_game_has_zero_interactions(self):
        value = {'a': 0.03, 'b': 0.05, 'c': 0.01}
        scores = powerset_scores(['a', 'b', 'c'],
                                 lambda s: sum(value[g] for g in s))
        inter = pairwise_interaction_index(scores, ['a', 'b', 'c'])
        for pair_value in inter.values():
            self.assertAlmostEqual(pair_value, 0.0)

    def test_redundant_pair_has_negative_interaction(self):
        # substitutes: each worth 0.1 alone, together still only 0.1
        scores = powerset_scores(['a', 'b'],
                                 lambda s: 0.1 if len(s) >= 1 else 0.0)
        inter = pairwise_interaction_index(scores, ['a', 'b'])
        self.assertAlmostEqual(inter[frozenset(['a', 'b'])], -0.1)

    def test_synergistic_pair_has_positive_interaction(self):
        # complements: worthless alone, worth 0.2 together
        scores = {
            frozenset(): 0.0, frozenset(['a']): 0.0,
            frozenset(['b']): 0.0, frozenset(['a', 'b']): 0.2,
        }
        inter = pairwise_interaction_index(scores, ['a', 'b'])
        self.assertAlmostEqual(inter[frozenset(['a', 'b'])], 0.2)

    def test_interaction_is_symmetric_via_frozenset_keys(self):
        scores = powerset_scores(['x', 'y'],
                                 lambda s: 0.04 * len(s) + (0.1 if s == frozenset(['x', 'y']) else 0))
        inter = pairwise_interaction_index(scores, ['x', 'y'])
        self.assertIn(frozenset(['x', 'y']), inter)
        self.assertEqual(len(inter), 1)

    def test_lower_is_better_redundant_pair_flips_sign(self):
        # Brier substitutes: each lowers Brier to 0.23 alone, together no better.
        scores = {
            frozenset(): 0.25, frozenset(['a']): 0.23,
            frozenset(['b']): 0.23, frozenset(['a', 'b']): 0.23,
        }
        inter = pairwise_interaction_index(scores, ['a', 'b'], higher_is_better=False)
        # raw second difference on Brier = 0.23-0.23-0.23+0.25 = +0.02; utility flips
        # to -0.02 (redundant: the two improvements do not add up)
        self.assertAlmostEqual(inter[frozenset(['a', 'b'])], -0.02)


class TestExactValuesThreeGroupInteractingGame(unittest.TestCase):
    """Pin the combinatorial weights against a fully hand-computed non-additive
    3-group game, so a corrupted weight formula (e.g. n! instead of (n-1)! in the
    interaction index) cannot pass silently."""

    # u(emptyset)=0; singles a=.10 b=.06 c=.02; one synergistic pair a&b adds +.04
    # on TOP of additivity whenever both present; c is purely additive.
    SCORES = {
        frozenset(): 0.00,
        frozenset(['a']): 0.10,
        frozenset(['b']): 0.06,
        frozenset(['c']): 0.02,
        frozenset(['a', 'b']): 0.20,           # .10+.06+.04 synergy
        frozenset(['a', 'c']): 0.12,           # .10+.02
        frozenset(['b', 'c']): 0.08,           # .06+.02
        frozenset(['a', 'b', 'c']): 0.22,      # .10+.06+.02+.04
    }

    def test_shapley_values_match_hand_computation(self):
        shap = group_shapley_values(self.SCORES, ['a', 'b', 'c'])
        # the +.04 a-b synergy is split evenly between a and b (Shapley symmetry)
        self.assertAlmostEqual(shap['a'], 0.10 + 0.02)
        self.assertAlmostEqual(shap['b'], 0.06 + 0.02)
        self.assertAlmostEqual(shap['c'], 0.02)
        # efficiency: values sum to u(grand) - u(empty)
        self.assertAlmostEqual(sum(shap.values()), 0.22)

    def test_interaction_index_isolates_the_synergistic_pair(self):
        inter = pairwise_interaction_index(self.SCORES, ['a', 'b', 'c'])
        # a&b carry the +.04 synergy at BOTH coalition sizes (S=empty and S={c});
        # the index averages those equal second differences -> exactly .04
        self.assertAlmostEqual(inter[frozenset(['a', 'b'])], 0.04)
        self.assertAlmostEqual(inter[frozenset(['a', 'c'])], 0.0)
        self.assertAlmostEqual(inter[frozenset(['b', 'c'])], 0.0)


class TestStandaloneAndLeaveOneOut(unittest.TestCase):

    def test_standalone_delta_is_value_over_base_only(self):
        scores = {
            frozenset(): 0.60, frozenset(['a']): 0.68,
            frozenset(['b']): 0.62, frozenset(['a', 'b']): 0.69,
        }
        standalone = standalone_deltas(scores, ['a', 'b'])
        self.assertAlmostEqual(standalone['a'], 0.08)
        self.assertAlmostEqual(standalone['b'], 0.02)

    def test_leave_one_out_delta_is_grand_minus_without(self):
        scores = {
            frozenset(): 0.60, frozenset(['a']): 0.68,
            frozenset(['b']): 0.62, frozenset(['a', 'b']): 0.69,
        }
        loo = leave_one_out_deltas(scores, ['a', 'b'])
        self.assertAlmostEqual(loo['a'], 0.69 - 0.62)  # adding a on top of b
        self.assertAlmostEqual(loo['b'], 0.69 - 0.68)  # adding b on top of a

    def test_lower_is_better_standalone(self):
        scores = {frozenset(): 0.25, frozenset(['a']): 0.22}
        standalone = standalone_deltas(scores, ['a'], higher_is_better=False)
        self.assertAlmostEqual(standalone['a'], 0.03)


if __name__ == '__main__':
    unittest.main()
