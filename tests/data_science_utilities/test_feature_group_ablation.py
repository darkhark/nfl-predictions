import unittest

from data_science_utilities.feature_groups.ablation import (
    GroupAblation,
    cross_validated_group_ablation,
    feature_subset,
)

GROUPS = {
    'box': ['off_box_a', 'off_box_b'],
    'pbp': ['off_pbp_a'],
}
BASE = ['week', 'div_game']


class TestFeatureSubset(unittest.TestCase):

    def test_base_always_included_then_selected_groups(self):
        self.assertEqual(feature_subset(BASE, GROUPS, []), ['week', 'div_game'])
        self.assertEqual(
            feature_subset(BASE, GROUPS, ['box']),
            ['week', 'div_game', 'off_box_a', 'off_box_b'])
        self.assertEqual(
            feature_subset(BASE, GROUPS, ['box', 'pbp']),
            ['week', 'div_game', 'off_box_a', 'off_box_b', 'off_pbp_a'])


class TestGroupAblationRun(unittest.TestCase):

    def test_scores_the_full_power_set(self):
        seen = []
        abl = GroupAblation(lambda feats: float(len(feats)), GROUPS, BASE)
        abl.run()
        # 2 groups -> 4 subsets
        self.assertEqual(len(abl.score_table()), 4)
        self.assertIn(frozenset(), abl.score_table())
        self.assertIn(frozenset(['box', 'pbp']), abl.score_table())

    def test_each_subset_feature_list_includes_base_and_its_groups(self):
        received = []
        GroupAblation(lambda feats: received.append(list(feats)) or 0.0,
                      GROUPS, BASE).run()
        for feats in received:
            self.assertTrue(set(BASE).issubset(feats), 'base features always present')
        # the grand coalition must contain every group's columns
        grand = max(received, key=len)
        self.assertEqual(set(grand),
                         set(BASE) | set(GROUPS['box']) | set(GROUPS['pbp']))

    def test_on_iteration_called_per_subset(self):
        rows = []
        GroupAblation(lambda feats: 0.0, GROUPS, BASE,
                      on_iteration=rows.append).run()
        self.assertEqual(len(rows), 4)
        self.assertIn('groups', rows[0])
        self.assertIn('score', rows[0])

    def test_shapley_matches_score_table_higher_is_better(self):
        # additive game on the two groups: box worth 0.04, pbp worth 0.02
        value = {'box': 0.04, 'pbp': 0.02}
        def fit(feats):
            s = 0.6
            if set(GROUPS['box']).issubset(feats): s += value['box']
            if set(GROUPS['pbp']).issubset(feats): s += value['pbp']
            return s
        abl = GroupAblation(fit, GROUPS, BASE, metric='roc_auc')
        abl.run()
        shap = abl.shapley_values()
        self.assertAlmostEqual(shap['box'], 0.04)
        self.assertAlmostEqual(shap['pbp'], 0.02)
        inter = abl.interaction_index()
        self.assertAlmostEqual(inter[frozenset(['box', 'pbp'])], 0.0)

    def test_brier_metric_is_lower_is_better(self):
        # adding box LOWERS brier from 0.25 to 0.23 -> positive shapley (beneficial)
        def fit(feats):
            return 0.23 if set(GROUPS['box']).issubset(feats) else 0.25
        single = {'box': GROUPS['box']}
        abl = GroupAblation(fit, single, BASE, metric='brier')
        abl.run()
        self.assertFalse(abl.higher_is_better)
        self.assertAlmostEqual(abl.shapley_values()['box'], 0.02)

    def test_empty_groups_yields_base_only(self):
        abl = GroupAblation(lambda feats: 0.5, {}, BASE)
        abl.run()
        self.assertEqual(list(abl.score_table()), [frozenset()])


class TestCrossValidatedGroupAblation(unittest.TestCase):

    def _additive_fold_scorer(self, box_value, pbp_value):
        def fit(feats):
            s = 0.0
            if set(GROUPS['box']).issubset(feats): s += box_value
            if set(GROUPS['pbp']).issubset(feats): s += pbp_value
            return s
        return fit

    def test_aggregates_mean_and_se_across_folds(self):
        # three folds with box value {0.02, 0.04, 0.06} -> mean 0.04, real spread
        fold_scorers = [
            self._additive_fold_scorer(0.02, 0.01),
            self._additive_fold_scorer(0.04, 0.01),
            self._additive_fold_scorer(0.06, 0.01),
        ]
        result = cross_validated_group_ablation(
            GROUPS, BASE, fold_scorers, metric='roc_auc')
        self.assertAlmostEqual(result['shapley']['box']['mean'], 0.04)
        self.assertGreater(result['shapley']['box']['se'], 0.0)
        self.assertAlmostEqual(result['shapley']['pbp']['mean'], 0.01)
        # pbp value identical across folds -> zero SE
        self.assertAlmostEqual(result['shapley']['pbp']['se'], 0.0)
        self.assertEqual(result['num_folds'], 3)

    def test_reports_interaction_mean_and_se(self):
        fold_scorers = [self._additive_fold_scorer(0.04, 0.02) for _ in range(3)]
        result = cross_validated_group_ablation(
            GROUPS, BASE, fold_scorers, metric='roc_auc')
        pair = frozenset(['box', 'pbp'])
        self.assertIn(pair, result['interactions'])
        self.assertAlmostEqual(result['interactions'][pair]['mean'], 0.0)

    def test_keeps_per_fold_records(self):
        fold_scorers = [self._additive_fold_scorer(0.04, 0.02) for _ in range(2)]
        result = cross_validated_group_ablation(
            GROUPS, BASE, fold_scorers, metric='roc_auc')
        self.assertEqual(len(result['per_fold']), 2)
        self.assertIn('shapley', result['per_fold'][0])

    def test_mean_scores_standalone_and_loo_aggregate_correctly(self):
        # box value {0.02, 0.06} across folds, pbp 0.02 both -> mean box 0.04
        fold_scorers = [self._additive_fold_scorer(0.02, 0.02),
                        self._additive_fold_scorer(0.06, 0.02)]
        result = cross_validated_group_ablation(
            GROUPS, BASE, fold_scorers, metric='roc_auc')
        # additive game: standalone == leave_one_out == own value, here box mean 0.04
        self.assertAlmostEqual(result['standalone']['box']['mean'], 0.04)
        self.assertAlmostEqual(result['leave_one_out']['box']['mean'], 0.04)
        # mean subset score for {box} = mean over folds of its box value
        self.assertAlmostEqual(result['mean_scores'][frozenset(['box'])], 0.04)
        self.assertAlmostEqual(result['mean_scores'][frozenset()], 0.0)


if __name__ == '__main__':
    unittest.main()
