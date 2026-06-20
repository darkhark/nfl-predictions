import unittest

import numpy as np
import pandas as pd

from data_science_utilities.models.bayes_logistic import data_prep

# The 54 features of the BART Run-6 set (b1410bd rfe_features_kfolds.csv .loc[54]).
EXPECTED_54 = [
    "is_home_target",
    "target_def_cumulative_avg_points_allowed_rank",
    "opp_def_cumulative_avg_points_allowed_rank",
    "off_opp_passing_epa_cumulative_average_rank",
    "target_off_cumulative_avg_score_rank",
    "opp_off_cumulative_avg_score_rank",
    "off_target_passing_epa_cumulative_average_rank",
    "off_target_pacr",
    "def_opp_rushing_yards_cumulative_average_rank",
    "off_target_interceptions_cumulative_average_rank",
    "def_target_carries_cumulative_average_rank",
    "def_target_rushing_fumbles_lost_cumulative_average_rank",
    "def_opp_attempts",
    "def_target_attempts",
    "def_target_rushing_yards_cumulative_average_rank",
    "def_opp_rushing_tds_cumulative_average_rank",
    "def_opp_racr",
    "def_target_rushing_tds_cumulative_average_rank",
    "off_target_sack_yards_cumulative_average_rank",
    "opp_game_count",
    "def_opp_rushing_first_downs_cumulative_average_rank",
    "off_opp_passing_epa",
    "off_target_passing_epa",
    "off_opp_rushing_epa",
    "def_opp_pacr_cumulative_average_rank_change",
    "off_target_passing_yards_after_catch",
    "off_target_passing_tds_cumulative_average_rank",
    "off_opp_sacks_cumulative_average_rank",
    "off_opp_carries_cumulative_average_rank",
    "off_target_sacks_cumulative_average_rank",
    "off_opp_receiving_fumbles_cumulative_average_rank",
    "def_opp_passing_tds_cumulative_average_rank",
    "def_target_receiving_fumbles_cumulative_average_rank_change",
    "off_opp_interceptions_cumulative_average_rank",
    "off_opp_passing_yards_after_catch",
    "off_opp_sack_yards_cumulative_average_rank",
    "def_opp_passing_yards",
    "def_target_sack_yards",
    "def_target_interceptions_cumulative_average_rank",
    "off_target_attempts_cumulative_average_rank_change",
    "def_opp_receiving_fumbles_cumulative_average_rank",
    "def_target_sacks_cumulative_average_rank",
    "def_opp_sack_yards",
    "def_target_rushing_fumbles_cumulative_average_rank",
    "off_opp_rushing_first_downs_cumulative_average_rank",
    "off_target_sack_fumbles_cumulative_average_rank",
    "off_target_rushing_tds_cumulative_average_rank",
    "off_target_passing_yards",
    "def_target_passing_tds_cumulative_average_rank",
    "def_opp_sack_fumbles_cumulative_average_rank",
    "off_target_carries_cumulative_average_rank_change",
    "def_target_sack_yards_cumulative_average_rank",
    "def_opp_passing_epa",
    "def_target_passing_epa",
]

LEAKAGE_COLS = {
    "game_id", "season", "season_type", "opp_team", "opp_score",
    "target_team", "target_score", "h_win",
}


class TestLoadFeatureList(unittest.TestCase):
    def test_returns_exact_54_run6_features_in_order(self):
        feats = data_prep.load_feature_list(data_prep.FEATURE_LIST_PATH)
        self.assertEqual(feats, EXPECTED_54)

    def test_no_leakage_columns_present(self):
        feats = data_prep.load_feature_list(data_prep.FEATURE_LIST_PATH)
        self.assertEqual(set(feats) & LEAKAGE_COLS, set())

    def test_all_features_exist_in_parquet(self):
        feats = data_prep.load_feature_list(data_prep.FEATURE_LIST_PATH)
        cols = pd.read_parquet(data_prep.PARQUET_PATH, columns=None).columns
        missing = [f for f in feats if f not in cols]
        self.assertEqual(missing, [])


def _synthetic_seasons():
    # two rows per season 2019..2025 so splits are non-empty
    rows = []
    for season in range(2019, 2026):
        for win in (0, 1):
            rows.append({"season": season, "target_win": win, "week": 1, "feat": season + win})
    return pd.DataFrame(rows)


class TestLoadAndSplit(unittest.TestCase):
    def test_split_boundaries(self):
        df = _synthetic_seasons()
        split = data_prep.split_frames(df)  # split a given (already shuffled) frame
        self.assertTrue((split.train["season"] < 2022).all())
        self.assertTrue(((split.valid["season"] >= 2022) & (split.valid["season"] < 2024)).all())
        self.assertTrue((split.holdout["season"] >= 2024).all())
        # every row lands in exactly one fold
        self.assertEqual(len(split.train) + len(split.valid) + len(split.holdout), len(df))

    def test_seed_is_reproducible(self):
        df = _synthetic_seasons()
        a = data_prep.shuffle(df, seed=32)
        b = data_prep.shuffle(df, seed=32)
        pd.testing.assert_frame_equal(a, b)

    def test_get_target_is_int_array(self):
        df = _synthetic_seasons()
        y = data_prep.get_target(df)
        self.assertEqual(y.dtype, np.dtype(int))
        self.assertEqual(set(np.unique(y)), {0, 1})


class TestLoadAndSplitRealData(unittest.TestCase):
    def test_holdout_is_2024_plus_2025(self):
        split = data_prep.load_and_split()
        self.assertEqual(sorted(split.holdout["season"].unique().tolist()), [2024, 2025])
        self.assertTrue((split.train["season"] < 2022).all())


class TestTrainFitPreprocessor(unittest.TestCase):
    def _frame(self, vals, extra=None):
        d = {"a": vals}
        if extra is not None:
            d["b"] = extra
        return pd.DataFrame(d)

    def test_standardizes_using_train_moments_only(self):
        train = self._frame([0.0, 2.0, 4.0])           # mean 2, std 2 (ddof=1)
        pre = data_prep.TrainFitPreprocessor(["a"]).fit(train)
        # a *different* frame must be standardized with TRAIN mean/std, not its own
        other = self._frame([2.0, 2.0, 2.0])
        out = pre.transform(other)
        np.testing.assert_allclose(out[:, 0], [0.0, 0.0, 0.0])  # (2-2)/2 == 0
        np.testing.assert_allclose(pre.means_, [2.0])
        np.testing.assert_allclose(pre.stds_, [2.0])

    def test_median_impute_and_missing_flag_on_train_nan(self):
        train = self._frame([1.0, np.nan, 3.0])         # train median 2.0
        pre = data_prep.TrainFitPreprocessor(["a"]).fit(train)
        self.assertIn("a_was_missing", pre.feature_names_out_)
        test = self._frame([np.nan, 5.0, 5.0])
        out = pre.transform(test)
        flag_idx = pre.feature_names_out_.index("a_was_missing")
        np.testing.assert_allclose(out[:, flag_idx], [1.0, 0.0, 0.0])  # flag from raw NaN
        # imputed value standardized: first row used train median 2.0 before scaling
        base_idx = pre.feature_names_out_.index("a")
        mean_a, std_a = pre.means_[base_idx], pre.stds_[base_idx]
        np.testing.assert_allclose(out[0, base_idx], (2.0 - mean_a) / std_a)

    def test_no_flag_when_train_has_no_nan(self):
        train = self._frame([1.0, 2.0, 3.0])
        pre = data_prep.TrainFitPreprocessor(["a"]).fit(train)
        self.assertEqual(pre.feature_names_out_, ["a"])  # no flag column

    def test_zero_variance_guard(self):
        train = self._frame([5.0, 5.0, 5.0])            # std 0
        pre = data_prep.TrainFitPreprocessor(["a"]).fit(train)
        out = pre.transform(train)
        self.assertFalse(np.isnan(out).any())
        np.testing.assert_allclose(out[:, 0], [0.0, 0.0, 0.0])

    def test_transform_output_has_no_nan(self):
        train = self._frame([1.0, np.nan, 3.0], extra=[1.0, 2.0, 3.0])
        pre = data_prep.TrainFitPreprocessor(["a", "b"]).fit(train)
        out = pre.transform(self._frame([np.nan, 1.0, 2.0], extra=[9.0, 8.0, 7.0]))
        self.assertFalse(np.isnan(out).any())


if __name__ == "__main__":
    unittest.main()
