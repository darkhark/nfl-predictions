import unittest

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


if __name__ == "__main__":
    unittest.main()


import numpy as np


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
