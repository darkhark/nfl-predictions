from src.data.schedule import collect
import unittest

HOME_TEAM = 'home_team'
AWAY_TEAM = 'away_team'
WEEK = 'week'
SEASON = 'season'
HOME_DAYS_SINCE_GAME = 'home_days_since_previous_game'
AWAY_DAYS_SINCE_GAME = 'away_days_since_previous_game'
AWAY_OFF_CUM_AVG_SCORE = 'away_off_cumulative_avg_score'
AWAY_OFF_CUM_AVG_SCORE_CHANGE = 'away_off_cumulative_avg_score_change'
AWAY_DEF_CUM_AVG_SCORE = 'away_def_cumulative_avg_points_allowed'
HOME_DEF_CUM_AVG_SCORE = 'home_def_cumulative_avg_points_allowed'
AWAY_DEF_CUM_AVG_SCORE_CHANGE = 'away_def_cumulative_avg_points_allowed_change'
HOME_DEF_CUM_AVG_SCORE_CHANGE = 'home_def_cumulative_avg_points_allowed_change'
HOME_OFF_CUM_AVG_SCORE = 'home_off_cumulative_avg_score'
HOME_OFF_SCORE_RANK = 'home_off_cumulative_avg_score_rank'
AWAY_OFF_SCORE_RANK = 'away_off_cumulative_avg_score_rank'
HOME_OFF_SCORE_RANK_CHANGE = 'home_off_cumulative_avg_score_rank_change'
AWAY_OFF_SCORE_RANK_CHANGE = 'away_off_cumulative_avg_score_rank_change'
HOME_DEF_PA_RANK = 'home_def_cumulative_avg_points_allowed_rank'
AWAY_DEF_PA_RANK = 'away_def_cumulative_avg_points_allowed_rank'


def _stack_team_values(df, value_col):
    """Build a per-team (team, value, rank) frame for one (season, week) slice from the
    wide home/away schedule rows."""
    import pandas as pd
    home = df[[HOME_TEAM, value_col.replace('away_', 'home_'), value_col.replace('away_', 'home_') + '_rank']].copy()
    home.columns = ['team', 'value', 'rank']
    away = df[[AWAY_TEAM, value_col.replace('home_', 'away_'), value_col.replace('home_', 'away_') + '_rank']].copy()
    away.columns = ['team', 'value', 'rank']
    return pd.concat([home, away], ignore_index=True)


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.years = [2022, 2023]
        self.schedule_data = collect.get_schedule_data(self.years)

    def test_week_one_days_since_last_game_is_always_240(self):
        week_one = self.schedule_data[self.schedule_data[WEEK] == 1]
        self.assertEqual(week_one[AWAY_DAYS_SINCE_GAME].unique(), [240])
        self.assertEqual(week_one[HOME_DAYS_SINCE_GAME].unique(), [240])

    def test_days_since_previous_game(self):
        # DET opened on Thursday as the away team and played home on Sunday in week 2
        week_two_det = self.schedule_data[
            (self.schedule_data[WEEK] == 2) &
            (self.schedule_data[HOME_TEAM] == 'DET') &
            (self.schedule_data[SEASON] == 2023)
        ]
        self.assertEqual(week_two_det[HOME_DAYS_SINCE_GAME].values[0], 10)
        # KC played DET as the home team on Thursday in week 1 and played away on Sunday in week 2
        week_two_kc = self.schedule_data[
            (self.schedule_data[WEEK] == 2) & (self.schedule_data[AWAY_TEAM] == 'KC')
        ]
        self.assertEqual(week_two_kc[AWAY_DAYS_SINCE_GAME].values[0], 10)
        # ARI had a bye week in week 14
        week_fourteen_ari = self.schedule_data[
            (self.schedule_data[WEEK] == 15) & (self.schedule_data[HOME_TEAM] == 'ARI')
        ]
        self.assertEqual(week_fourteen_ari[HOME_DAYS_SINCE_GAME].values[0], 14)

    def test_cumulative_score_off(self):
        la_rams = self.schedule_data[
            ((self.schedule_data[HOME_TEAM] == 'LA') | (self.schedule_data[AWAY_TEAM] == 'LA')) &
            (self.schedule_data[SEASON] == 2023)
        ]
        la_rams = la_rams[la_rams[WEEK] <= 18]
        sum_of_points = (
                la_rams[la_rams[HOME_TEAM] == 'LA']['home_score'].sum() +
                la_rams[la_rams[AWAY_TEAM] == 'LA']['away_score'].sum()
        )
        # get the cumulative score for the first 8 weeks
        cumulative_avg_score = sum_of_points / 17
        lar_week_8_cumulative_score = la_rams[AWAY_OFF_CUM_AVG_SCORE].values[-1]
        self.assertEqual(lar_week_8_cumulative_score, cumulative_avg_score)

    def test_cumulative_score_change_off(self):
        la_rams = self.schedule_data[
            ((self.schedule_data[HOME_TEAM] == 'LA') | (self.schedule_data[AWAY_TEAM] == 'LA')) &
            (self.schedule_data[SEASON] == 2023)
        ]
        la_rams_week_7 = la_rams[la_rams[WEEK] <= 17]
        sum_of_points_week_7 = (
                la_rams_week_7[la_rams_week_7[HOME_TEAM] == 'LA']['home_score'].sum() +
                la_rams_week_7[la_rams_week_7[AWAY_TEAM] == 'LA']['away_score'].sum()
        )
        cumulative_avg_score_week_7 = sum_of_points_week_7 / 16
        sum_of_points_week_8 = la_rams[la_rams[WEEK] == 18]['away_score'].values[0] + sum_of_points_week_7
        cumulative_avg_score_week_8 = sum_of_points_week_8 / 17
        change_in_avg_score = cumulative_avg_score_week_8 - cumulative_avg_score_week_7
        lar_week_8_cumulative_score_change = la_rams[la_rams[WEEK] == 18][AWAY_OFF_CUM_AVG_SCORE_CHANGE].values[-1]
        self.assertEqual(lar_week_8_cumulative_score_change, change_in_avg_score)

    def test_cumulative_score_def(self):
        la_rams = self.schedule_data[
            ((self.schedule_data[HOME_TEAM] == 'LA') | (self.schedule_data[AWAY_TEAM] == 'LA')) &
            (self.schedule_data[SEASON] == 2023)
        ]
        la_rams = la_rams[la_rams[WEEK] <= 18]
        sum_of_points = (
                la_rams[la_rams[HOME_TEAM] != 'LA']['home_score'].sum() +
                la_rams[la_rams[AWAY_TEAM] != 'LA']['away_score'].sum()
        )
        cumulative_avg_score = sum_of_points / 17
        lar_week_16_cumulative_score = la_rams[HOME_DEF_CUM_AVG_SCORE].values[-1]
        self.assertEqual(lar_week_16_cumulative_score, cumulative_avg_score)

    def test_cumulative_score_change_def(self):
        # get the score for the first 8 weeks of the LAR season
        la_rams = self.schedule_data[
            ((self.schedule_data[HOME_TEAM] == 'LA') | (self.schedule_data[AWAY_TEAM] == 'LA')) &
            (self.schedule_data[SEASON] == 2023)
        ]
        la_rams_week_17 = la_rams[la_rams[WEEK] <= 17]
        # get the sum of the scores for the first 16 weeks when lar is home or away
        sum_of_points_week_17 = (
                la_rams_week_17[la_rams_week_17[HOME_TEAM] != 'LA']['home_score'].sum() +
                la_rams_week_17[la_rams_week_17[AWAY_TEAM] != 'LA']['away_score'].sum()
        )
        cumulative_avg_score_week_17 = sum_of_points_week_17 / 16
        sum_of_points_week_18 = la_rams[la_rams[WEEK] == 18]['home_score'].values[0] + sum_of_points_week_17
        cumulative_avg_score_week_18 = sum_of_points_week_18 / 17
        change_in_avg_score = cumulative_avg_score_week_18 - cumulative_avg_score_week_17
        lar_week_18_cumulative_score_change = la_rams[la_rams[WEEK] == 18][AWAY_DEF_CUM_AVG_SCORE_CHANGE].values[-1]
        self.assertEqual(lar_week_18_cumulative_score_change, change_in_avg_score)

    def test_cumulative_score_change_first_game(self):
        la_rams = self.schedule_data[
            ((self.schedule_data[HOME_TEAM] == 'LA') | (self.schedule_data[AWAY_TEAM] == 'LA')) &
            (self.schedule_data[SEASON] == 2023)
        ]
        la_rams_week_1 = la_rams[la_rams[WEEK] == 1]
        lar_week_1_cumulative_score_change = la_rams_week_1[AWAY_OFF_CUM_AVG_SCORE_CHANGE].values[0]
        self.assertEqual(lar_week_1_cumulative_score_change, 0)

    def test_offense_points_rank_one_holds_the_max_cumulative_average(self):
        week_ten = self.schedule_data[
            (self.schedule_data[WEEK] == 10) & (self.schedule_data[SEASON] == 2023)
        ]
        teams = _stack_team_values(week_ten, HOME_OFF_CUM_AVG_SCORE)
        rank_one = teams[teams['rank'] == 1]
        self.assertFalse(rank_one.empty)
        for value in rank_one['value']:
            self.assertEqual(value, teams['value'].max())

    def test_defense_points_allowed_rank_one_holds_the_min_cumulative_average(self):
        week_ten = self.schedule_data[
            (self.schedule_data[WEEK] == 10) & (self.schedule_data[SEASON] == 2023)
        ]
        teams = _stack_team_values(week_ten, HOME_DEF_CUM_AVG_SCORE)
        rank_one = teams[teams['rank'] == 1]
        self.assertFalse(rank_one.empty)
        for value in rank_one['value']:
            self.assertEqual(value, teams['value'].min())

    def test_offense_points_rank_change_equals_consecutive_week_diff(self):
        la_rams = self.schedule_data[
            ((self.schedule_data[HOME_TEAM] == 'LA') | (self.schedule_data[AWAY_TEAM] == 'LA')) &
            (self.schedule_data[SEASON] == 2023)
        ].sort_values(WEEK)
        previous_rank = None
        for _, row in la_rams.iterrows():
            is_home = row[HOME_TEAM] == 'LA'
            rank = row[HOME_OFF_SCORE_RANK] if is_home else row[AWAY_OFF_SCORE_RANK]
            change = row[HOME_OFF_SCORE_RANK_CHANGE] if is_home else row[AWAY_OFF_SCORE_RANK_CHANGE]
            if previous_rank is None:
                self.assertEqual(change, 0)
            else:
                self.assertEqual(change, rank - previous_rank)
            previous_rank = rank

    def test_schedule_data_has_a_single_week_column(self):
        self.assertEqual(list(self.schedule_data.columns).count(WEEK), 1)

    def test_indoor_is_accurate(self):
        la_rams = self.schedule_data[self.schedule_data[HOME_TEAM] == 'LA']
        self.assertTrue(la_rams['indoor'].all())


class TestScheduleRecency(unittest.TestCase):

    def test_ewma_rolling_avg_score(self):
        import pandas as pd
        team_df = pd.DataFrame({
            'team': ['AAA'] * 4, 'season': [2023] * 4, 'week': [1, 2, 3, 4],
            'score': [10.0, 20.0, 30.0, 40.0],
        })
        out = collect._calculate_cumulative_avg_score(team_df, offense=True)
        s = pd.Series([10.0, 20.0, 30.0, 40.0])
        self.assertTrue((abs(out['ewma_avg_score'].to_numpy()
                             - s.ewm(halflife=3, adjust=True).mean().to_numpy()) < 1e-9).all())
        self.assertTrue((abs(out['rolling_avg_score'].to_numpy()
                             - s.rolling(4, min_periods=1).mean().to_numpy()) < 1e-9).all())

    def test_recency_columns_get_ranked(self):
        import pandas as pd
        team_df = pd.DataFrame({
            'team': ['AAA'] * 4, 'season': [2023] * 4, 'week': [1, 2, 3, 4],
            'score': [10.0, 20.0, 30.0, 40.0],
        })
        out = collect._calculate_cumulative_avg_score(team_df, offense=True)
        self.assertIn('ewma_avg_score_rank', out.columns)
        self.assertIn('rolling_avg_score_rank_change', out.columns)


class TestScheduleRecencyMerge(unittest.TestCase):
    def setUp(self):
        self.schedule_data = collect.get_schedule_data([2022, 2023])

    def test_merged_frame_has_recency_families(self):
        for col in (
            'home_off_ewma_avg_score', 'away_off_ewma_avg_score',
            'home_off_rolling_avg_score', 'away_off_rolling_avg_score',
            'home_def_ewma_avg_points_allowed', 'home_def_rolling_avg_points_allowed',
            'home_off_ewma_avg_score_rank', 'home_off_rolling_avg_score_rank_change',
        ):
            self.assertIn(col, self.schedule_data.columns)


if __name__ == '__main__':
    unittest.main()
