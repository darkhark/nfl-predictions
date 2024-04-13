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
AWAY_DEF_CUM_AVG_SCORE_CHANGE = 'away_def_cumulative_avg_points_allowed_change'


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
        # get the score for the first 8 weeks of the LAR season
        la_rams = self.schedule_data[
            ((self.schedule_data[HOME_TEAM] == 'LA') | (self.schedule_data[AWAY_TEAM] == 'LA')) &
            (self.schedule_data[SEASON] == 2023)
        ]
        la_rams = la_rams[la_rams[WEEK] <= 8]
        # get the sum of the scores for the first 8 weeks when lar is home or away
        sum_of_points = (
                la_rams[la_rams[HOME_TEAM] == 'LA']['home_score'].sum() +
                la_rams[la_rams[AWAY_TEAM] == 'LA']['away_score'].sum()
        )
        # get the cumulative score for the first 8 weeks
        cumulative_avg_score = sum_of_points / 8
        lar_week_8_cumulative_score = la_rams[AWAY_OFF_CUM_AVG_SCORE].values[-1]
        self.assertEqual(lar_week_8_cumulative_score, cumulative_avg_score)

    def test_cumulative_score_change_off(self):
        # get the score for the first 8 weeks of the LAR season
        la_rams = self.schedule_data[
            ((self.schedule_data[HOME_TEAM] == 'LA') | (self.schedule_data[AWAY_TEAM] == 'LA')) &
            (self.schedule_data[SEASON] == 2023)
        ]
        la_rams_week_7 = la_rams[la_rams[WEEK] <= 7]
        # get the sum of the scores for the first 8 weeks when lar is home or away
        sum_of_points_week_7 = (
                la_rams_week_7[la_rams_week_7[HOME_TEAM] == 'LA']['home_score'].sum() +
                la_rams_week_7[la_rams_week_7[AWAY_TEAM] == 'LA']['away_score'].sum()
        )
        cumulative_avg_score_week_7 = sum_of_points_week_7 / 7
        sum_of_points_week_8 = la_rams[la_rams[WEEK] == 8]['away_score'].values[0] + sum_of_points_week_7
        cumulative_avg_score_week_8 = sum_of_points_week_8 / 8
        change_in_avg_score = cumulative_avg_score_week_8 - cumulative_avg_score_week_7
        lar_week_8_cumulative_score_change = la_rams[la_rams[WEEK] == 8][AWAY_OFF_CUM_AVG_SCORE_CHANGE].values[-1]
        self.assertEqual(lar_week_8_cumulative_score_change, change_in_avg_score)

    def test_cumulative_score_def(self):
        # get the score for the first 8 weeks of the LAR season
        la_rams = self.schedule_data[
            ((self.schedule_data[HOME_TEAM] == 'LA') | (self.schedule_data[AWAY_TEAM] == 'LA')) &
            (self.schedule_data[SEASON] == 2023)
        ]
        la_rams = la_rams[la_rams[WEEK] <= 8]
        # get the sum of the scores for the first 8 weeks when lar is home or away
        sum_of_points = (
                la_rams[la_rams[HOME_TEAM] != 'LA']['home_score'].sum() +
                la_rams[la_rams[AWAY_TEAM] != 'LA']['away_score'].sum()
        )
        # get the cumulative score for the first 8 weeks
        cumulative_avg_score = sum_of_points / 8
        lar_week_8_cumulative_score = la_rams[AWAY_DEF_CUM_AVG_SCORE].values[-1]
        self.assertEqual(lar_week_8_cumulative_score, cumulative_avg_score)

    def test_cumulative_score_change_def(self):
        # get the score for the first 8 weeks of the LAR season
        la_rams = self.schedule_data[
            ((self.schedule_data[HOME_TEAM] == 'LA') | (self.schedule_data[AWAY_TEAM] == 'LA')) &
            (self.schedule_data[SEASON] == 2023)
        ]
        la_rams_week_7 = la_rams[la_rams[WEEK] <= 7]
        # get the sum of the scores for the first 8 weeks when lar is home or away
        sum_of_points_week_7 = (
                la_rams_week_7[la_rams_week_7[HOME_TEAM] != 'LA']['home_score'].sum() +
                la_rams_week_7[la_rams_week_7[AWAY_TEAM] != 'LA']['away_score'].sum()
        )
        cumulative_avg_score_week_7 = sum_of_points_week_7 / 7
        sum_of_points_week_8 = la_rams[la_rams[WEEK] == 8]['home_score'].values[0] + sum_of_points_week_7
        cumulative_avg_score_week_8 = sum_of_points_week_8 / 8
        change_in_avg_score = cumulative_avg_score_week_8 - cumulative_avg_score_week_7
        lar_week_8_cumulative_score_change = la_rams[la_rams[WEEK] == 8][AWAY_DEF_CUM_AVG_SCORE_CHANGE].values[-1]
        self.assertEqual(lar_week_8_cumulative_score_change, change_in_avg_score)

    def test_cumulative_score_change_first_game(self):
        la_rams = self.schedule_data[
            ((self.schedule_data[HOME_TEAM] == 'LA') | (self.schedule_data[AWAY_TEAM] == 'LA')) &
            (self.schedule_data[SEASON] == 2023)
        ]
        la_rams_week_1 = la_rams[la_rams[WEEK] == 1]
        lar_week_1_cumulative_score_change = la_rams_week_1[AWAY_OFF_CUM_AVG_SCORE_CHANGE].values[0]
        self.assertEqual(lar_week_1_cumulative_score_change, 0)

    def test_indoor_is_accurate(self):
        la_rams = self.schedule_data[self.schedule_data[HOME_TEAM] == 'LA']
        self.assertTrue(la_rams['indoor'].all())


if __name__ == '__main__':
    unittest.main()
