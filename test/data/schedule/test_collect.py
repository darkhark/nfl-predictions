from src.data.schedule import collect
import unittest


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.years = [2022, 2023]
        self.schedule_data = collect.get_schedule_data(self.years)

    def test_week_one_days_since_last_game_is_always_240(self):
        week_one = self.schedule_data[self.schedule_data['week'] == 1]
        self.assertEqual(week_one['away_days_since_previous_game'].unique(), [240])
        self.assertEqual(week_one['home_days_since_previous_game'].unique(), [240])

    def test_days_since_previous_game(self):
        # DET opened on Thursday as the away team and played home on Sunday in week 2
        week_two_det = self.schedule_data[
            (self.schedule_data['week'] == 2) &
            (self.schedule_data['home_team'] == 'DET') &
            (self.schedule_data['season'] == 2023)
        ]
        self.assertEqual(week_two_det['home_days_since_previous_game'].values[0], 10)
        # KC played DET as the home team on Thursday in week 1 and played away on Sunday in week 2
        week_two_kc = self.schedule_data[
            (self.schedule_data['week'] == 2) & (self.schedule_data['away_team'] == 'KC')
        ]
        self.assertEqual(week_two_kc['away_days_since_previous_game'].values[0], 10)
        # ARI had a bye week in week 14
        week_fourteen_ari = self.schedule_data[
            (self.schedule_data['week'] == 15) & (self.schedule_data['home_team'] == 'ARI')
        ]
        self.assertEqual(week_fourteen_ari['home_days_since_previous_game'].values[0], 14)

    def test_cumulative_score(self):
        # get the score for the first 8 weeks of the LAR season
        la_rams = self.schedule_data[
            ((self.schedule_data['home_team'] == 'LA') | (self.schedule_data['away_team'] == 'LA')) &
            (self.schedule_data['season'] == 2023)
        ]
        la_rams = la_rams[la_rams['week'] <= 8]
        # get the sum of the scores for the first 8 weeks when lar is home or away
        sum_of_points = (
                la_rams[la_rams['home_team'] == 'LA']['home_score'].sum() +
                la_rams[la_rams['away_team'] == 'LA']['away_score'].sum()
        )
        # get the cumulative score for the first 8 weeks
        cumulative_avg_score = sum_of_points / 8
        lar_week_8_cumulative_score = la_rams['away_cumulative_avg_score'].values[-1]
        self.assertEqual(lar_week_8_cumulative_score, cumulative_avg_score)

    def test_cumulative_score_change(self):
        # get the score for the first 8 weeks of the LAR season
        la_rams = self.schedule_data[
            ((self.schedule_data['home_team'] == 'LA') | (self.schedule_data['away_team'] == 'LA')) &
            (self.schedule_data['season'] == 2023)
        ]
        la_rams_week_7 = la_rams[la_rams['week'] <= 7]
        # get the sum of the scores for the first 8 weeks when lar is home or away
        sum_of_points_week_7 = (
                la_rams_week_7[la_rams_week_7['home_team'] == 'LA']['home_score'].sum() +
                la_rams_week_7[la_rams_week_7['away_team'] == 'LA']['away_score'].sum()
        )
        cumulative_avg_score_week_7 = sum_of_points_week_7 / 7
        sum_of_points_week_8 = la_rams[la_rams['week'] == 8]['away_score'].values[0] + sum_of_points_week_7
        cumulative_avg_score_week_8 = sum_of_points_week_8 / 8
        change_in_avg_score = cumulative_avg_score_week_8 - cumulative_avg_score_week_7
        lar_week_8_cumulative_score_change = la_rams[la_rams['week'] == 8]['away_cumulative_avg_score_change'].values[-1]
        self.assertEqual(lar_week_8_cumulative_score_change, change_in_avg_score)

    def test_cumulative_score_change_first_game(self):
        la_rams = self.schedule_data[
            ((self.schedule_data['home_team'] == 'LA') | (self.schedule_data['away_team'] == 'LA')) &
            (self.schedule_data['season'] == 2023)
        ]
        la_rams_week_1 = la_rams[la_rams['week'] == 1]
        lar_week_1_cumulative_score_change = la_rams_week_1['away_cumulative_avg_score_change'].values[0]
        self.assertEqual(lar_week_1_cumulative_score_change, 0)

    def test_indoor_is_accurate(self):
        la_rams = self.schedule_data[self.schedule_data['home_team'] == 'LA']
        self.assertTrue(la_rams['indoor'].all())


if __name__ == '__main__':
    unittest.main()
