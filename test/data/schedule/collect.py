from src.data.schedule import collect
import unittest


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.years = [2023]
        self.schedule_data = collect.get_schedule_data(self.years)

    def test_week_one_days_since_last_game_is_always_240(self):
        week_one = self.schedule_data[self.schedule_data['week'] == 1]
        self.assertEqual(week_one['away_days_since_previous_game'].unique(), [240])
        self.assertEqual(week_one['home_days_since_previous_game'].unique(), [240])

    def test_days_since_previous_game(self):
        # DET opened on Thursday as the away team and played home on Sunday in week 2
        week_two_det = self.schedule_data[
            (self.schedule_data['week'] == 2) & (self.schedule_data['home_team'] == 'DET')
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


if __name__ == '__main__':
    unittest.main()
