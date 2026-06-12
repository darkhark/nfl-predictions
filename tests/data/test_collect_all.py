import unittest
import pandas as pd

from src.data import collect_all
from src.data.weekly import collect as weekly_collect
from src.data.schedule import collect as schedule_collect


class MyTestCase(unittest.TestCase):

    BYE_WEEK_2022 = 7
    BYE_WEEK_2023 = 10
    def setUp(self):
        self.years = [2022, 2023]
        self.before_shift_df = self._get_data_before_shift(self.years)
        self.stats_cols = self._get_stats_columns()
        self.all_data = collect_all.get_schedule_and_weekly_data(self.years)
        self.all_data.set_index(['game_id', 'is_home_target'], inplace=True)
        self.before_shift_df.set_index(['game_id', 'is_home_target'], inplace=True)
        # display 20 columns
        pd.set_option('display.max_columns', 20)

    def test_week_after_target_team_stats_equal_week_1_stats_in_weekly(self):
        """
        Checks that the stats for the target team before the shift match the next week after the shift.
        :return:
        """
        for week in range(1, 18):
            earlier_week_data = self.before_shift_df[self.before_shift_df['week'] == week][self.stats_cols + ['target_team', 'season']]
            week_after_data = self.all_data[self.all_data['week'] == week + 1][self.stats_cols + ['target_team', 'season']]
            target_stats_cols = [col for col in self.stats_cols if 'target' in col]
            earlier_week_data_target_la = earlier_week_data[earlier_week_data['target_team'] == 'LA'][['season'] + target_stats_cols]
            week_after_data_target_la = week_after_data[week_after_data['target_team'] == 'LA'][['season'] + target_stats_cols]
            earlier_week_data_target_la.reset_index(drop=True, inplace=True)
            week_after_data_target_la.reset_index(drop=True, inplace=True)
            # make the index the season
            earlier_week_data_target_la.set_index('season', inplace=True)
            week_after_data_target_la.set_index('season', inplace=True)
            # sort by the index
            earlier_week_data_target_la.sort_index(inplace=True)
            week_after_data_target_la.sort_index(inplace=True)
            # drop the bye week row from 2022 and 2023 for the correct season
            if week == self.BYE_WEEK_2022 or week + 1 == self.BYE_WEEK_2022:
                # check if the season is in the index before dropping
                if 2022 in earlier_week_data_target_la.index:
                    earlier_week_data_target_la = earlier_week_data_target_la.drop(2022)
                if 2022 in week_after_data_target_la.index:
                    week_after_data_target_la = week_after_data_target_la.drop(2022)
            if week == self.BYE_WEEK_2023 or week + 1 == self.BYE_WEEK_2023:
                # check if the season is in the index before dropping
                if 2023 in earlier_week_data_target_la.index:
                    earlier_week_data_target_la = earlier_week_data_target_la.drop(2023)
                if 2023 in week_after_data_target_la.index:
                    week_after_data_target_la = week_after_data_target_la.drop(2023)
            diff_df_target = earlier_week_data_target_la[earlier_week_data_target_la != week_after_data_target_la]
            # print only the columns where not all values are NaN
            diff_df_target = diff_df_target[diff_df_target.columns[diff_df_target.notna().any()]]
            self.assertTrue(diff_df_target.empty)

    def test_week_2_opp_team_stats_in_all_equal_week_1_stats_in_weekly(self):
        for week in range(1, 18):
            week_one_data = self.before_shift_df[self.before_shift_df['week'] == week][self.stats_cols + ['opp_team', 'season']]
            week_two_data = self.all_data[self.all_data['week'] == week + 1][self.stats_cols + ['opp_team', 'season']]
            non_target_stats_cols = [col for col in self.stats_cols if 'opp' in col]
            week_one_data_opp_la = week_one_data[week_one_data['opp_team'] == 'LA'][['season'] + non_target_stats_cols]
            week_two_data_opp_la = week_two_data[week_two_data['opp_team'] == 'LA'][['season'] + non_target_stats_cols]
            week_one_data_opp_la.reset_index(drop=True, inplace=True)
            week_two_data_opp_la.reset_index(drop=True, inplace=True)
            # make the index the season
            week_one_data_opp_la.set_index('season', inplace=True)
            week_two_data_opp_la.set_index('season', inplace=True)
            # sort by the index
            week_one_data_opp_la.sort_index(inplace=True)
            week_two_data_opp_la.sort_index(inplace=True)
            # drop the bye week row from 2022 and 2023 for the correct season
            if week == self.BYE_WEEK_2022 or week + 1 == self.BYE_WEEK_2022:
                # check if the season is in the index before dropping
                if 2022 in week_one_data_opp_la.index:
                    week_one_data_opp_la = week_one_data_opp_la.drop(2022)
                if 2022 in week_two_data_opp_la.index:
                    week_two_data_opp_la = week_two_data_opp_la.drop(2022)
            if week == self.BYE_WEEK_2023 or week + 1 == self.BYE_WEEK_2023:
                # check if the season is in the index before dropping
                if 2023 in week_one_data_opp_la.index:
                    week_one_data_opp_la = week_one_data_opp_la.drop(2023)
                if 2023 in week_two_data_opp_la.index:
                    week_two_data_opp_la = week_two_data_opp_la.drop(2023)
            diff_df_opp = week_one_data_opp_la[week_one_data_opp_la != week_two_data_opp_la]
            # print only the columns where not all values are NaN
            diff_df_opp = diff_df_opp[diff_df_opp.columns[diff_df_opp.notna().any()]]
            self.assertTrue(diff_df_opp.empty)

    def test_rank_columns_are_routed_to_all_four_families(self):
        expected_columns = [
            'off_target_passing_yards_cumulative_average_rank',
            'off_opp_passing_yards_cumulative_average_rank',
            'def_target_passing_yards_cumulative_average_rank',
            'def_opp_passing_yards_cumulative_average_rank',
            'target_off_cumulative_avg_score_rank',
            'opp_def_cumulative_avg_points_allowed_rank',
        ]
        for column in expected_columns:
            self.assertIn(column, self.all_data.columns)

    def test_rank_column_is_shifted_one_game_forward(self):
        rank_col = 'def_target_passing_yards_cumulative_average_rank'
        for week in range(1, 18):
            if week == self.BYE_WEEK_2023 or week + 1 == self.BYE_WEEK_2023:
                continue
            before = self.before_shift_df[
                (self.before_shift_df['week'] == week) &
                (self.before_shift_df['target_team'] == 'LA') &
                (self.before_shift_df['season'] == 2023)
            ][rank_col]
            after = self.all_data[
                (self.all_data['week'] == week + 1) &
                (self.all_data['target_team'] == 'LA') &
                (self.all_data['season'] == 2023)
            ][rank_col]
            if before.empty or after.empty:
                continue
            self.assertEqual(before.values[0], after.values[0])

    def _get_stats_columns(self):
        stats_data_cols = [col for col in self.before_shift_df.columns if 'target' in col or 'opp' in col]
        cols_not_stats = [
            'target_days_since_previous_game', 'opp_days_since_previous_game',
            'target_game_count', 'opp_game_count', 'target_team', 'opp_team',
            'target_score', 'opp_score', 'is_home_target'
        ]
        for col in cols_not_stats:
            stats_data_cols.remove(col)
        return stats_data_cols

    def _get_data_before_shift(self, years):
        weekly_data = weekly_collect.get_weekly_data(years).reset_index(drop=True)
        schedule_data = schedule_collect.get_schedule_data(years).reset_index(drop=True)

        home_team_is_team = self._get_home_or_away_team_in_weekly_data(schedule_data, weekly_data, is_home=True)
        away_team_is_team = self._get_home_or_away_team_in_weekly_data(schedule_data, weekly_data, is_home=False)
        combined_data = pd.merge(
            home_team_is_team,
            away_team_is_team,
            on=['season', 'week', 'home_team', 'away_team'],
            suffixes=('', '_y')
        )
        # drop the columns with _y suffix
        combined_data = combined_data[
            [col for col in combined_data.columns if not col.endswith('_y')]
        ]

        # duplicate each row, but change either the home or away team to the target team
        home_is_target = self._convert_team_to_target(combined_data, is_home_target=True)
        away_is_target = self._convert_team_to_target(combined_data, is_home_target=False)
        return pd.concat([home_is_target, away_is_target], ignore_index=True)

    def _get_home_or_away_team_in_weekly_data(self, schedule_data, weekly_data, is_home=True):
        if is_home:
            team_type = 'home'
            opp_team_type = 'away'
        else:
            team_type = 'away'
            opp_team_type = 'home'
        team_df = schedule_data.merge(
            weekly_data,
            left_on=[f'{team_type}_team', 'season', 'week'],
            right_on=['team', 'season', 'week'],
            how='inner'
        )
        # replace the 'team' column with 'home_team'
        team_df.drop(columns=['team', 'opp_team'], inplace=True)
        team_df = team_df.rename(
            columns=lambda x: x.replace('opp', opp_team_type) if 'opp' in x else x
        )
        team_df = team_df.rename(
            columns=lambda x: x.replace('off', f'off_{team_type}') if x.startswith('off') else x
        )
        team_df = team_df.rename(
            columns=lambda x: x.replace('team_', f'{team_type}_') if x.startswith('team_') else x
        )
        return team_df

    def _convert_team_to_target(self, df, is_home_target: bool):
        if is_home_target:
            team_type = 'home'
            opp_team_type = 'away'
        else:
            team_type = 'away'
            opp_team_type = 'home'
        target_df = df.rename(
            columns=lambda x: x.replace(team_type, 'target') if team_type in x else x
        )
        target_df = target_df.rename(
            columns=lambda x: x.replace(opp_team_type, 'opp') if opp_team_type in x else x
        )
        target_df['is_home_target'] = 1 if is_home_target else 0
        return target_df


class TestPlayByPlayIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.all_data = collect_all.get_schedule_and_weekly_data([2023], include_play_by_play=True)

    def test_play_by_play_columns_reach_all_four_families(self):
        expected_columns = [
            'off_target_epa_per_play_competitive_cumulative_average_rank',
            'off_opp_epa_per_play_competitive_cumulative_average_rank',
            'def_target_epa_per_play_garbage_trailing_cumulative_average_rank',
            'def_opp_epa_per_play_garbage_trailing_cumulative_average_rank',
        ]
        for column in expected_columns:
            self.assertIn(column, self.all_data.columns)

    def test_play_by_play_features_are_populated_after_week_two(self):
        late_weeks = self.all_data[self.all_data['week'] >= 3]
        non_null_fraction = late_weeks[
            'off_target_epa_per_play_competitive_cumulative_average'
        ].notna().mean()
        self.assertGreater(non_null_fraction, 0.95)

    def test_play_by_play_features_are_shifted_off_week_one(self):
        # The leakage shift moves every stat forward one game, so week 1 (a team's first
        # game) must have no play-by-play feature values. Unshifted columns would be
        # populated here.
        week_one = self.all_data[self.all_data['week'] == 1]
        self.assertTrue(
            week_one['off_target_epa_per_play_competitive_cumulative_average'].isna().all()
        )

    def test_excluding_play_by_play_keeps_legacy_columns_only(self):
        legacy = collect_all.get_schedule_and_weekly_data([2023])
        pbp_columns = [col for col in legacy.columns if 'epa_per_play' in col]
        self.assertEqual(pbp_columns, [])


if __name__ == '__main__':
    unittest.main()
