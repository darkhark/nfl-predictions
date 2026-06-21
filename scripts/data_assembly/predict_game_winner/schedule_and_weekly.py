# %%
import os
import pandas as pd

from src.data.collect_all import get_schedule_and_weekly_data

# %%
# In 1999 and 2002, there are a couple games with nul values in the stats.
# TODO: Investigate why these are null
YEARS = [year for year in range(2003, 2026)]
MODULE_PATH = os.path.abspath(os.path.join('.'))
MODEL_FEATURES_LIST_PATH = os.path.join(
    MODULE_PATH, 'data', 'predict_games', 'model_features_in', 'xgb_features_list.csv'
)

# %%
schedule_and_weekly_df = get_schedule_and_weekly_data(YEARS, include_play_by_play=True)
schedule_and_weekly_df.head()

# %%
TARGET = 'target_win'
# 'season' is excluded as a candidate feature: hold-out seasons lie outside the training
# range, so tree splits on it can only encode era drift, never anything that generalizes
# forward. It stays in the parquet for the time-based train/hold-out splits. 'week' stays
# a candidate (in-range, can carry real late-season effects); RFE decides its fate.
EXCLUDE_COLUMNS_LIST = [
    'game_id', 'opp_team', 'opp_score', 'target_team', 'target_score', 'season_type',
    'h_win', 'season'
]

# %%
# Remove all records where week == 1 and season == 2003
# This is because there are no previous games to calculate all the stats
schedule_and_weekly_df = schedule_and_weekly_df[
    (schedule_and_weekly_df['week'] != 1) | (schedule_and_weekly_df['season'] != 2003)
]

# %%
# Remove any records where game_type != 'REG'.
# NOTE ON ORDERING: this filter runs AFTER the one-week leakage shift inside
# get_schedule_and_weekly_data (collect_all._shift_data). So postseason rows are present
# during the shift and only dropped here. The effect is that a regular-season week-1 row
# inherits the team's last PRIOR game including playoffs -- their final playoff game if they
# made the postseason, otherwise their regular-season finale -- rather than starting fresh.
# The postseason rows themselves are removed below, but the stats they shifted forward into
# the surviving week-1 REG rows stay. Every season opener from 2004 on carries these values;
# only 2003 week-1 (dropped above) is truly empty. To make season openers start fresh, group
# the shift by season and apply this REG filter before the shift. See _shift_data's docstring.
schedule_and_weekly_df = schedule_and_weekly_df[schedule_and_weekly_df['game_type'] == 'REG']
schedule_and_weekly_df.drop(columns=['game_type', 'season_type'], inplace=True)

MODEL_FEATURES_LIST = [
    col for col in schedule_and_weekly_df.columns if col not in EXCLUDE_COLUMNS_LIST
]

# %%
# if opp_days_since_previous_game or target_days_since_previous_game is the only NaN in the row,
# drop the record
# Each of these have been checked and it's due to week one byes before 2017 and a hurricane in 2017
schedule_and_weekly_df.dropna(
    subset=['opp_days_since_previous_game', 'target_days_since_previous_game'],
    inplace=True
)

# %%
# Show all rows with any NaNs
schedule_and_weekly_df[schedule_and_weekly_df.isna().any(axis=1)]

# %%
# Save MODEL_FEATURES_LIST to a csv
MODEL_FEATURES_LIST_DF = pd.DataFrame(MODEL_FEATURES_LIST, columns=['feature'])
MODEL_FEATURES_LIST_DF.to_csv(MODEL_FEATURES_LIST_PATH, index=False)

# %%
# Save the schedule and weekly data to a parquet file
SCHEDULE_AND_WEEKLY_PATH = os.path.join(
    MODULE_PATH, 'data', 'predict_games', 'input_data', 'schedule_and_weekly.parquet'
)
schedule_and_weekly_df.to_parquet(SCHEDULE_AND_WEEKLY_PATH, index=False)
