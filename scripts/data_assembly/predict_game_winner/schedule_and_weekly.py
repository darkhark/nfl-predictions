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
schedule_and_weekly_df = get_schedule_and_weekly_data(YEARS)
schedule_and_weekly_df.head()

# %%
TARGET = 'target_win'
EXCLUDE_COLUMNS_LIST = [
    'game_id', 'opp_team', 'opp_score', 'target_team', 'target_score', 'season_type',
    'h_win'
]

# %%
# Remove all records where week == 1 and season == 2003
# This is because there are no previous games to calculate all the stats
schedule_and_weekly_df = schedule_and_weekly_df[
    (schedule_and_weekly_df['week'] != 1) | (schedule_and_weekly_df['season'] != 2003)
]

# %%
# Remove any records where game_type != 'REG'
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
