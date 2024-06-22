# %%
import pandas as pd

from src.data.collect_all import get_schedule_and_weekly_data

# %%
YEARS = [year for year in range(1999, 2024)]

# %%
schedule_and_weekly_df = get_schedule_and_weekly_data(YEARS)
schedule_and_weekly_df.head()

# %%
# Remove all records where week == 1 and season == 1999
schedule_and_weekly_df = schedule_and_weekly_df[
    (schedule_and_weekly_df['week'] != 1) | (schedule_and_weekly_df['season'] != 1999)
]

# %%
# Remove any records where game_type != 'REG'
schedule_and_weekly_df = schedule_and_weekly_df[schedule_and_weekly_df['game_type'] == 'REG']

# %%
# Remove all NaNs
schedule_and_weekly_df.dropna(inplace=True)

# %%