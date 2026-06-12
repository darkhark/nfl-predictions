import os

import nfl_data_py as nfl
import pandas as pd

from src.data import transformations
from src.data.transformations import (
    TEAM_ABBR_MAPPINGS, TEAM_COL, OPPONENT_TEAM_COL, SEASON_COL, WEEK_COL,
    TEAM_GAME_COUNT_COL, OPP_GAME_COUNT_COL,
)

CACHE_DIR = os.path.join('data', 'play_by_play', 'aggregated')

SEASON_TYPE_COL = 'season_type'
CONTEXT_COL = 'wp_context'

# Verified present for every season 2003-2025. epa/wp/success/fixed_drive are fully
# populated on pass/rush plays back to 2003; xpass is fully null before 2006, which
# makes PROE NaN there by the zero-denominator rule.
REQUIRED_PBP_COLUMNS = [
    'posteam', 'defteam', 'season', 'week', 'season_type', 'play_id', 'game_id',
    'pass', 'rush', 'down', 'yardline_100', 'third_down_converted', 'success',
    'epa', 'wp', 'xpass', 'fixed_drive', 'fixed_drive_result',
]

COMPETITIVE = 'competitive'
GARBAGE_LEADING = 'garbage_leading'
GARBAGE_TRAILING = 'garbage_trailing'
WP_CONTEXTS = [COMPETITIVE, GARBAGE_LEADING, GARBAGE_TRAILING]
GARBAGE_WP_THRESHOLD = 0.95
# 1 - 0.95 is 0.050000000000000044 in floating point, which would misclassify the
# inclusive 0.05 boundary; round keeps the bounds coupled and exact.
GARBAGE_WP_LOWER_THRESHOLD = round(1 - GARBAGE_WP_THRESHOLD, 10)
RED_ZONE_YARDLINE = 20

# wp is always the offense's win probability. From the defense's perspective the
# offense's garbage_leading is garbage_trailing and vice versa; competitive is symmetric.
DEFENSE_CONTEXT_SWAP = {
    COMPETITIVE: COMPETITIVE,
    GARBAGE_LEADING: GARBAGE_TRAILING,
    GARBAGE_TRAILING: GARBAGE_LEADING,
}

AGGREGATION_KEY_COLUMNS = ['posteam', 'season', 'week', 'season_type', 'defteam']

PLAY_COMPONENT_COLUMNS = [
    'play_count', 'epa_sum', 'success_sum',
    'dropback_count', 'dropback_epa_sum', 'dropback_success_sum',
    'rush_count', 'rush_epa_sum', 'rush_success_sum',
    'early_down_count', 'early_down_success_sum',
    'third_down_count', 'third_down_conversion_sum',
    'xpass_play_count', 'pass_minus_xpass_sum',
]
DRIVE_COMPONENT_COLUMNS = ['red_zone_drive_count', 'red_zone_td_drive_count']
COMPONENT_COLUMNS = PLAY_COMPONENT_COLUMNS + DRIVE_COMPONENT_COLUMNS

# (metric_name, numerator_component, denominator_component). Cumulative rates are always
# cumsum(numerator) / cumsum(denominator) so sparse weeks accumulate correctly.
RATE_METRICS = [
    ('epa_per_play', 'epa_sum', 'play_count'),
    ('pass_epa_per_dropback', 'dropback_epa_sum', 'dropback_count'),
    ('rush_epa_per_carry', 'rush_epa_sum', 'rush_count'),
    ('success_rate', 'success_sum', 'play_count'),
    ('pass_success_rate', 'dropback_success_sum', 'dropback_count'),
    ('rush_success_rate', 'rush_success_sum', 'rush_count'),
    ('early_down_success_rate', 'early_down_success_sum', 'early_down_count'),
    ('third_down_conversion_rate', 'third_down_conversion_sum', 'third_down_count'),
    ('red_zone_td_rate', 'red_zone_td_drive_count', 'red_zone_drive_count'),
    ('proe', 'pass_minus_xpass_sum', 'xpass_play_count'),
]


def _assign_wp_context(wp):
    """
    Bucket each play by the offense's pre-snap win probability: competitive
    (0.05 <= wp <= 0.95, boundaries inclusive), garbage_leading (wp > 0.95) or
    garbage_trailing (wp < 0.05). NaN wp defaults to competitive so the play is
    not silently dropped (wp is fully populated on pass/rush plays back to 2003).
    """
    context = pd.Series(COMPETITIVE, index=wp.index)
    context[wp > GARBAGE_WP_THRESHOLD] = GARBAGE_LEADING
    context[wp < GARBAGE_WP_LOWER_THRESHOLD] = GARBAGE_TRAILING
    return context


def _aggregate_play_components(pbp_df):
    """
    Aggregate the offensive play universe (pass or rush plays with an EPA value) to
    component sums per team-week-context. Components are numerator/denominator building
    blocks; rates are only ever computed from season-to-date component sums.
    """
    plays = pbp_df[((pbp_df['pass'] == 1) | (pbp_df['rush'] == 1)) & pbp_df['epa'].notna()].copy()
    plays[CONTEXT_COL] = _assign_wp_context(plays['wp'])

    plays['play_count'] = 1
    plays['epa_sum'] = plays['epa']
    plays['success_sum'] = plays['success']
    plays['dropback_count'] = plays['pass']
    plays['dropback_epa_sum'] = plays['epa'] * plays['pass']
    plays['dropback_success_sum'] = plays['success'] * plays['pass']
    plays['rush_count'] = plays['rush']
    plays['rush_epa_sum'] = plays['epa'] * plays['rush']
    plays['rush_success_sum'] = plays['success'] * plays['rush']
    plays['early_down_count'] = plays['down'].isin([1, 2]).astype(int)
    plays['early_down_success_sum'] = plays['early_down_count'] * plays['success']
    plays['third_down_count'] = (plays['down'] == 3).astype(int)
    plays['third_down_conversion_sum'] = plays['third_down_converted'].fillna(0)
    has_xpass = plays['xpass'].notna()
    plays['xpass_play_count'] = has_xpass.astype(int)
    plays['pass_minus_xpass_sum'] = (plays['pass'] - plays['xpass']).where(has_xpass, 0)

    return plays.groupby(
        AGGREGATION_KEY_COLUMNS + [CONTEXT_COL]
    )[PLAY_COMPONENT_COLUMNS].sum().reset_index()
