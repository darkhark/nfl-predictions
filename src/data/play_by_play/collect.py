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


def _aggregate_red_zone_components(pbp_df):
    """
    Count red-zone trips per team-week-context at the drive level: a drive counts as a
    red-zone trip when any of its scrimmage plays starts at or inside the opponent's 20.
    Only scrimmage plays (pass or rush) are considered: PAT and kickoff rows share the
    drive's fixed_drive number at misleading yardlines (a PAT snapped at the 15 would
    otherwise turn every long touchdown into a fake red-zone trip). A drive's context
    comes from the win probability on its first scrimmage play (drives can drift across
    contexts mid-drive; the first snap reflects the situation the drive started in).
    Drives that enter the red zone only via a kick or kneel (e.g. driving to the 22 and
    kicking a field goal from the 14) are intentionally excluded on both sides of the
    red_zone_td_rate ratio. fixed_drive numbers drives across the whole game, so
    (game_id, fixed_drive) is unique.
    """
    scrimmage = pbp_df[(pbp_df['pass'] == 1) | (pbp_df['rush'] == 1)]
    drive_plays = scrimmage[scrimmage['fixed_drive'].notna() & scrimmage['posteam'].notna()].sort_values('play_id')
    drives = drive_plays.groupby(['game_id', 'fixed_drive'] + AGGREGATION_KEY_COLUMNS).agg(
        min_yardline_100=('yardline_100', 'min'),
        first_play_wp=('wp', 'first'),
        drive_result=('fixed_drive_result', 'first'),
    ).reset_index()

    red_zone_drives = drives[drives['min_yardline_100'] <= RED_ZONE_YARDLINE].copy()
    red_zone_drives[CONTEXT_COL] = _assign_wp_context(red_zone_drives['first_play_wp'])
    red_zone_drives['red_zone_drive_count'] = 1
    red_zone_drives['red_zone_td_drive_count'] = (red_zone_drives['drive_result'] == 'Touchdown').astype(int)

    return red_zone_drives.groupby(
        AGGREGATION_KEY_COLUMNS + [CONTEXT_COL]
    )[DRIVE_COMPONENT_COLUMNS].sum().reset_index()


def _validate_required_columns(pbp_df):
    missing = sorted(set(REQUIRED_PBP_COLUMNS) - set(pbp_df.columns))
    if missing:
        raise ValueError(f'Play-by-play data is missing required columns: {missing}')


def _pivot_context_components(components):
    """
    Pivot the long (team-week-context) component frame wide so each component becomes
    three columns, one per context (e.g. play_count_competitive). A context absent for a
    team-week means zero plays happened in it, so component sums fill with 0; the
    zero-denominator rule later turns the corresponding rates into NaN.
    """
    pivoted = components.pivot_table(
        index=[TEAM_COL, SEASON_COL, WEEK_COL, SEASON_TYPE_COL, OPPONENT_TEAM_COL],
        columns=CONTEXT_COL,
        values=COMPONENT_COLUMNS,
        aggfunc='sum',
        fill_value=0,
    )
    pivoted.columns = [f'{component}_{context}' for component, context in pivoted.columns]
    # Reindex to a canonical column order: pivot_table's output order depends on which
    # contexts appear in the data, and the per-season parquet caches must share one
    # schema. reindex also backfills any component/context column absent from the data
    # (zero plays in that context).
    ordered_columns = [f'{component}_{context}'
                       for component in COMPONENT_COLUMNS for context in WP_CONTEXTS]
    pivoted = pivoted.reindex(columns=ordered_columns, fill_value=0)
    return pivoted.reset_index()


def _aggregate_season(pbp_df):
    """
    Reduce one season of raw play-by-play to one row per team-game with component sums
    per wp context. This is the frame that gets cached per season.
    """
    _validate_required_columns(pbp_df)
    pbp_df = pbp_df[pbp_df['posteam'].notna() & pbp_df['defteam'].notna()].copy()
    pbp_df['posteam'] = pbp_df['posteam'].replace(TEAM_ABBR_MAPPINGS)
    pbp_df['defteam'] = pbp_df['defteam'].replace(TEAM_ABBR_MAPPINGS)

    play_components = _aggregate_play_components(pbp_df)
    drive_components = _aggregate_red_zone_components(pbp_df)
    components = play_components.merge(
        drive_components, on=AGGREGATION_KEY_COLUMNS + [CONTEXT_COL], how='outer'
    )
    # Fill only the component columns: a side missing from the outer merge means zero
    # plays/drives, and restricting the fill keeps pandas from object-downcasting keys.
    components[COMPONENT_COLUMNS] = components[COMPONENT_COLUMNS].fillna(0)
    components = components.rename(columns={'posteam': TEAM_COL, 'defteam': OPPONENT_TEAM_COL})
    return _pivot_context_components(components)


def get_play_by_play_data(years, refresh=False):
    """
    Return team-week component sums for the specified season(s), one row per team-game.

    Raw play-by-play is ~50k rows x 396 columns per season, so each season is downloaded
    once (selecting only REQUIRED_PBP_COLUMNS), aggregated, and cached to
    CACHE_DIR/{year}.parquet. Subsequent calls read the small aggregated frame. Pass
    refresh=True to re-download (needed while a season is in progress).

    :param years: list of years to collect data for or a single year
    :param refresh: re-download and re-aggregate even when a cache file exists
    :return: concatenated component frame across the requested seasons
    """
    if isinstance(years, int):
        years = [years]
    elif not isinstance(years, list):
        raise TypeError('years must be a list or an integer')

    season_frames = []
    for year in years:
        cache_path = os.path.join(CACHE_DIR, f'{year}.parquet')
        if os.path.exists(cache_path) and not refresh:
            season_frames.append(pd.read_parquet(cache_path))
            continue
        raw_pbp = nfl.import_pbp_data(years=[year], columns=REQUIRED_PBP_COLUMNS, downcast=False)
        season_components = _aggregate_season(raw_pbp)
        os.makedirs(CACHE_DIR, exist_ok=True)
        season_components.to_parquet(cache_path, index=False)
        season_frames.append(season_components)
    return pd.concat(season_frames, ignore_index=True)
