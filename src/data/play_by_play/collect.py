import os

import nfl_data_py as nfl
import pandas as pd

from src.data import transformations
from src.data.transformations import (
    TEAM_ABBR_MAPPINGS, TEAM_COL, OPPONENT_TEAM_COL, SEASON_COL, WEEK_COL,
    TEAM_GAME_COUNT_COL, OPP_GAME_COUNT_COL,
)

# Anchored to the repo root via this file's location so notebook callers (whose CWD is
# the notebook's directory) read and write the same cache as scripts run from the root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CACHE_DIR = os.path.join(_REPO_ROOT, 'data', 'play_by_play', 'aggregated')

SEASON_TYPE_COL = 'season_type'
CONTEXT_COL = 'wp_context'

# Verified present for every season 2003-2025. epa/wp/success/fixed_drive are fully
# populated on pass/rush plays back to 2003; xpass is fully null before 2006, which
# makes PROE NaN there by the zero-denominator rule. pass_location/pass_length are
# fully null before 2006 too (directional pass features go NaN there the same way);
# run_location is ~95% populated on rushes in all eras, run_gap ~70% (middle runs
# have no gap by definition). Phase 3: sack/qb_hit/qb_scramble/shotgun/no_huddle/
# fumble/fumble_lost are ~99.9% populated on universe rows in all eras (the rare NaN
# are EPA-valued plays with nullified yardage; groupby-sum skips them so counts are
# unaffected). CAUTION: qb_hit is zero-FILLED (not NaN) before 2006 when hit charting
# began, so qb_hit_rate is a literal 0.0 for 2003-2005, not NaN — within-season ranks
# are unaffected (everyone ties), but the raw rate is not comparable across that era
# boundary. game_seconds_remaining is ~99.9% non-null over all rows; cpoe and
# yards_after_catch/xyac_mean_yardage are null
# before 2006 (CPOE and YAC-over-expected go NaN there); penalty is ~97% non-null
# (compare with == 1, which is False for NaN), penalty_team is always set on penalty
# rows, penalty_yards may be NaN.
REQUIRED_PBP_COLUMNS = [
    'posteam', 'defteam', 'season', 'week', 'season_type', 'play_id', 'game_id',
    'pass', 'rush', 'down', 'ydstogo', 'goal_to_go',
    'yardline_100', 'third_down_converted', 'success',
    'epa', 'wp', 'xpass', 'fixed_drive', 'fixed_drive_result',
    'yards_gained', 'run_location', 'run_gap', 'pass_location', 'pass_length',
    'sack', 'qb_hit', 'qb_scramble', 'shotgun', 'no_huddle',
    'fumble', 'fumble_lost', 'cpoe', 'complete_pass',
    'yards_after_catch', 'xyac_mean_yardage',
    'penalty', 'penalty_team', 'penalty_yards', 'game_seconds_remaining',
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

# Phase 2: directional buckets. Runs split by run_location x run_gap (middle has no
# gap by definition); passes by pass_length x pass_location. Plays with unlabeled
# direction contribute to no bucket (they remain in the aggregate Phase 1 metrics);
# per-bucket denominators are bucket attempt counts, so rates cover labeled plays only.
RUN_BUCKETS = [
    'run_left_end', 'run_left_tackle', 'run_left_guard', 'run_middle',
    'run_right_guard', 'run_right_tackle', 'run_right_end',
]
PASS_BUCKETS = [
    'pass_short_left', 'pass_short_middle', 'pass_short_right',
    'pass_deep_left', 'pass_deep_middle', 'pass_deep_right',
]
DIRECTIONAL_BUCKETS = RUN_BUCKETS + PASS_BUCKETS

EXPLOSIVE_RUSH_YARDS = 10
EXPLOSIVE_PASS_YARDS = 20

DIRECTIONAL_COMPONENT_COLUMNS = [
    f'{bucket}_{component}'
    for bucket in DIRECTIONAL_BUCKETS
    for component in ('attempt_count', 'yards_sum', 'explosive_count')
]

DIRECTIONAL_RATE_METRICS = (
    [(f'{bucket}_yards_per_attempt', f'{bucket}_yards_sum', f'{bucket}_attempt_count')
     for bucket in DIRECTIONAL_BUCKETS]
    + [(f'{bucket}_explosive_rate', f'{bucket}_explosive_count', f'{bucket}_attempt_count')
       for bucket in DIRECTIONAL_BUCKETS]
)

# Situational play-call buckets: down x distance for downs 2/3 (1st down is ~always
# 1st-and-10, so it is a single bucket; goal_to_go overrides distance; 4th down excluded).
SITUATIONAL_SHORT_MAX = 2          # short  = ydstogo <= 2
SITUATIONAL_MEDIUM_MAX = 6         # medium = 3..6 ; long = >= 7
SITUATIONAL_BUCKETS = [
    'down1',
    'down2_short', 'down2_med', 'down2_long',
    'down3_short', 'down3_med', 'down3_long',
    'goalToGo',
]
_SITUATIONAL_DOWN3 = {'down3_short', 'down3_med', 'down3_long'}

SITUATIONAL_COMPONENT_COLUMNS = []
for _b in SITUATIONAL_BUCKETS:
    SITUATIONAL_COMPONENT_COLUMNS += [f'{_b}_play_count', f'{_b}_pass_count', f'{_b}_run_count']
    if _b in _SITUATIONAL_DOWN3:
        SITUATIONAL_COMPONENT_COLUMNS += [f'{_b}_pass_conversion_sum', f'{_b}_run_conversion_sum']
    else:
        SITUATIONAL_COMPONENT_COLUMNS += [f'{_b}_pass_success_sum', f'{_b}_run_success_sum']

SITUATIONAL_RATE_METRICS = []
for _b in SITUATIONAL_BUCKETS:
    SITUATIONAL_RATE_METRICS.append((f'{_b}_pass_rate', f'{_b}_pass_count', f'{_b}_play_count'))
    if _b in _SITUATIONAL_DOWN3:
        SITUATIONAL_RATE_METRICS.append((f'{_b}_conversion_rate_pass', f'{_b}_pass_conversion_sum', f'{_b}_pass_count'))
        SITUATIONAL_RATE_METRICS.append((f'{_b}_conversion_rate_run', f'{_b}_run_conversion_sum', f'{_b}_run_count'))
    else:
        SITUATIONAL_RATE_METRICS.append((f'{_b}_success_rate_pass', f'{_b}_pass_success_sum', f'{_b}_pass_count'))
        SITUATIONAL_RATE_METRICS.append((f'{_b}_success_rate_run', f'{_b}_run_success_sum', f'{_b}_run_count'))

# Phase 3: trenches, turnover luck, and tendency components. Sacks/hits/scrambles are
# per dropback, stuffs per carry, fumble-recovery luck per fumble; cpoe and
# YAC-over-expected average only over plays where nflfastR charts them (2006+ —
# their zero denominators make the rates NaN before that). qb_hit_rate is the one
# era asymmetry: zero-filled pre-2006, so it reads 0.0 rather than NaN there.
PHASE3_PLAY_COMPONENT_COLUMNS = [
    'sack_count', 'qb_hit_count', 'stuff_count',
    'fumble_sum', 'fumble_lost_sum',
    'scramble_count', 'shotgun_count', 'no_huddle_count',
    'cpoe_sum', 'cpoe_play_count',
    'yac_minus_xyac_sum', 'xyac_play_count',
]

# Penalties live partly on no-play rows outside the pass/rush universe, so they get
# their own aggregation pass. Committed = by the offense (penalty_team == posteam);
# drawn = by the defense against it (penalty_team == defteam). Rates are per
# scrimmage play (play_count denominator).
PENALTY_COMPONENT_COLUMNS = [
    'pen_committed_count', 'pen_committed_yards_sum',
    'pen_drawn_count', 'pen_drawn_yards_sum',
]

PHASE3_RATE_METRICS = [
    ('sack_rate', 'sack_count', 'dropback_count'),
    ('qb_hit_rate', 'qb_hit_count', 'dropback_count'),
    ('scramble_rate', 'scramble_count', 'dropback_count'),
    ('stuff_rate', 'stuff_count', 'rush_count'),
    ('fumble_lost_rate', 'fumble_lost_sum', 'fumble_sum'),
    ('shotgun_rate', 'shotgun_count', 'play_count'),
    ('no_huddle_rate', 'no_huddle_count', 'play_count'),
    ('cpoe', 'cpoe_sum', 'cpoe_play_count'),
    ('yac_over_expected', 'yac_minus_xyac_sum', 'xyac_play_count'),
]
PENALTY_RATE_METRICS = [
    ('pen_committed_rate', 'pen_committed_count', 'play_count'),
    ('pen_committed_yards_per_play', 'pen_committed_yards_sum', 'play_count'),
    ('pen_drawn_rate', 'pen_drawn_count', 'play_count'),
    ('pen_drawn_yards_per_play', 'pen_drawn_yards_sum', 'play_count'),
]
# pace_seconds_sum / pace_play_count are drive-level components and live in
# DRIVE_COMPONENT_COLUMNS (wired by the drive-aggregation task), not in a PACE_* list.
PACE_RATE_METRICS = [
    ('seconds_per_play', 'pace_seconds_sum', 'pace_play_count'),
]

PLAY_COMPONENT_COLUMNS = [
    'play_count', 'epa_sum', 'success_sum',
    'dropback_count', 'dropback_epa_sum', 'dropback_success_sum',
    'rush_count', 'rush_epa_sum', 'rush_success_sum',
    'early_down_count', 'early_down_success_sum',
    'third_down_count', 'third_down_conversion_sum',
    'xpass_play_count', 'pass_minus_xpass_sum',
] + DIRECTIONAL_COMPONENT_COLUMNS + PHASE3_PLAY_COMPONENT_COLUMNS
DRIVE_COMPONENT_COLUMNS = [
    'red_zone_drive_count', 'red_zone_td_drive_count',
    'pace_seconds_sum', 'pace_play_count',
]
COMPONENT_COLUMNS = PLAY_COMPONENT_COLUMNS + DRIVE_COMPONENT_COLUMNS + PENALTY_COMPONENT_COLUMNS

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
] + DIRECTIONAL_RATE_METRICS + PHASE3_RATE_METRICS + PENALTY_RATE_METRICS + PACE_RATE_METRICS


def _assign_run_bucket(plays):
    """
    Label each rush with its directional bucket. Middle runs have no gap by definition
    and bucket as run_middle; left/right runs need a gap label. Unlabeled runs (NaN
    run_location, ~3% of rushes; sided runs with NaN gap are ~0% in real data) get no
    bucket — they still count in the aggregate Phase 1 metrics, and per-bucket
    denominators only cover labeled runs. A foreign gap label (e.g. a future nflfastR
    value outside end/tackle/guard) would form a name outside DIRECTIONAL_BUCKETS and
    also land in no bucket.
    """
    bucket = pd.Series(None, index=plays.index, dtype='object')
    is_rush = plays['rush'] == 1
    bucket[is_rush & (plays['run_location'] == 'middle')] = 'run_middle'
    sided = is_rush & plays['run_location'].isin(['left', 'right']) & plays['run_gap'].notna()
    bucket[sided] = 'run_' + plays.loc[sided, 'run_location'] + '_' + plays.loc[sided, 'run_gap']
    return bucket


def _assign_pass_bucket(plays):
    """
    Label each located pass with its depth x direction bucket. Sacks, scrambles, and
    throwaways carry pass == 1 with no location/length and get no bucket; before 2006
    nflfastR has no pass charting at all, so every pass is unlabeled there and the
    directional pass features are NaN for those seasons (like PROE/CPOE).
    """
    bucket = pd.Series(None, index=plays.index, dtype='object')
    located = (
        (plays['pass'] == 1)
        & plays['pass_location'].notna()
        & plays['pass_length'].notna()
    )
    bucket[located] = (
        'pass_' + plays.loc[located, 'pass_length'] + '_' + plays.loc[located, 'pass_location']
    )
    return bucket


def _assign_situational_bucket(plays):
    """Label each play with its down x distance situational bucket (or None). goal_to_go
    overrides down/distance; 1st down is a single bucket; 4th down is unlabeled. Distance
    bins: short <= 2, medium 3..6, long >= 7. Plays outside any bucket still count in the
    aggregate Phase 1/2/3 metrics."""
    bucket = pd.Series(None, index=plays.index, dtype='object')
    goal = plays['goal_to_go'] == 1
    bucket[goal] = 'goalToGo'
    rest = ~goal
    down = plays['down']
    ytg = plays['ydstogo']
    bucket[rest & (down == 1)] = 'down1'
    for d, prefix in ((2, 'down2'), (3, 'down3')):
        sel = rest & (down == d)
        bucket[sel & (ytg <= SITUATIONAL_SHORT_MAX)] = f'{prefix}_short'
        bucket[sel & (ytg > SITUATIONAL_SHORT_MAX) & (ytg <= SITUATIONAL_MEDIUM_MAX)] = f'{prefix}_med'
        bucket[sel & (ytg > SITUATIONAL_MEDIUM_MAX)] = f'{prefix}_long'
    return bucket


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

    directional_bucket = _assign_run_bucket(plays)
    directional_bucket = directional_bucket.where(directional_bucket.notna(),
                                                  _assign_pass_bucket(plays))
    yards = plays['yards_gained'].fillna(0)
    is_explosive = (
        ((plays['rush'] == 1) & (yards >= EXPLOSIVE_RUSH_YARDS))
        | ((plays['pass'] == 1) & (yards >= EXPLOSIVE_PASS_YARDS))
    )
    for bucket in DIRECTIONAL_BUCKETS:
        in_bucket = (directional_bucket == bucket).astype(int)
        plays[f'{bucket}_attempt_count'] = in_bucket
        plays[f'{bucket}_yards_sum'] = yards * in_bucket
        plays[f'{bucket}_explosive_count'] = is_explosive.astype(int) * in_bucket

    plays['sack_count'] = plays['sack']
    plays['qb_hit_count'] = plays['qb_hit']
    # NaN yards_gained must not count as a stuff: the raw column comparison is False
    # for NaN, unlike the zero-filled `yards` used for explosives above.
    plays['stuff_count'] = ((plays['rush'] == 1) & (plays['yards_gained'] <= 0)).astype(int)
    plays['fumble_sum'] = plays['fumble']
    plays['fumble_lost_sum'] = plays['fumble_lost']
    plays['scramble_count'] = plays['qb_scramble']
    plays['shotgun_count'] = plays['shotgun']
    plays['no_huddle_count'] = plays['no_huddle']
    has_cpoe = plays['cpoe'].notna()
    plays['cpoe_play_count'] = has_cpoe.astype(int)
    plays['cpoe_sum'] = plays['cpoe'].where(has_cpoe, 0)
    has_xyac = (
        (plays['complete_pass'] == 1)
        & plays['xyac_mean_yardage'].notna()
        & plays['yards_after_catch'].notna()
    )
    plays['xyac_play_count'] = has_xyac.astype(int)
    plays['yac_minus_xyac_sum'] = (
        plays['yards_after_catch'] - plays['xyac_mean_yardage']
    ).where(has_xyac, 0)

    return plays.groupby(
        AGGREGATION_KEY_COLUMNS + [CONTEXT_COL]
    )[PLAY_COMPONENT_COLUMNS].sum().reset_index()


def _aggregate_drive_components(pbp_df):
    """
    Drive-level components per team-week-context, over scrimmage plays only (pass or
    rush): PAT and kickoff rows share the drive's fixed_drive number at misleading
    yardlines (a PAT snapped at the 15 would otherwise turn every long touchdown into a
    fake red-zone trip). A drive's context comes from the win probability on its first
    scrimmage play.

    Red zone: a drive counts as a trip when any of its scrimmage plays starts at or
    inside the opponent's 20; drives that enter only via a kick or kneel are
    intentionally excluded on both sides of the red_zone_td_rate ratio.

    Pace: game-clock seconds elapsed between the drive's first and last scrimmage snap,
    over its scrimmage snap count. This undercounts by the final play's duration
    (n snaps bound n-1 intervals — at the league's ~6 snaps/drive that reads ~16% below
    a true per-snap clock) and covers scrimmage snaps only. The bias is consistent
    across teams, so the within-week ranks the model consumes are unaffected.
    clip(lower=0) guards overtime clock quirks (verified never firing on real data).

    fixed_drive numbers drives across the whole game, so (game_id, fixed_drive) is
    unique.
    """
    scrimmage = pbp_df[(pbp_df['pass'] == 1) | (pbp_df['rush'] == 1)]
    drive_plays = scrimmage[scrimmage['fixed_drive'].notna() & scrimmage['posteam'].notna()].sort_values('play_id')
    drives = drive_plays.groupby(['game_id', 'fixed_drive'] + AGGREGATION_KEY_COLUMNS).agg(
        min_yardline_100=('yardline_100', 'min'),
        first_play_wp=('wp', 'first'),
        drive_result=('fixed_drive_result', 'first'),
        first_gsr=('game_seconds_remaining', 'first'),
        last_gsr=('game_seconds_remaining', 'last'),
        scrimmage_snaps=('play_id', 'count'),
    ).reset_index()

    drives[CONTEXT_COL] = _assign_wp_context(drives['first_play_wp'])
    reached_red_zone = drives['min_yardline_100'] <= RED_ZONE_YARDLINE
    drives['red_zone_drive_count'] = reached_red_zone.astype(int)
    drives['red_zone_td_drive_count'] = (
        reached_red_zone & (drives['drive_result'] == 'Touchdown')
    ).astype(int)
    drives['pace_seconds_sum'] = (drives['first_gsr'] - drives['last_gsr']).clip(lower=0)
    drives['pace_play_count'] = drives['scrimmage_snaps']

    return drives.groupby(
        AGGREGATION_KEY_COLUMNS + [CONTEXT_COL]
    )[DRIVE_COMPONENT_COLUMNS].sum().reset_index()


def _aggregate_penalty_components(pbp_df):
    """
    Count penalties per team-week-context over ALL rows with teams attached: accepted
    penalties frequently live on no-play rows outside the pass/rush universe, so this is
    a separate aggregation pass (like drives). Committed = flagged on the offense
    (penalty_team == posteam); drawn = flagged on the defense (penalty_team == defteam).
    The matching rates use scrimmage play_count as the denominator, so they read as
    "penalties per offensive snap". NaN penalty values compare False and are ignored.
    Note: nflfastR's penalty flag covers ACCEPTED penalties only (declined/offsetting
    are unflagged), so these rates read low vs league totals that include declined.
    """
    penalties = pbp_df[
        (pbp_df['penalty'] == 1)
        & pbp_df['posteam'].notna()
        & pbp_df['defteam'].notna()
    ].copy()
    penalties[CONTEXT_COL] = _assign_wp_context(penalties['wp'])

    committed = penalties['penalty_team'] == penalties['posteam']
    drawn = penalties['penalty_team'] == penalties['defteam']
    yards = penalties['penalty_yards'].fillna(0)
    penalties['pen_committed_count'] = committed.astype(int)
    penalties['pen_committed_yards_sum'] = yards * committed
    penalties['pen_drawn_count'] = drawn.astype(int)
    penalties['pen_drawn_yards_sum'] = yards * drawn

    return penalties.groupby(
        AGGREGATION_KEY_COLUMNS + [CONTEXT_COL]
    )[PENALTY_COMPONENT_COLUMNS].sum().reset_index()


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
    pbp_df['penalty_team'] = pbp_df['penalty_team'].replace(TEAM_ABBR_MAPPINGS)

    play_components = _aggregate_play_components(pbp_df)
    drive_components = _aggregate_drive_components(pbp_df)
    penalty_components = _aggregate_penalty_components(pbp_df)
    components = play_components.merge(
        drive_components, on=AGGREGATION_KEY_COLUMNS + [CONTEXT_COL], how='outer'
    ).merge(
        penalty_components, on=AGGREGATION_KEY_COLUMNS + [CONTEXT_COL], how='outer'
    )
    # Fill only the component columns: a side missing from the outer merge means zero
    # plays/drives/penalties, and restricting the fill keeps pandas from
    # object-downcasting keys.
    components[COMPONENT_COLUMNS] = components[COMPONENT_COLUMNS].fillna(0)
    components = components.rename(columns={'posteam': TEAM_COL, 'defteam': OPPONENT_TEAM_COL})
    return _pivot_context_components(components)


def _add_game_count_columns(df):
    """
    Add per-team and per-opponent game counters within each season. Week number is not a
    reliable divisor because of byes, so cumulative math orders and groups by these
    counters, mirroring the weekly module.
    """
    df = df.copy()
    df.sort_values(by=[TEAM_COL, SEASON_COL, WEEK_COL], inplace=True)
    df[TEAM_GAME_COUNT_COL] = df.groupby([TEAM_COL, SEASON_COL]).cumcount() + 1
    df.sort_values(by=[OPPONENT_TEAM_COL, SEASON_COL, WEEK_COL], inplace=True)
    df[OPP_GAME_COUNT_COL] = df.groupby([OPPONENT_TEAM_COL, SEASON_COL]).cumcount() + 1
    df.sort_values(by=[TEAM_COL, SEASON_COL, WEEK_COL], inplace=True)
    return df


def _add_cumulative_rate_columns(df):
    """
    Add season-to-date rate columns for every metric/context, offense and defense.

    Every rate is cumsum(numerator) / cumsum(denominator) within the season - never an
    average of weekly rates - so thin contexts accumulate correctly. A zero cumulative
    denominator yields NaN: no snaps means no rate, not a rate of zero.

    Offense accumulates within (team, season). Defense accumulates the same base
    components within (opp_team, season) - exactly the weekly module's def_opp pattern,
    so def_opp_* on a row describes the opponent's defense season-to-date. Defense
    context labels are swapped via DEFENSE_CONTEXT_SWAP because wp belongs to the
    offense: plays where the offense was garbage_leading are the defense's
    garbage_trailing snaps.

    Requires a unique index (reset_index before calling) and game-count columns.
    """
    if not df.index.is_unique:
        raise ValueError(
            'cumulative rate computation requires a unique index; call reset_index first '
            '(a duplicated index silently scrambles def_opp values during realignment)'
        )
    component_cols = [f'{component}_{context}'
                      for component in COMPONENT_COLUMNS for context in WP_CONTEXTS]

    df = df.sort_values(by=[TEAM_COL, SEASON_COL, TEAM_GAME_COUNT_COL])
    off_cumulative = df.groupby([TEAM_COL, SEASON_COL])[component_cols].cumsum()

    opp_ordered = df.sort_values(by=[OPPONENT_TEAM_COL, SEASON_COL, OPP_GAME_COUNT_COL])
    def_cumulative = opp_ordered.groupby(
        [OPPONENT_TEAM_COL, SEASON_COL]
    )[component_cols].cumsum()

    rate_columns = {}
    for metric, numerator, denominator in RATE_METRICS:
        for context in WP_CONTEXTS:
            offense_numerator = off_cumulative[f'{numerator}_{context}']
            offense_denominator = off_cumulative[f'{denominator}_{context}']
            rate_columns[f'off_{metric}_{context}_cumulative_average'] = (
                offense_numerator / offense_denominator.where(offense_denominator != 0)
            )

            defense_context = DEFENSE_CONTEXT_SWAP[context]
            defense_numerator = def_cumulative[f'{numerator}_{context}']
            defense_denominator = def_cumulative[f'{denominator}_{context}']
            rate_columns[f'def_opp_{metric}_{defense_context}_cumulative_average'] = (
                defense_numerator / defense_denominator.where(defense_denominator != 0)
            )

    return pd.concat([df, pd.DataFrame(rate_columns)], axis=1)


def get_play_by_play_features(years, refresh=False):
    """
    Build the model-facing play-by-play feature frame: one row per team-game keyed by
    (team, season, week), with cumulative rate, rank, and rank-change columns for every
    Phase 1 metric, wp context, and side of the ball. Column naming follows the weekly
    module's off_* / def_opp_* convention so collect_all's home/away renames, target
    duplication, and one-week leakage shift apply to these features unchanged.

    :param years: list of years to collect data for or a single year
    :param refresh: re-download and re-aggregate even when a season cache file exists
    :return: feature frame ready to merge onto the weekly frame
    """
    components = get_play_by_play_data(years, refresh=refresh).reset_index(drop=True)
    components = _add_game_count_columns(components).reset_index(drop=True)
    df = _add_cumulative_rate_columns(components)

    off_cols = [col for col in df.columns
                if col.startswith('off_') and col.endswith('_cumulative_average')]
    def_cols = [col for col in df.columns
                if col.startswith('def_opp_') and col.endswith('_cumulative_average')]
    df = transformations.add_rank_and_rank_change_columns(df, off_cols, def_cols)

    feature_cols = [col for col in df.columns
                    if col.startswith('off_') or col.startswith('def_opp_')]
    return df[[TEAM_COL, SEASON_COL, WEEK_COL] + feature_cols].reset_index(drop=True)


def get_play_by_play_data(years, refresh=False):
    """
    Return team-week component sums for the specified season(s), one row per team-game.

    Raw play-by-play is ~50k rows x 396 columns per season, so each season is downloaded
    once (selecting only REQUIRED_PBP_COLUMNS), aggregated, and cached to
    CACHE_DIR/{year}.parquet. Subsequent calls read the small aggregated frame. Pass
    refresh=True to re-download — needed while a season is in progress, and after any
    code change that alters the cached schema (cached files are read back verbatim).

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
