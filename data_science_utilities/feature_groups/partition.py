"""Partition the schedule_and_weekly model-ready feature columns into feature
families purely from their NAMES.

Family membership lives on two orthogonal axes, both encoded in the column name:

* **Content family** (what the metric measures) -- one of CONTENT_FAMILIES. This is
  a mutually-exclusive, exhaustive partition of every candidate feature: validated to
  classify all ~9,297 columns of xgb_features_list.csv with zero unknowns and zero
  collisions (see tests/.../test_feature_group_partition.py).
* **Form** (the temporal transform) -- raw | cumulative | EWMA-recency | rolling-recency,
  crossed with value | rank | rank_change. This cuts ACROSS every content family: a
  recency column is still a member of exactly one content family.

The README's "recency" and "aggregated values" families are FORM families, not content
families -- toggling them filters the form axis (see ``form_tags`` / ``is_rank_only_kept``),
which is what RANK_ONLY in rfe.ipynb does.

Naming conventions (set in src/data/collect_all.py and the collectors):

* Weekly box-score and ALL play-by-play features use perspective prefixes
  ``off_target_`` / ``off_opp_`` / ``def_target_`` / ``def_opp_`` (off = team offense,
  def_opp_/def_target_ = defense described via opponent-allowed; target/opp = which of
  the two duplicated game perspectives).
* Schedule-points features use the REVERSED ordering ``target_off_`` / ``opp_off_`` /
  ``target_def_`` / ``opp_def_`` and always contain ``score`` or ``points_allowed``.
* A play-by-play feature is told apart from a box-score feature solely by carrying a
  win-probability context token (``_competitive`` | ``_garbage_leading`` |
  ``_garbage_trailing``); box-score weekly features never have one.
"""

# --- Perspective prefixes -------------------------------------------------------
PERSPECTIVE_PREFIXES = ('off_target_', 'off_opp_', 'def_target_', 'def_opp_')
SCHEDULE_PREFIXES = ('target_off_', 'opp_off_', 'target_def_', 'opp_def_')

# --- Form-axis markers ----------------------------------------------------------
# A column whose name contains any of these is an aggregate over prior weeks; the
# RANK_ONLY filter (rfe.ipynb) drops such columns UNLESS they end in _rank/_rank_change.
RAW_VALUE_MARKERS = ('cumulative', 'ewma', 'rolling')
WP_CONTEXTS = ('_competitive', '_garbage_leading', '_garbage_trailing')
_FORM_SUFFIXES = (
    '_cumulative_average_change', '_cumulative_average',
    '_ewma_average', '_rolling_average',
)

# --- Content-family metric vocabularies (from the data-assembly collectors) ------
PHASE1_METRICS = frozenset({
    'epa_per_play', 'pass_epa_per_dropback', 'rush_epa_per_carry', 'success_rate',
    'pass_success_rate', 'rush_success_rate', 'early_down_success_rate',
    'third_down_conversion_rate', 'red_zone_td_rate', 'proe',
})
PHASE3_METRICS = frozenset({
    'sack_rate', 'qb_hit_rate', 'scramble_rate', 'stuff_rate', 'fumble_lost_rate',
    'shotgun_rate', 'no_huddle_rate', 'cpoe', 'yac_over_expected',
    'pen_committed_rate', 'pen_committed_yards_per_play', 'pen_drawn_rate',
    'pen_drawn_yards_per_play', 'seconds_per_play',
})
BOX_SCORE_METRICS = frozenset({
    'completions', 'attempts', 'passing_yards', 'passing_tds', 'interceptions',
    'sacks', 'sack_yards', 'sack_fumbles', 'sack_fumbles_lost', 'passing_air_yards',
    'passing_yards_after_catch', 'passing_first_downs', 'passing_epa', 'pacr',
    'carries', 'rushing_yards', 'rushing_tds', 'rushing_fumbles',
    'rushing_fumbles_lost', 'rushing_first_downs', 'rushing_epa', 'receiving_fumbles',
    'receiving_fumbles_lost', 'racr', 'wopr', 'special_teams_tds',
})
# These are the RAW schedule odds column names (src/data/schedule/collect.py ODDS_COLS).
# They are absent unless the dataset is regenerated with keep_odds=True, AND if odds are
# wired through the target/opp perspective rename they may surface under renamed forms --
# verify the post-rename names against the assembly before relying on this family.
MARKET_COLUMNS = frozenset({
    'away_moneyline', 'home_moneyline', 'spread_line', 'away_spread_odds',
    'home_spread_odds', 'total_line', 'under_odds', 'over_odds',
})
CONTEXT_REST_COLUMNS = frozenset({
    'week', 'div_game', 'indoor', 'is_home_target',
    'target_days_since_previous_game', 'opp_days_since_previous_game',
    'target_game_count', 'opp_game_count',
})

# --- Meta / target (never a model feature) --------------------------------------
# Most of these are true leakage (game_id, the scores, team names, h_win, the target);
# 'season' is excluded for a different reason -- era drift (the hold-out lies outside its
# training range) -- but it is grouped here because it must never be fed as a feature.
TARGET = 'target_win'
META_ID_COLUMNS = frozenset({
    'game_id', 'season', 'season_type', 'opp_team', 'opp_score',
    'target_team', 'target_score', 'h_win',
})

# Ordered so callers get a stable family iteration order.
CONTENT_FAMILIES = (
    'context_rest', 'market', 'schedule_points', 'box_score',
    'pbp_phase1', 'pbp_phase2_directional', 'pbp_phase3',
    'situational_playcall', 'snap_share',
)


def bare_metric(column):
    """Strip a perspective prefix, the rank/rank_change suffix, the form suffix and
    the win-probability context token to recover the underlying metric token."""
    stripped = column
    for prefix in PERSPECTIVE_PREFIXES:
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix):]
            break
    # rank_change must be tested before rank (it ends in '_rank' too).
    if stripped.endswith('_rank_change'):
        stripped = stripped[:-len('_rank_change')]
    elif stripped.endswith('_rank'):
        stripped = stripped[:-len('_rank')]
    for suffix in _FORM_SUFFIXES:
        if stripped.endswith(suffix):
            stripped = stripped[:-len(suffix)]
            break
    for context in WP_CONTEXTS:
        if stripped.endswith(context):
            stripped = stripped[:-len(context)]
            break
    return stripped


def content_family(column):
    """Return the single content family for a feature column.

    Raises ValueError if the column matches no family (so an unexpected naming
    change fails loudly rather than being silently dropped from the partition)."""
    if column in CONTEXT_REST_COLUMNS:
        return 'context_rest'
    if column in MARKET_COLUMNS:
        return 'market'
    if column.startswith(SCHEDULE_PREFIXES) and (
            'score' in column or 'points_allowed' in column):
        return 'schedule_points'
    if column.startswith(PERSPECTIVE_PREFIXES):
        metric = bare_metric(column)
        if metric in PHASE1_METRICS:
            return 'pbp_phase1'
        if (metric.startswith(('run_', 'pass_'))
                and (metric.endswith('_yards_per_attempt')
                     or metric.endswith('_explosive_rate'))):
            return 'pbp_phase2_directional'
        if metric in PHASE3_METRICS:
            return 'pbp_phase3'
        if metric.startswith(('down1_', 'down2_', 'down3_', 'goalToGo')):
            return 'situational_playcall'
        if metric == 'snap_share':
            return 'snap_share'
        if metric in BOX_SCORE_METRICS:
            return 'box_score'
    raise ValueError(f'unclassified feature column: {column!r}')


def form_tags(column):
    """Return the orthogonal form-axis tags for a column.

    :returns: dict with keys ``temporal`` (cumulative|ewma|rolling|raw),
        ``state`` (value|rank|rank_change), ``is_recency`` (the README 'recency'
        family: an EWMA or rolling transform) and ``is_aggregated_value`` (exactly
        the set RANK_ONLY drops: an aggregated transform that is NOT a rank)."""
    if 'ewma' in column:
        temporal = 'ewma'
    elif 'rolling' in column:
        temporal = 'rolling'
    elif 'cumulative' in column:
        temporal = 'cumulative'
    else:
        temporal = 'raw'
    if column.endswith('_rank_change'):
        state = 'rank_change'
    elif column.endswith('_rank'):
        state = 'rank'
    else:
        state = 'value'
    return {
        'temporal': temporal,
        'state': state,
        'is_recency': temporal in ('ewma', 'rolling'),
        'is_aggregated_value': temporal != 'raw' and state == 'value',
    }


def is_rank_only_kept(column):
    """Whether ``column`` survives the RANK_ONLY filter from rfe.ipynb: drop any
    name containing an aggregate marker unless it ends in _rank/_rank_change."""
    if not any(marker in column for marker in RAW_VALUE_MARKERS):
        return True
    return column.endswith('_rank') or column.endswith('_rank_change')


def partition_features(features, rank_only=False):
    """Group feature columns by content family.

    :param features: iterable of column names (meta/id columns and the target are
        skipped automatically).
    :param rank_only: when True, apply the RANK_ONLY filter first, dropping the
        aggregated continuous-value columns exactly as rfe.ipynb does.
    :returns: dict family -> list of columns, preserving input order within a family.
        Empty families are omitted; family keys follow CONTENT_FAMILIES order.
    """
    groups = {}
    for column in features:
        if column == TARGET or column in META_ID_COLUMNS:
            continue
        if rank_only and not is_rank_only_kept(column):
            continue
        groups.setdefault(content_family(column), []).append(column)
    return {family: groups[family] for family in CONTENT_FAMILIES if family in groups}
