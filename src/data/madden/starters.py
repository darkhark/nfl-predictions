"""Leakage-safe weekly starters: depth-chart rank-1 minus Out/Doubtful players,
promoting the next available player at each position. All inputs are pre-kickoff."""
import logging
import nfl_data_py as nfl
import pandas as pd
from src.data.transformations import TEAM_ABBR_MAPPINGS
from src.data.madden import depth_2025

OUT_STATUSES = ('Out', 'Doubtful')
logger = logging.getLogger(__name__)


def available_starters(depth, injuries):
    """One team-week. Return the highest-available player per position.

    :param depth: depth rows with [position, depth_team, gsis_id].
    :param injuries: rows with [gsis_id, report_status].
    :returns: [gsis_id, position, depth_chart_rank] one row per position.
    """
    unavailable = set(
        injuries.loc[injuries['report_status'].isin(OUT_STATUSES), 'gsis_id'])
    d = depth.copy()
    d['rank'] = pd.to_numeric(d['depth_team'], errors='coerce')
    d = d[~d['gsis_id'].isin(unavailable)]
    d = d.sort_values(['position', 'rank'])
    chosen = d.groupby('position', as_index=False).first()
    return chosen[['gsis_id', 'position', 'rank']].rename(
        columns={'rank': 'depth_chart_rank'})


_OLD_REQUIRED = {'game_type', 'club_code', 'depth_team', 'gsis_id', 'position'}
_DEPTH_CONTRACT = ['season', 'week', 'club_code', 'game_type',
                   'position', 'depth_team', 'gsis_id']


def _normalized_old_depth(years):
    """Pre-2025 nflverse depth schema -> the shared depth contract (REG only)."""
    try:
        depth = nfl.import_depth_charts(years)
    except Exception as exc:
        logger.warning('import_depth_charts(%s) unavailable (%s)', years, exc)
        return None
    if depth is None or depth.empty:
        return None
    missing = _OLD_REQUIRED - set(depth.columns)
    if missing:
        logger.warning('import_depth_charts(%s) unexpected schema (missing %s)',
                       years, missing)
        return None
    depth = depth[depth['game_type'] == 'REG'].copy()
    depth['club_code'] = depth['club_code'].replace(TEAM_ABBR_MAPPINGS)
    return depth[_DEPTH_CONTRACT]


def _normalized_new_depth(year):
    """2025+ nflverse depth schema -> the shared depth contract via depth_2025."""
    try:
        depth_raw = nfl.import_depth_charts([year])
        schedule = nfl.import_schedules([year])
    except Exception as exc:
        logger.warning('2025+ depth/schedule import for %s failed (%s)', year, exc)
        return None
    norm = depth_2025.normalize_2025_depth(depth_raw, schedule, year)
    return norm if not norm.empty else None


def _load_injuries(years):
    try:
        injuries = nfl.import_injuries(years)
        injuries['team'] = injuries['team'].replace(TEAM_ABBR_MAPPINGS)
        return injuries
    except Exception as exc:
        logger.warning('import_injuries(%s) unavailable (%s) — all players available',
                       years, exc)
        return pd.DataFrame(columns=['gsis_id', 'report_status', 'team', 'season', 'week'])


def _starters_from_depth(depth, injuries):
    frames = []
    for (season, week, club), grp in depth.groupby(['season', 'week', 'club_code']):
        inj = injuries[(injuries['season'] == season) & (injuries['week'] == week)
                       & (injuries['team'] == club)]
        chosen = available_starters(grp, inj)
        chosen['season'], chosen['week'], chosen['team'] = season, week, club
        frames.append(chosen)
    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True)
    return out[['season', 'week', 'team', 'gsis_id', 'position']]


def get_weekly_starters(years):
    years = [years] if isinstance(years, int) else list(years)
    _empty_starters = pd.DataFrame(
        columns=['season', 'week', 'team', 'gsis_id', 'position'])
    parts = []
    old_years = [y for y in years if y < 2025]
    if old_years:
        od = _normalized_old_depth(old_years)
        if od is not None:
            parts.append(od)
    for y in (y for y in years if y >= 2025):
        nd = _normalized_new_depth(y)
        if nd is not None:
            parts.append(nd)
    if not parts:
        logger.warning('get_weekly_starters(%s): no usable depth charts — empty', years)
        return _empty_starters
    depth = pd.concat(parts, ignore_index=True)
    starters = _starters_from_depth(depth, _load_injuries(years))
    return _empty_starters if starters is None else starters
