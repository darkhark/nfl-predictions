"""Leakage-safe weekly starters: depth-chart rank-1 minus Out/Doubtful players,
promoting the next available player at each position. All inputs are pre-kickoff."""
import logging
import nfl_data_py as nfl
import pandas as pd
from src.data.transformations import TEAM_ABBR_MAPPINGS

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


def get_weekly_starters(years):
    _empty_starters = pd.DataFrame(
        columns=['season', 'week', 'team', 'gsis_id', 'position'])
    try:
        depth = nfl.import_depth_charts(years)
    except Exception as exc:
        logger.warning('import_depth_charts(%s) unavailable (%s) — returning empty frame',
                       years, exc)
        return _empty_starters
    if depth is None or depth.empty:
        logger.warning('import_depth_charts(%s) returned empty — returning empty frame', years)
        return _empty_starters
    # Validate that the depth-chart frame has the expected schema.
    # The nflverse 2025+ API returns a different schema without game_type/club_code/depth_team.
    required_cols = {'game_type', 'club_code', 'depth_team', 'gsis_id', 'position'}
    missing_cols = required_cols - set(depth.columns)
    if missing_cols:
        logger.warning(
            'import_depth_charts(%s) returned unexpected schema (missing %s) '
            '— returning empty frame', years, missing_cols)
        return _empty_starters
    depth = depth[depth['game_type'] == 'REG'].copy()
    depth['club_code'] = depth['club_code'].replace(TEAM_ABBR_MAPPINGS)
    try:
        injuries = nfl.import_injuries(years)
        injuries['team'] = injuries['team'].replace(TEAM_ABBR_MAPPINGS)
    except Exception as exc:
        logger.warning('import_injuries(%s) unavailable (%s) — treating all players as available',
                       years, exc)
        injuries = pd.DataFrame(columns=['gsis_id', 'report_status', 'team', 'season', 'week'])

    frames = []
    keys = ['season', 'week', 'club_code']
    for (season, week, club), grp in depth.groupby(keys):
        inj = injuries[(injuries['season'] == season) & (injuries['week'] == week)
                       & (injuries['team'] == club)]
        chosen = available_starters(grp, inj)
        chosen['season'], chosen['week'], chosen['team'] = season, week, club
        frames.append(chosen)
    if not frames:
        logger.warning('get_weekly_starters(%s): no depth-chart groups found — returning empty', years)
        return _empty_starters
    out = pd.concat(frames, ignore_index=True)
    return out[['season', 'week', 'team', 'gsis_id', 'position']]
