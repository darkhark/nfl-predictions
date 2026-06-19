"""Leakage-safe weekly starters: depth-chart rank-1 minus Out/Doubtful players,
promoting the next available player at each position. All inputs are pre-kickoff."""
import nfl_data_py as nfl
import pandas as pd
from src.data.transformations import TEAM_ABBR_MAPPINGS

OUT_STATUSES = ('Out', 'Doubtful')


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
    depth = nfl.import_depth_charts(years)
    depth = depth[depth['game_type'] == 'REG'].copy()
    depth['club_code'] = depth['club_code'].replace(TEAM_ABBR_MAPPINGS)
    injuries = nfl.import_injuries(years)
    injuries['team'] = injuries['team'].replace(TEAM_ABBR_MAPPINGS)

    frames = []
    keys = ['season', 'week', 'club_code']
    for (season, week, club), grp in depth.groupby(keys):
        inj = injuries[(injuries['season'] == season) & (injuries['week'] == week)
                       & (injuries['team'] == club)]
        chosen = available_starters(grp, inj)
        chosen['season'], chosen['week'], chosen['team'] = season, week, club
        frames.append(chosen)
    out = pd.concat(frames, ignore_index=True)
    return out[['season', 'week', 'team', 'gsis_id', 'position']]
