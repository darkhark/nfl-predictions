"""Normalize nflverse's 2025+ depth-chart schema (dated snapshots, granular
side-aware `pos_abb`, no week/game_type) into the legacy week-keyed contract that
starters.py consumes. For each team-week we take the latest snapshot strictly
before the game day (leakage-safe) and coarsen positions to the pre-2025 vocabulary
so `available_starters` selects one starter per coarse slot — matching the
all-old-schema training distribution."""
import logging
import pandas as pd

logger = logging.getLogger(__name__)

_CONTRACT = ['season', 'week', 'club_code', 'game_type',
             'position', 'depth_team', 'gsis_id']

# new granular pos_abb -> pre-2025 coarse `position`. Unmapped (H/KR/PR returners,
# kept off the starter set just like the old vocabulary) drop out.
POS_ABB_TO_COARSE = {
    'QB': 'QB', 'RB': 'RB', 'FB': 'FB', 'TE': 'TE', 'WR': 'WR', 'C': 'C',
    'LT': 'T', 'RT': 'T', 'LG': 'G', 'RG': 'G',
    'LCB': 'CB', 'RCB': 'CB', 'NB': 'CB', 'FS': 'FS', 'SS': 'SS',
    'LDE': 'DE', 'RDE': 'DE', 'LDT': 'DT', 'RDT': 'DT', 'NT': 'NT',
    'LILB': 'ILB', 'RILB': 'ILB', 'MLB': 'MLB', 'SLB': 'OLB', 'WLB': 'OLB',
    'PK': 'K', 'P': 'P', 'LS': 'LS',
}


def _empty():
    return pd.DataFrame(columns=_CONTRACT)


def _team_gameday(schedule, season):
    """Long frame [week, team, gameday] for REG games (gameday as midnight Timestamp)."""
    reg = schedule[(schedule['season'] == season) & (schedule['game_type'] == 'REG')]
    gameday = pd.to_datetime(reg['gameday'], errors='coerce')
    parts = []
    for side in ('home_team', 'away_team'):
        parts.append(pd.DataFrame({'week': reg['week'].values,
                                   'team': reg[side].values,
                                   'gameday': gameday.values}))
    return pd.concat(parts, ignore_index=True)


def normalize_2025_depth(depth_raw, schedule, season):
    """See module docstring. Returns the `_CONTRACT` frame (possibly empty)."""
    if depth_raw is None or depth_raw.empty or 'pos_abb' not in depth_raw.columns:
        if depth_raw is not None and not depth_raw.empty:
            logger.warning('season %s: depth frame lacks 2025 schema (no pos_abb) '
                           '— quarantining', season)
        return _empty()
    d = depth_raw.copy()
    d['position'] = d['pos_abb'].map(POS_ABB_TO_COARSE)
    d = d[d['position'].notna()].copy()
    d['_dt'] = pd.to_datetime(d['dt'], errors='coerce', utc=True).dt.tz_localize(None)
    schedule_days = _team_gameday(schedule, season)
    frames = []
    for team, team_days in schedule_days.groupby('team'):
        snaps = d[d['team'] == team]
        if snaps.empty:
            continue
        for _, gw in team_days.iterrows():
            pre = snaps[snaps['_dt'] < gw['gameday']]
            if pre.empty:
                continue
            latest = pre[pre['_dt'] == pre['_dt'].max()].copy()
            latest['season'] = season
            latest['week'] = gw['week']
            latest['club_code'] = team
            latest['game_type'] = 'REG'
            latest['depth_team'] = latest['pos_rank']
            frames.append(latest[_CONTRACT])
    return pd.concat(frames, ignore_index=True) if frames else _empty()
