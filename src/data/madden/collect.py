# src/data/madden/collect.py
"""Public entry point: assemble team-week Madden features for the given years."""
import logging
import pandas as pd
from src.data.madden import ingest, ids, roles, features, launch
from src.data.madden import starters as starters_mod

logger = logging.getLogger(__name__)


def season_to_game_version(season):
    """NFL season -> madden-tools game version. Madden NFL N covers season N-1, so
    the version number is season-1999 (2025 -> 'madden-26')."""
    return f'madden-{season - 1999}'


def _player_season(season):
    """Plan-1 pipeline for one season -> classified player-season frame.

    Routes by season: >=2025 uses the madden-tools launch JSON + nflverse-rosters
    gsis bridge; <=2024 keeps the theedgepredictor raw CSV + processed/ bridge.
    Returns an empty OUTPUT_COLUMNS frame if the season is unavailable."""
    if season >= 2025:
        df = launch.load_madden_launch(season_to_game_version(season), season)
    else:
        df = ingest.load_madden_season(season)
    if df.empty:
        logger.warning('season %s: empty Madden ingest — skipping', season)
        return df
    try:
        if season >= 2025:
            df = ids.attach_gsis_id_from_rosters(df, season)
        else:
            df = ids.attach_gsis_id(df, season)
        return roles.classify_roles(df)
    except Exception as exc:
        logger.warning('season %s: Madden id/role pipeline failed (%s) — skipping',
                       season, exc)
        return pd.DataFrame()


def get_madden_data(years):
    years = [years] if isinstance(years, int) else list(years)
    players = {}
    for s in years:
        try:
            players[s] = _player_season(s)
        except Exception as exc:
            logger.warning('season %s: _player_season failed (%s) — skipping', s, exc)
            players[s] = pd.DataFrame()
    starters = starters_mod.get_weekly_starters(years)

    per_season = []
    for season in years:
        s_rows = starters[starters['season'] == season]
        season_players = players[season]
        if season_players.empty or s_rows.empty:
            logger.warning('season %s: no starters or players data — skipping overalls', season)
            continue
        overalls = features.build_team_week_overalls(s_rows, season_players)
        if overalls.empty:
            logger.warning('season %s: build_team_week_overalls returned empty', season)
            continue
        overalls = features.add_within_season_diffs(overalls)
        prev_season = season - 1
        if prev_season in players and not players[prev_season].empty:
            prev_starters = starters[starters['season'] == prev_season]
            if not prev_starters.empty:
                prev = features.build_team_week_overalls(prev_starters, players[prev_season])
                if not prev.empty:
                    overalls = features.add_prev_season_diff(overalls, prev)
        per_season.append(overalls)
    if not per_season:
        return pd.DataFrame(columns=['team', 'season', 'week'])
    result = pd.concat(per_season, ignore_index=True)
    result = features.zscore_overalls_within_season(result)
    return result
