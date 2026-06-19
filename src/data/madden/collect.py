# src/data/madden/collect.py
"""Public entry point: assemble team-week Madden features for the given years."""
import pandas as pd
from src.data.madden import ingest, ids, roles, features
from src.data.madden import starters as starters_mod


def _player_season(season):
    """Plan-1 pipeline for one season -> classified player-season frame."""
    df = ingest.load_madden_season(season)
    df = ids.attach_gsis_id(df, season)
    return roles.classify_roles(df)


def get_madden_data(years):
    years = [years] if isinstance(years, int) else list(years)
    players = {s: _player_season(s) for s in years}
    starters = starters_mod.get_weekly_starters(years)

    per_season = []
    for season in years:
        s_rows = starters[starters['season'] == season]
        overalls = features.build_team_week_overalls(s_rows, players[season])
        overalls = features.add_within_season_diffs(overalls)
        if (season - 1) in players:
            prev = features.build_team_week_overalls(
                starters[starters['season'] == season - 1], players[season - 1])
            overalls = features.add_prev_season_diff(overalls, prev)
        per_season.append(overalls)
    return pd.concat(per_season, ignore_index=True)
