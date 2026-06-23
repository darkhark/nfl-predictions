"""Adapt the madden-tools per-iteration `players.json` (+ `teams.json`) launch
snapshot into the shared `ingest.OUTPUT_COLUMNS` contract, so 2025+ launch ratings
flow through the same ids/roles/features pipeline as the historical theedgepredictor
source. Launch is the iteration whose `iterations.json` label == "Launch"."""
import logging
import pandas as pd
from src.data.madden.ingest import OUTPUT_COLUMNS
from src.data.transformations import TEAM_ABBR_MAPPINGS

logger = logging.getLogger(__name__)


def select_launch_iteration(iterations):
    """Return the single launch iteration entry (label == 'Launch', case-insensitive)."""
    launch = [it for it in iterations
              if str(it.get('label', '')).strip().lower() == 'launch']
    if len(launch) != 1:
        raise ValueError(f'expected exactly one Launch iteration, found {len(launch)}')
    return launch[0]


def _team_id_to_abbr(teams):
    """madden-tools team_id -> standard NFL abbreviation via teams.json `acronym`."""
    return {t['id']: str(t['acronym']).strip() for t in teams}


def parse_launch_ratings(players, teams, season):
    """Map a launch `players.json` (+ `teams.json`) to ingest.OUTPUT_COLUMNS.

    Empty `players` -> empty frame with OUTPUT_COLUMNS (mirrors ingest's resilience)."""
    if not players:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    id_to_abbr = _team_id_to_abbr(teams)
    rows = []
    for p in players:
        first = str(p.get('first_name', '') or '').strip()
        last = str(p.get('last_name', '') or '').strip()
        rows.append({
            'season': season,
            'full_name': (first + ' ' + last).strip(),
            'team': id_to_abbr.get(p.get('team_id')),
            'position': p.get('position'),
            'overall': p.get('rating_overall'),
            'weight': p.get('weight'),
            'power_moves': p.get('rating_power_moves'),
            'finesse_moves': p.get('rating_finesse_moves'),
        })
    out = pd.DataFrame(rows)
    out['team'] = out['team'].replace(TEAM_ABBR_MAPPINGS)
    return out[OUTPUT_COLUMNS]
