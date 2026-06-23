"""Adapt the madden-tools per-iteration `players.json` (+ `teams.json`) launch
snapshot into the shared `ingest.OUTPUT_COLUMNS` contract, so 2025+ launch ratings
flow through the same ids/roles/features pipeline as the historical theedgepredictor
source. Launch is the iteration whose `iterations.json` label == "Launch"."""
import logging
import os

import requests
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


# madden-tools DigitalOcean Spaces CDN. The roster JSON is public-read (the webapp
# fetches it client-side), so a plain GET works — no credentials for reads. Path
# layout mirrors json-generator output: {base}/{version}/json/iterations.json and
# {base}/{version}/json/iterations/{id}/{players,teams}.json
# (verified in madden-tools webapp GuideMetadataCache.ts / draft-genius CDNDataRepository.ts).
# Set MADDEN_TOOLS_CDN_BASE to the confirmed CDN domain (e.g. the value of the
# webapp's NEXT_PUBLIC_CDN_DOMAIN) before the real-data run (Task 7).


def _cdn_json(url):
    """GET a JSON document. The single network seam (patched in tests)."""
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def load_madden_launch(game_version, season, *, cdn_base=None):
    """Fetch + parse the launch-iteration players/teams JSON for `game_version`.

    Returns an empty OUTPUT_COLUMNS frame on any failure (so collect proceeds with
    NaN Madden features for the season)."""
    base = (cdn_base if cdn_base is not None
            else os.environ.get('MADDEN_TOOLS_CDN_BASE', '')).rstrip('/')
    if not base:
        logger.warning('season %s: MADDEN_TOOLS_CDN_BASE unset — skipping launch load', season)
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    try:
        root = f'{base}/{game_version}/json'
        iterations = _cdn_json(f'{root}/iterations.json')
        launch_id = select_launch_iteration(iterations)['id']
        it_dir = f'{root}/iterations/{launch_id}'
        players = _cdn_json(f'{it_dir}/players.json')
        teams = _cdn_json(f'{it_dir}/teams.json')
        return parse_launch_ratings(players, teams, season)
    except Exception as exc:
        logger.warning('season %s: failed to load madden-tools launch (%s)', season, exc)
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
