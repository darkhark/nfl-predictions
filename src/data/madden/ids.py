# src/data/madden/ids.py
"""Attach nflverse gsis_id to Madden players by bridging through the edgepredictor
`processed/` file (which carries both `fullname` and the gsis `player_id`). For the
2025+ madden-tools source (no gsis in the payload), `attach_gsis_id_from_rosters`
bridges via nflverse seasonal rosters (whose `player_id` is the gsis id)."""
import re
import pandas as pd
import nfl_data_py as nfl
from src.data.transformations import TEAM_ABBR_MAPPINGS

PROCESSED_URL_TEMPLATE = (
    'https://raw.githubusercontent.com/theedgepredictor/nfl-madden-data/'
    'main/data/madden/processed/{season}.csv'
)
_SUFFIXES = {'jr', 'sr', 'ii', 'iii', 'iv', 'v'}

# Map Madden side-aware positions to the coarse `position` vocab in processed/*.csv
# (processed vocab: QB RB WR TE OL DL LB DB K P LS)
MADDEN_TO_COARSE_POSITION = {
    'QB': 'QB',
    'HB': 'RB', 'FB': 'RB',
    'WR': 'WR',
    'TE': 'TE',
    'LT': 'OL', 'LG': 'OL', 'C': 'OL', 'RG': 'OL', 'RT': 'OL',
    'LE': 'DL', 'RE': 'DL', 'DT': 'DL', 'NT': 'DL',
    'LOLB': 'LB', 'ROLB': 'LB', 'MLB': 'LB', 'OLB': 'LB', 'ILB': 'LB',
    'CB': 'DB', 'FS': 'DB', 'SS': 'DB',
    'K': 'K', 'P': 'P', 'LS': 'LS',
}


def normalize_name(name):
    """Lowercase, drop punctuation and generational suffixes for joining."""
    cleaned = re.sub(r'[^a-z ]', '', str(name).lower())
    tokens = [t for t in cleaned.split() if t not in _SUFFIXES]
    return ' '.join(tokens)


def _build_name_pos_lookup(proc):
    """Build a name|coarse_position -> gsis_id dict; entries with >1 distinct gsis
    are set to None so the fallback never creates a false match.
    Returns an empty dict if `proc` has no `position` column."""
    if 'position' not in proc.columns:
        return {}
    lookup = {}
    for _, row in proc.iterrows():
        k = row['_norm_name'] + '|' + str(row['position'])
        if k in lookup:
            if lookup[k] != row['player_id']:
                lookup[k] = None  # ambiguous — multiple distinct gsis
        else:
            lookup[k] = row['player_id']
    return lookup


def attach_gsis_id(df, season, processed=None):
    if processed is None:
        processed = pd.read_csv(PROCESSED_URL_TEMPLATE.format(season=season))
    proc = processed.copy()
    proc['team'] = proc['team'].replace(TEAM_ABBR_MAPPINGS)
    proc['_norm_name'] = proc['fullname'].map(normalize_name)

    # Primary lookup: normalized_name | team
    proc['_key'] = proc['_norm_name'] + '|' + proc['team'].astype(str)
    lookup_primary = dict(zip(proc['_key'], proc['player_id']))

    # Fallback lookup: normalized_name | coarse_position (no team)
    # Only used when primary misses; skipped if key is ambiguous (None).
    lookup_fallback = _build_name_pos_lookup(proc)

    out = df.copy()
    out['_norm_name'] = out['full_name'].map(normalize_name)

    # Primary match
    primary_keys = out['_norm_name'] + '|' + out['team'].astype(str)
    out['gsis_id'] = primary_keys.map(lookup_primary)

    # Fallback: relax team, disambiguate by coarse position
    mask = out['gsis_id'].isna()
    if mask.any():
        coarse_pos = out.loc[mask, 'position'].map(MADDEN_TO_COARSE_POSITION)
        fallback_keys = out.loc[mask, '_norm_name'] + '|' + coarse_pos.astype(str)
        out.loc[mask, 'gsis_id'] = fallback_keys.map(lookup_fallback)

    out = out.drop(columns=['_norm_name'])
    return out


def attach_gsis_id_from_rosters(df, season, rosters=None):
    """Attach gsis_id by matching to nflverse seasonal rosters (player_id IS gsis).

    Primary key normalized_name|team; fallback name-only (ambiguous -> None). Used
    for the 2025+ madden-tools source, whose JSON carries no gsis/pfr id."""
    if rosters is None:
        rosters = nfl.import_seasonal_rosters([season])
    proc = rosters.copy()
    proc['team'] = proc['team'].replace(TEAM_ABBR_MAPPINGS)
    proc['_norm_name'] = proc['player_name'].map(normalize_name)

    proc['_key'] = proc['_norm_name'] + '|' + proc['team'].astype(str)
    # last-write-wins on duplicate name|team; nflverse seasonal rosters are deduplicated per season
    lookup_primary = dict(zip(proc['_key'], proc['player_id']))

    name_lookup = {}
    for _, r in proc.iterrows():
        n = r['_norm_name']
        if n in name_lookup:
            if name_lookup[n] != r['player_id']:
                name_lookup[n] = None  # ambiguous — multiple distinct gsis
        else:
            name_lookup[n] = r['player_id']

    out = df.copy()
    out['_norm_name'] = out['full_name'].map(normalize_name)
    out['gsis_id'] = (out['_norm_name'] + '|' + out['team'].astype(str)).map(lookup_primary)
    mask = out['gsis_id'].isna()
    if mask.any():
        out.loc[mask, 'gsis_id'] = out.loc[mask, '_norm_name'].map(name_lookup)
    return out.drop(columns=['_norm_name'])
