# src/data/madden/ids.py
"""Attach nflverse gsis_id to Madden players by bridging through the edgepredictor
`processed/` file (which carries both `fullname` and the gsis `player_id`)."""
import re
import pandas as pd
from src.data.transformations import TEAM_ABBR_MAPPINGS

PROCESSED_URL_TEMPLATE = (
    'https://raw.githubusercontent.com/theedgepredictor/nfl-madden-data/'
    'main/data/madden/processed/{season}.csv'
)
_SUFFIXES = {'jr', 'sr', 'ii', 'iii', 'iv', 'v'}


def normalize_name(name):
    """Lowercase, drop punctuation and generational suffixes for joining."""
    cleaned = re.sub(r'[^a-z ]', '', str(name).lower())
    tokens = [t for t in cleaned.split() if t not in _SUFFIXES]
    return ' '.join(tokens)


def attach_gsis_id(df, season, processed=None):
    if processed is None:
        processed = pd.read_csv(PROCESSED_URL_TEMPLATE.format(season=season))
    proc = processed.copy()
    proc['team'] = proc['team'].replace(TEAM_ABBR_MAPPINGS)
    proc['_key'] = proc['fullname'].map(normalize_name) + '|' + proc['team'].astype(str)
    lookup = dict(zip(proc['_key'], proc['player_id']))

    out = df.copy()
    keys = out['full_name'].map(normalize_name) + '|' + out['team'].astype(str)
    out['gsis_id'] = keys.map(lookup)
    out['gsis_id'] = out['gsis_id'].where(out['gsis_id'].notna(), other=pd.NA)
    return out
