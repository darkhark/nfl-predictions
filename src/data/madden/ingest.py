# src/data/madden/ingest.py
"""Download + era-normalize the edgepredictor Madden `raw/` ratings files.

`raw/` mixes two schemas; detection is per-file (presence of any `stats/` column),
never per-year (2023 classic, 2024 nested, 2025 classic). Output is one tidy frame
with a fixed column contract regardless of source schema."""
import pandas as pd
from src.data.transformations import TEAM_ABBR_MAPPINGS

RAW_URL_TEMPLATE = (
    'https://raw.githubusercontent.com/theedgepredictor/nfl-madden-data/'
    'main/data/madden/raw/{season}.csv'
)

# Team nicknames used by the nested (EA-API) files -> standard abbreviations.
NICKNAME_TO_ABBR = {
    'Cardinals': 'ARI', 'Falcons': 'ATL', 'Ravens': 'BAL', 'Bills': 'BUF',
    'Panthers': 'CAR', 'Bears': 'CHI', 'Bengals': 'CIN', 'Browns': 'CLE',
    'Cowboys': 'DAL', 'Broncos': 'DEN', 'Lions': 'DET', 'Packers': 'GB',
    'Texans': 'HOU', 'Colts': 'IND', 'Jaguars': 'JAX', 'Chiefs': 'KC',
    'Raiders': 'LV', 'Chargers': 'LAC', 'Rams': 'LA', 'Dolphins': 'MIA',
    'Vikings': 'MIN', 'Patriots': 'NE', 'Saints': 'NO', 'Giants': 'NYG',
    'Jets': 'NYJ', 'Eagles': 'PHI', 'Steelers': 'PIT', '49ers': 'SF',
    'Seahawks': 'SEA', 'Buccaneers': 'TB', 'Titans': 'TEN', 'Commanders': 'WAS',
}

OUTPUT_COLUMNS = ['season', 'full_name', 'team', 'position', 'overall', 'weight',
                  'power_moves', 'finesse_moves']

CLASSIC_COLUMN_MAP = {
    'Position': 'position', 'Overall': 'overall', 'Overall Rating': 'overall',
    'Weight': 'weight', 'Power Moves': 'power_moves', 'Finesse Moves': 'finesse_moves',
}
NESTED_COLUMN_MAP = {
    'Position': 'position', 'stats/overall/value': 'overall', 'weight': 'weight',
    'stats/powerMoves/value': 'power_moves', 'stats/finesseMoves/value': 'finesse_moves',
}


def is_nested_schema(columns):
    """Nested (EA-API) files carry flattened `stats/<attr>/value` columns."""
    return any('stats/' in str(c) for c in columns)


def _resolve_column(df, candidates):
    """Return the first candidate column present in df, else None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def normalize_madden_frame(df, season):
    """Map either raw schema to OUTPUT_COLUMNS, normalize team to standard abbr."""
    nested = is_nested_schema(df.columns)
    out = pd.DataFrame()
    out['season'] = [season] * len(df)
    if nested:
        out['full_name'] = (df['firstName'].astype(str).str.strip() + ' '
                            + df['lastName'].astype(str).str.strip()).str.strip()
        out['team'] = df['team'].map(NICKNAME_TO_ABBR).fillna(df['team'])
        colmap = NESTED_COLUMN_MAP
    else:
        name_col = _resolve_column(df, ['Name', 'full_name'])
        out['full_name'] = df[name_col].astype(str).str.strip()
        out['team'] = df['Team']
        colmap = CLASSIC_COLUMN_MAP
    for src_col, dst_col in colmap.items():
        if src_col in df.columns and dst_col not in out.columns:
            out[dst_col] = df[src_col].values
    out['team'] = out['team'].replace(TEAM_ABBR_MAPPINGS)
    for required in ('position', 'overall', 'weight', 'power_moves', 'finesse_moves'):
        if required not in out.columns:
            out[required] = pd.NA
    return out[OUTPUT_COLUMNS]


def load_madden_season(season):
    """Download and normalize one season's raw Madden ratings."""
    df = pd.read_csv(RAW_URL_TEMPLATE.format(season=season))
    return normalize_madden_frame(df, season)
