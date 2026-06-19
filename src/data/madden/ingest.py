# src/data/madden/ingest.py
"""Download + era-normalize the edgepredictor Madden `raw/` ratings files.

`raw/` mixes several schemas across eras; detection is per-file. Output is one tidy
frame with a fixed column contract regardless of source schema.

Known schemas (2003-2025):
  - Classic (2003, 2009, 2011, 2018-2023): Name/Full Name, Team, Position/POS, Overall/OVR
  - Split-name classic (2012-2017, 2010): First Name / Last Name variants with mixed case
  - Old classic (2004-2008): FIRSTNAME / LASTNAME, no Team -> handled by resilience skip
  - Nested EA-API (2024): firstName/lastName, Team, stats/* columns
  - New EA-API (2025): first_name/last_name, team_name (no abbr column) -> resilience skip
"""
import logging
import pandas as pd
from src.data.transformations import TEAM_ABBR_MAPPINGS

logger = logging.getLogger(__name__)

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
    # position candidates (checked in order; first match wins via the loop below)
    'Position': 'position', 'POSITION': 'position', 'POS': 'position',
    # overall candidates (many era-specific names)
    'Overall': 'overall', 'OVERALL': 'overall',
    'Overall Rating': 'overall', 'OVERALL RATING': 'overall',
    'OVR': 'overall', 'OVERALLRATING': 'overall', 'PLYR_OVERALLRATING': 'overall',
    'Overall_Rating': 'overall',
    # weight
    'Weight': 'weight', 'WEIGHT': 'weight',
    # power moves
    'Power Moves': 'power_moves', 'POWER MOVES': 'power_moves', 'POWERMOVES': 'power_moves',
    'Power Move': 'power_moves', 'POWER MOVE': 'power_moves',
    # finesse moves
    'Finesse Moves': 'finesse_moves', 'FINESSEMOVES': 'finesse_moves',
    'Finessee Move': 'finesse_moves',
}

# 2004-2008 raw files use underscore-separated full team names -> standard abbreviations.
FULL_TEAM_NAME_TO_ABBR = {
    'arizona_cardinals': 'ARI', 'atlanta_falcons': 'ATL', 'baltimore_ravens': 'BAL',
    'buffalo_bills': 'BUF', 'carolina_panthers': 'CAR', 'chicago_bears': 'CHI',
    'cincinnati_bengals': 'CIN', 'cleveland_browns': 'CLE', 'dallas_cowboys': 'DAL',
    'denver_broncos': 'DEN', 'detroit_lions': 'DET', 'green_bay_packers': 'GB',
    'houston_texans': 'HOU', 'indianapolis_colts': 'IND', 'jacksonville_jaguars': 'JAX',
    'kansas_city_chiefs': 'KC', 'oakland_raiders': 'LV', 'los_angeles_chargers': 'LAC',
    'san_diego_chargers': 'LAC', 'los_angeles_rams': 'LA', 'st_louis_rams': 'LA',
    'miami_dolphins': 'MIA', 'minnesota_vikings': 'MIN', 'new_england_patriots': 'NE',
    'new_orleans_saints': 'NO', 'new_york_giants': 'NYG', 'new_york_jets': 'NYJ',
    'philadelphia_eagles': 'PHI', 'pittsburgh_steelers': 'PIT', 'san_francisco_49ers': 'SF',
    'seattle_seahawks': 'SEA', 'tampa_bay_buccaneers': 'TB', 'tennessee_titans': 'TEN',
    'washington_redskins': 'WAS', 'washington_commanders': 'WAS', 'washington_football_team': 'WAS',
    'las_vegas_raiders': 'LV',
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


def _resolve_full_name(df, nested):
    """Build a full_name Series from whichever name columns exist.

    Handles eras where the combined-name column exists but is all NaN (e.g. 2009),
    falling through to split first/last columns in that case."""
    if nested:
        return (df['firstName'].astype(str).str.strip() + ' '
                + df['lastName'].astype(str).str.strip()).str.strip()
    # Combined-name column: use only if it has values in the majority of rows.
    # Some eras (2009) have a `Name` column that is almost entirely NaN with a
    # handful of exceptions — those rows should use First/Last columns instead.
    single = _resolve_column(df, ['Name', 'Full Name', 'full_name'])
    if single is not None:
        series = df[single].astype(str).str.strip()
        real_values = series.replace('nan', pd.NA).notna()
        # Only use the combined-name column if it covers at least half the rows
        if real_values.mean() >= 0.5:
            return series.replace('nan', '')
    # Split first/last: try many era-specific spellings
    first = _resolve_column(df, [
        'First Name', 'FIRST NAME', 'FIRSTNAME', 'First_Name', 'first_name',
        'PLYR_FIRSTNAME', 'First',
    ])
    last = _resolve_column(df, [
        'Last Name', 'LAST NAME', 'LASTNAME', 'Last_Name', 'last_name',
        'PLYR_LASTNAME', 'Last',
    ])
    if first is not None:
        first_ser = df[first].astype(str).str.strip()
        last_ser = df[last].astype(str).str.strip() if last is not None else ''
        return (first_ser + ' ' + last_ser).str.strip()
    return pd.Series([''] * len(df), index=df.index)


def _normalize_team_str(val):
    """Normalize a raw team string: strip whitespace and trailing underscores."""
    if pd.isna(val):
        return val
    return str(val).strip().rstrip('_').strip()


def _resolve_team(df, nested):
    """Return the team Series (standard abbreviations or raw nicknames)."""
    col = _resolve_column(df, ['team', 'Team', 'TEAM', 'PLYR_TEAM'])
    if col is None:
        # 2025 schema: team_name is a full nickname, not an abbreviation
        col = _resolve_column(df, ['team_name'])
    if col is None:
        return pd.Series([pd.NA] * len(df), index=df.index)
    # Normalize: strip whitespace and trailing underscores that appear in some eras
    raw = df[col].map(_normalize_team_str)
    # Try NICKNAME_TO_ABBR first (nickname-style: 'Raiders', 'Steelers', '49ers', etc.)
    mapped = raw.map(NICKNAME_TO_ABBR)
    # Fall back to FULL_TEAM_NAME_TO_ABBR (underscore style: 'pittsburgh_steelers')
    fallback_mask = mapped.isna()
    if fallback_mask.any():
        mapped = mapped.copy()
        mapped[fallback_mask] = raw[fallback_mask].map(FULL_TEAM_NAME_TO_ABBR)
    # Final fallback: pass through as-is (standard abbreviations like 'PIT', 'KC')
    still_na = mapped.isna()
    if still_na.any():
        mapped = mapped.copy()
        mapped[still_na] = raw[still_na]
    return mapped


def normalize_madden_frame(df, season):
    """Map any raw schema era to OUTPUT_COLUMNS, normalize team to standard abbr.

    Returns an empty frame (with OUTPUT_COLUMNS) if the schema is unrecognisable or
    has no usable team column, so the caller can proceed with NaN madden features."""
    nested = is_nested_schema(df.columns)
    out = pd.DataFrame()
    out['season'] = [season] * len(df)
    out['full_name'] = _resolve_full_name(df, nested)
    out['team'] = _resolve_team(df, nested)

    # If team is entirely NA (e.g. 2025 unknown nickname mapping), warn and bail out.
    if out['team'].isna().all():
        logger.warning('season %s: no recognizable team column — returning empty frame', season)
        empty = pd.DataFrame(columns=OUTPUT_COLUMNS)
        return empty

    colmap = NESTED_COLUMN_MAP if nested else CLASSIC_COLUMN_MAP
    for src_col, dst_col in colmap.items():
        if src_col in df.columns and dst_col not in out.columns:
            out[dst_col] = df[src_col].values

    out['team'] = out['team'].replace(TEAM_ABBR_MAPPINGS)
    for required in ('position', 'overall', 'weight', 'power_moves', 'finesse_moves'):
        if required not in out.columns:
            out[required] = pd.NA
    return out[OUTPUT_COLUMNS]


def load_madden_season(season):
    """Download and normalize one season's raw Madden ratings.

    Returns an empty frame with OUTPUT_COLUMNS if the season is unavailable or its
    schema cannot be normalized (so the caller can proceed gracefully)."""
    try:
        df = pd.read_csv(RAW_URL_TEMPLATE.format(season=season))
    except Exception as exc:
        logger.warning('season %s: failed to download Madden raw CSV (%s)', season, exc)
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    return normalize_madden_frame(df, season)
