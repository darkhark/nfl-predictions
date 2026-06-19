"""Build team-week Madden overall columns from weekly starters + the player-season
table, then within-season / season-over-season diffs and post-reframe matchup deltas."""
import pandas as pd

# granular single-player slots keyed by (madden_position predicate)
_OL_SIDE = {'left': 'lt', 'right': 'rt'}
GROUP_ROLES = {
    'backfield': 'backfield', 'receivers': 'receiver', 'tight_end': 'tight_end',
    'interior_ol': 'interior_ol', 'exterior_ol': 'exterior_ol', 'edge': 'edge',
    'interior_dl': 'interior_dl', 'linebacker': 'off_ball_lb',
    'cornerback': 'cornerback', 'safety': 'safety',
}

GRANULAR_SLOTS = [
    'qb', 'rb', 'te',
    'wr1', 'wr2', 'wr3',
    'lt', 'lg', 'c', 'rg', 'rt',
    'edge_left', 'edge_right',
]


def _slot_of(player):
    """Map a starting player (joined to Madden) to a granular slot name, or None."""
    pos, side, role = player['position'], player['side'], player['role']
    if role == 'qb':
        return 'qb'
    if role == 'backfield':
        return 'rb'
    if role == 'tight_end':
        return 'te'
    if pos in ('LG', 'RG', 'C'):
        return pos.lower()
    if role == 'exterior_ol':
        return _OL_SIDE.get(side)            # lt / rt
    if role == 'edge':
        return 'edge_left' if side == 'left' else 'edge_right' if side == 'right' else None
    return None  # WRs handled by ranking; interior_dl/lb/db only feed groups


def build_team_week_overalls(starters, players):
    starters_renamed = starters.rename(columns={'position': 'nfl_position'})
    joined = starters_renamed.merge(
        players[['gsis_id', 'overall', 'position', 'role', 'side']],
        on='gsis_id', how='left')
    joined = joined[joined['overall'].notna()]
    records = []
    for (season, week, team), grp in joined.groupby(['season', 'week', 'team']):
        rec = {'season': season, 'week': week, 'team': team}
        # single-player slots: highest overall among players mapping to that slot
        grp = grp.assign(_slot=[_slot_of(r) for _, r in grp.iterrows()])
        for slot, sub in grp.dropna(subset=['_slot']).groupby('_slot'):
            rec[f'madden_{slot}_ovr'] = sub['overall'].max()
        # WR1/2/3 by overall desc among receiver-role starters
        wrs = grp[grp['role'] == 'receiver'].sort_values('overall', ascending=False)
        for i, ovr in enumerate(wrs['overall'].head(3).tolist(), start=1):
            rec[f'madden_wr{i}_ovr'] = ovr
        # group means by role
        for group, role in GROUP_ROLES.items():
            members = grp[grp['role'] == role]
            if len(members):
                rec[f'madden_{group}_ovr'] = members['overall'].mean()
        records.append(rec)
    return pd.DataFrame(records)


def _ovr_columns(df):
    return [c for c in df.columns if c.startswith('madden_') and c.endswith('_ovr')]


def add_within_season_diffs(df):
    out = df.sort_values(['team', 'season', 'week']).copy()
    grp = out.groupby(['team', 'season'])
    for col in _ovr_columns(out):
        out[f'{col}_diff_prev'] = out[col] - grp[col].shift(1)
        out[f'{col}_diff_4g'] = out[col] - grp[col].shift(4)
    return out


def add_prev_season_diff(df, prev):
    """Per-team season-over-season launch diff. `prev` is the prior season frame;
    a team's value is constant within a season, so any prior-season row suffices."""
    out = df.copy()
    ovr_cols = _ovr_columns(out)
    prev_by_team = (prev.sort_values('week').groupby('team')[ovr_cols].first())
    for col in ovr_cols:
        mapped = out['team'].map(prev_by_team[col])
        out[f'{col}_diff_prev_season'] = out[col] - mapped
    return out


def _safe(df, col):
    return df[col] if col in df.columns else pd.Series([pd.NA] * len(df), index=df.index)


def add_madden_matchup_columns(df):
    """Directional trench/coverage talent deltas, computed AFTER the target/opp merge."""
    out = df.copy()
    out['madden_matchup_pass_pro'] = (
        _safe(out, 'target_madden_exterior_ol_ovr') - _safe(out, 'opp_madden_edge_ovr'))
    out['madden_matchup_interior'] = (
        _safe(out, 'target_madden_interior_ol_ovr')
        - _safe(out, 'opp_madden_interior_dl_ovr'))
    out['madden_matchup_skill_cover'] = (
        _safe(out, 'target_madden_receivers_ovr')
        - (_safe(out, 'opp_madden_cornerback_ovr')
           + _safe(out, 'opp_madden_safety_ovr')) / 2)
    out['madden_matchup_pass_rush'] = (
        _safe(out, 'opp_madden_exterior_ol_ovr') - _safe(out, 'target_madden_edge_ovr'))
    return out
