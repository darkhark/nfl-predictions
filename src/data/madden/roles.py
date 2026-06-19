# src/data/madden/roles.py
"""Classify a Madden player into a scheme/era-invariant positional role.

Unambiguous labels (offense, DBs, DT/MLB, and the modern LEDGE/REDGE/SAM/MIKE/WILL)
map directly. The ambiguous classic labels -- DE (LE/RE) and OLB (LOLB/ROLB) -- are
split by attributes, porting the pre-EDGE guide-generator thresholds:
  edge        = OLB with power_moves+finesse_moves >= 130, or DE with weight <= 280
  interior_dl = DT, or DE with weight >= 280
  off_ball_lb = OLB with power_moves+finesse_moves < 130, or MLB
"""
import pandas as pd

EDGE_PASS_RUSH_THRESHOLD = 130
INTERIOR_DL_WEIGHT_THRESHOLD = 280

POSITION_TO_ROLE = {
    'QB': 'qb', 'HB': 'backfield', 'RB': 'backfield', 'FB': 'backfield',
    'WR': 'receiver', 'TE': 'tight_end',
    'LT': 'exterior_ol', 'RT': 'exterior_ol',
    'LG': 'interior_ol', 'RG': 'interior_ol', 'C': 'interior_ol',
    'DT': 'interior_dl', 'NT': 'interior_dl',
    'MLB': 'off_ball_lb', 'MIKE': 'off_ball_lb', 'WILL': 'off_ball_lb',
    'SAM': 'off_ball_lb', 'ILB': 'off_ball_lb',
    'LEDGE': 'edge', 'REDGE': 'edge',
    'CB': 'cornerback', 'FS': 'safety', 'SS': 'safety',
    'K': 'specialist', 'P': 'specialist', 'LS': 'specialist',
}
_OLB_LABELS = {'LOLB', 'ROLB', 'OLB'}
_DE_LABELS = {'LE', 'RE', 'DE'}


def _num(value):
    return 0.0 if pd.isna(value) else float(value)


def assign_role(position, weight, power_moves, finesse_moves):
    if position in POSITION_TO_ROLE:
        return POSITION_TO_ROLE[position]
    if position in _OLB_LABELS:
        if _num(power_moves) + _num(finesse_moves) >= EDGE_PASS_RUSH_THRESHOLD:
            return 'edge'
        return 'off_ball_lb'
    if position in _DE_LABELS:
        if _num(weight) <= INTERIOR_DL_WEIGHT_THRESHOLD:
            return 'edge'
        return 'interior_dl'
    return 'unknown'


def assign_side(position):
    p = str(position)
    if p.startswith('L'):
        return 'left'
    if p.startswith('R'):
        return 'right'
    return 'none'


def classify_roles(df):
    out = df.copy()
    out['role'] = [
        assign_role(r['position'], r['weight'], r['power_moves'], r['finesse_moves'])
        for _, r in out.iterrows()
    ]
    out['side'] = out['position'].map(assign_side)
    return out
