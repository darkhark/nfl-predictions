"""Phase-0 data-quality report: per-season Madden `_ovr` coverage (proves 2025
0% -> populated) and gsis match rate. Run after the season-routing changes land."""
import json
import os
import numpy as np
import pandas as pd
from src.data.madden import collect


def season_ovr_coverage(team_week):
    ovr = [c for c in team_week.columns
           if c.startswith('madden_') and c.endswith('_ovr')]
    if not ovr or team_week.empty:
        return 0.0
    return float(team_week[ovr].notna().to_numpy().mean())


def coverage_report(seasons):
    rows = []
    for s in seasons:
        tw = collect.get_madden_data([s])
        rows.append({'season': s, 'n_team_weeks': int(len(tw)),
                     'ovr_coverage': round(season_ovr_coverage(tw), 4)})
    return {'seasons': rows}


def main():
    seasons = list(range(2021, 2026))
    report = coverage_report(seasons)
    out_dir = 'data/predict_games/madden_coverage'
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'coverage_report.json'), 'w') as fh:
        json.dump(report, fh, indent=2)
    print(f"{'season':>6} {'team_weeks':>11} {'ovr_coverage':>13}")
    for r in report['seasons']:
        print(f"{r['season']:>6} {r['n_team_weeks']:>11} {r['ovr_coverage']:>13.4f}")


if __name__ == '__main__':
    main()
