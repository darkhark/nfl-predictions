# Madden Group Ablation Results

## Experiment: Z-Scored vs Raw _ovr Levels (2026-06-19)

The `madden_*_ovr` columns were z-scored within season (commit f73b395) to remove cross-era
Madden rating drift. A 7-fold rolling-origin ablation (test seasons 2019-2025) compared
base-vs-base+madden_ratings under both raw and z-scored conditions.

### Summary

| Condition | roc_auc_delta_mean | brier_delta_mean | Folds favoring Madden |
|---|---|---|---|
| RAW _ovr (baseline) | +0.0075 | -0.0016 | 6/7 |
| Z-SCORED _ovr | **+0.0129** | **-0.0034** | 6/7 |

**Verdict: Z-scoring HELPED substantially.** ROC-AUC signal improved +72%; Brier signal doubled.
Fold 2021 flipped from Madden hurting (-0.0087) to helping (+0.0087) under z-scoring.

### Files

- `madden_ablation_raw.json` — RAW _ovr ablation (before z-scoring)
- `madden_ablation_zscore.json` — Z-SCORED _ovr ablation (after commit f73b395)
- `madden_ablation.json` — Latest run (same as zscore)

Full report: `.git/sdd/zscore-ablation-report.md`
