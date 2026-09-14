"""
Pooled test R2 for the full hierarchical model, mean of N_STABILITY_RUNS runs.

WHY THIS SCRIPT EXISTS:
forecast.py's own main block already builds main_benchmark_table.csv with a
'test_r2_pooled' column, but hardcodes it to NaN for this model (WINNING_LABEL)
-- pooled R2 was computed for the naive baselines but never for the sparse
model itself. This script fills that specific gap, using the exact same
production model-fitting and evaluation code forecast.py already uses
(fit_winning_model, evaluate_on_split), so the numbers are directly
comparable to everything already in results_report.txt and to
benchmark_lstm_gb_sindy.py / benchmark_ar_fourier.py, which already report
pooled test R2 for the other baselines.

WHAT "POOLED" MEANS HERE:
evaluate_on_split() returns, per province, an aligned (y_true, y_hat) pair in
real case-rate units (already inverse-log-transformed). This script
concatenates those arrays across every province into one long vector and
computes ONE R2 on the combined data -- not an average of per-province R2
values. This is computed twice per run: once across all 28 modelled
provinces (primary), and once restricted to the 26 provinces that clear the
quality gate (secondary, matching the paper's existing filtered convention).

WHAT IT DOES NOT DO:
It does not change, refit, or re-tune anything about the model. It imports
forecast.py as-is (which runs the same data loading and setup forecast.py's
own __main__ block relies on) and calls the same fit_winning_model /
evaluate_on_split functions forecast.py's stability loop already calls, so
this reuses the exact production model and evaluation logic rather than
reimplementing anything. Each of the N_STABILITY_RUNS runs is therefore a
fresh model fit (winning FunSearch structure refit + a fresh live LLM
round loop against Ollama), same as every other stability run reported in
this repo -- expect this to take roughly the same wall-clock time per run
as forecast.py's own stability loop (LLM calls dominate, ~40s/round x 4
rounds/run per the existing llm_diagnostics.py numbers).

Usage:
    python3 compute_pooled_r2.py

Writes:
    results/pooled_r2_runs_raw.csv      -- one row per run: pooled R2 (all
        28 provinces), pooled R2 (26-province quality-filtered subset), and
        the existing mean-per-province R2 (filtered) for cross-check against
        stability_runs_raw.csv's test_r2_filt column, which should match.
    results/pooled_r2_summary.csv       -- mean/std/min/max of the above
        across all runs.
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

import forecast  # noqa: E402  (importing runs forecast.py's module-level data/setup code)

HERE = __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__)))  # repo root: this script now lives one level down, in src/
OUT_DIR = __import__('os').path.join(HERE, 'results')  # matches where every other results/*.csv already lives, NOT forecast.OUT_DIR ('outputs/'), which is unused/stale in this repo
__import__('os').makedirs(OUT_DIR, exist_ok=True)
N_RUNS = forecast.N_STABILITY_RUNS


def pooled_r2_from_prov_preds(prov_preds: dict, provinces_to_use) -> float:
    """Concatenate per-province (y_true, y_hat) arrays for the given
    provinces into one vector each, then compute a single R2 on the
    combined data (not an average of per-province R2 values)."""
    y_true_all, y_hat_all = [], []
    for prov, (y_true, y_hat) in prov_preds.items():
        if prov not in provinces_to_use:
            continue
        y_true_all.append(y_true)
        y_hat_all.append(y_hat)
    if not y_true_all:
        return float('nan')
    y_true_all = np.concatenate(y_true_all)
    y_hat_all = np.concatenate(y_hat_all)
    return float(r2_score(y_true_all, y_hat_all))


if __name__ == '__main__':
    train_pd = forecast._build_province_dict(forecast.latent, forecast.TRAIN_YEARS)
    val_pd = forecast._build_province_dict(forecast.latent, forecast.VAL_YEARS)
    trainval_pd = forecast._build_province_dict(forecast.latent, forecast.TRAINVAL_YEARS)
    test_pd = forecast._build_province_dict(forecast.latent, forecast.TEST_YEARS)

    quality_df = forecast.compute_province_quality_scores(train_pd)
    included_provinces = set(
        quality_df.loc[quality_df['quality_score'] >= forecast.QUALITY_THRESHOLD, 'province']
    )
    all_provinces = set(quality_df['province'])

    rows = []
    for run_idx in range(1, N_RUNS + 1):
        print(f'\n[STAGE] pooled-R2 run {run_idx}/{N_RUNS}')
        model_final = forecast.fit_winning_model(train_pd, val_pd, trainval_pd)
        test_r2_mean_per_prov, test_df_cell, prov_preds = forecast.evaluate_on_split(test_pd, model_final)

        # cross-check: mean-per-province R2, filtered to the 26-province
        # quality-gated subset -- should match stability_runs_raw.csv's
        # test_r2_filt column for a comparable run.
        test_df_incl = test_df_cell[test_df_cell['province'].isin(included_provinces)]
        test_r2_filt_mean_per_prov = float(test_df_incl['r2'].mean()) if len(test_df_incl) else float('nan')

        pooled_all28 = pooled_r2_from_prov_preds(prov_preds, all_provinces)
        pooled_filt26 = pooled_r2_from_prov_preds(prov_preds, included_provinces)

        rows.append({
            'run': run_idx,
            'pooled_r2_all28': pooled_all28,
            'pooled_r2_filtered26': pooled_filt26,
            'mean_per_province_r2_filtered26_crosscheck': test_r2_filt_mean_per_prov,
            'n_llm_terms_surviving': len(model_final['llm_formulas_used']),
        })
        print(f'[STAGE] pooled-R2 run {run_idx}/{N_RUNS} done  '
              f'pooled_all28={pooled_all28:.4f}  pooled_filtered26={pooled_filt26:.4f}  '
              f'mean_per_province_filtered26={test_r2_filt_mean_per_prov:.4f}')

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, 'pooled_r2_runs_raw.csv'), index=False)

    metric_cols = [c for c in df.columns if c != 'run']
    summary = df[metric_cols].agg(['mean', 'std', 'min', 'max']).T
    summary.index.name = 'metric'
    summary = summary.reset_index()
    summary.to_csv(os.path.join(OUT_DIR, 'pooled_r2_summary.csv'), index=False)

    print(f'\n[STAGE] pooled R2 summary, mean / std over {N_RUNS} runs')
    for _, r in summary.iterrows():
        print(f"  {r['metric']:42s}  mean={r['mean']:+.4f}  std={r['std']:.4f}  "
              f"min={r['min']:+.4f}  max={r['max']:+.4f}")

    print(f"\nWrote results/pooled_r2_runs_raw.csv and results/pooled_r2_summary.csv to {OUT_DIR}")
