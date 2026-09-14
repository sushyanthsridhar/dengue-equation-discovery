"""
Public-health decision-support outputs, regenerated under the corrected
pipeline (referee point 2.1 temporal-split fix).

WHY THIS SCRIPT EXISTS:
The old \\section{Results} in the manuscript reports an alert-threshold sweep,
a province risk ranking, and reliability flags, but those numbers came from
running forecast.py's own __main__ block, which writes everything to
OUT_DIR = 'outputs/' -- a folder that does not exist anywhere in either
Final_Repo or referee_report (confirmed by direct search). Whatever run
originally produced those numbers is not reproducible from anything on disk,
and it predates the corrected temporal split described in Section 1 of
results_report.txt. This script regenerates the same four decision-support
artifacts fresh, under the corrected split and the current production model,
using forecast_temporal_fix.py's own functions unmodified:
compute_risk_ranking, compute_reliability_flags, backtest_lead_time, and
summarize_backtest -- plus the same forward multi-step forecast and driver
attribution the old Results section's per-province forecasts came from.

WHAT IT DOES NOT DO:
It does not change, retune, or reimplement any of these functions. It fits
one production model (fit_winning_model, the same call used everywhere else
in this repo) and runs the existing decision-support functions against it.
This is a single run, not a 5-run stability sweep, since these are
descriptive/operational outputs (a risk ranking, an alert sweep) rather
than an accuracy metric that needs a mean and std across runs.

Usage:
    python3 compute_decision_support.py

Writes (to ./results/, matching where every other results/*.csv in this
project actually lives, not forecast_temporal_fix.py's own unused OUT_DIR):
    results/risk_ranking.csv
    results/reliability_flags.csv
    results/backtest_lead_time_raw.csv
    results/backtest_lead_time_summary.csv
    results/forecast_multistep.csv
    results/driver_attribution.csv
"""
import os

try:
    import forecast as F
except ImportError:
    import forecast_temporal_fix as F

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, 'results')
os.makedirs(OUT_DIR, exist_ok=True)

import numpy as np
import pandas as pd

if __name__ == '__main__':
    print('[STAGE] building province dicts')
    train_pd = F._build_province_dict(F.latent, F.TRAIN_YEARS)
    val_pd = F._build_province_dict(F.latent, F.VAL_YEARS)
    trainval_pd = F._build_province_dict(F.latent, F.TRAINVAL_YEARS)
    test_pd = F._build_province_dict(F.latent, F.TEST_YEARS)

    quality_df = F.compute_province_quality_scores(train_pd)
    included_provinces = set(
        quality_df.loc[quality_df['quality_score'] >= F.QUALITY_THRESHOLD, 'province']
    )

    print('[STAGE] fitting production model (fit_winning_model -- fresh LLM rounds, real Ollama calls)')
    model_final = F.fit_winning_model(train_pd, val_pd, trainval_pd)

    print('[STAGE] one-step-ahead test evaluation (needed for reliability flags)')
    test_r2, test_df_cell, prov_preds = F.evaluate_on_split(test_pd, model_final)
    print(f'[STAGE] test R2 (mean per province, all 28) = {test_r2:.4f}')

    print('[STAGE] computing province risk ranking')
    risk_df = F.compute_risk_ranking(trainval_pd, test_pd)
    risk_df.to_csv(os.path.join(OUT_DIR, 'risk_ranking.csv'), index=False)

    print('[STAGE] computing reliability flags')
    reliability_df = F.compute_reliability_flags(quality_df, test_df_cell)
    reliability_df.to_csv(os.path.join(OUT_DIR, 'reliability_flags.csv'), index=False)

    print('[STAGE] running backtest lead-time / alert-threshold sweep (this is the slow step -- '
          f'{F.BACKTEST_STRIDE}-week stride, {len(F.ALERT_THRESHOLD_SWEEP)} thresholds, 150 sims per anchor)')
    backtest_df = F.backtest_lead_time(
        model_final, test_pd, trainval_pd,
        stride=F.BACKTEST_STRIDE, horizon=F.FORECAST_HORIZON,
        alert_thresholds=F.ALERT_THRESHOLD_SWEEP,
    )
    backtest_summary = F.summarize_backtest(backtest_df)
    backtest_df.to_csv(os.path.join(OUT_DIR, 'backtest_lead_time_raw.csv'), index=False)
    backtest_summary.to_csv(os.path.join(OUT_DIR, 'backtest_lead_time_summary.csv'), index=False)

    pooled_by_thr = backtest_summary[backtest_summary['level'] == 'pooled_by_threshold']
    print('\n[STAGE] alert threshold sweep, pooled across provinces')
    for _, r in pooled_by_thr.iterrows():
        print(f"  cutoff={r['alert_threshold']:.2f}  recall={r['recall']:.2f}  "
              f"precision={r['precision']:.2f}  mean_lead_time_wk={r['mean_lead_time_weeks']:.2f}  "
              f"n_anchors={int(r['n_anchors'])}")

    print('\n[STAGE] forecast simulation and driver attribution (8-week forward paths per province)')
    all_forecast_rows, driver_rows = [], []
    rng_master = np.random.default_rng(F.FS_RANDOM_SEED)
    for prov, pd_t in test_pd.items():
        s_obs, weeks, cid = pd_t['s_obs'], pd_t['weeks'], pd_t['cluster']
        if len(s_obs) <= model_final['n_lags'] + 1:
            continue
        meets_quality_gate = prov in included_provinces
        beta_p = F._beta_for(model_final, prov, cid)
        q_diag_p = model_final['q_diag'].copy()
        if prov in model_final.get('q_inc_by_prov', {}):
            q_diag_p[F.INC_COL] = model_final['q_inc_by_prov'][prov]
        threshold_raw = float(np.percentile(
            F._inc_scaled_to_raw(trainval_pd[prov]['s_obs'][:, F.INC_COL]), F.OUTBREAK_PERCENTILE
        )) if prov in trainval_pd else float('nan')
        paths_raw, fc_weeks = F.simulate_forecast_paths(
            model_final, s_obs, weeks, beta_p, q_diag_p,
            horizon=F.FORECAST_HORIZON, n_sims=F.N_SIM_PATHS, rng=rng_master,
        )
        fc_df = F.summarize_forecast(paths_raw, fc_weeks, threshold_raw)
        fc_df.insert(0, 'province', prov)
        fc_df.insert(1, 'cluster_id', cid)
        fc_df['outbreak_threshold_incidence'] = threshold_raw
        fc_df['meets_quality_gate'] = meets_quality_gate
        all_forecast_rows.append(fc_df)

        drv = F.attribute_drivers(model_final, s_obs, weeks, beta_p, top_k=F.TOP_DRIVER_TERMS)
        drv.insert(0, 'province', prov)
        drv['meets_quality_gate'] = meets_quality_gate
        driver_rows.append(drv)

    forecast_all = pd.concat(all_forecast_rows, ignore_index=True) if all_forecast_rows else pd.DataFrame()
    driver_all = pd.concat(driver_rows, ignore_index=True) if driver_rows else pd.DataFrame()
    forecast_all.to_csv(os.path.join(OUT_DIR, 'forecast_multistep.csv'), index=False)
    driver_all.to_csv(os.path.join(OUT_DIR, 'driver_attribution.csv'), index=False)

    print(f'\nWrote risk_ranking.csv, reliability_flags.csv, backtest_lead_time_raw.csv, '
          f'backtest_lead_time_summary.csv, forecast_multistep.csv, driver_attribution.csv to {OUT_DIR}')
