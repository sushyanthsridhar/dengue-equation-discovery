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
    train_pd = F._build_province_dict(F.latent, F.TRAIN_YEARS)
    val_pd = F._build_province_dict(F.latent, F.VAL_YEARS)
    trainval_pd = F._build_province_dict(F.latent, F.TRAINVAL_YEARS)
    test_pd = F._build_province_dict(F.latent, F.TEST_YEARS)
    quality_df = F.compute_province_quality_scores(train_pd)
    included_provinces = set(quality_df.loc[quality_df['quality_score'] >= F.QUALITY_THRESHOLD, 'province'])
    model_final = F.fit_winning_model(train_pd, val_pd, trainval_pd)
    (test_r2, test_df_cell, prov_preds) = F.evaluate_on_split(test_pd, model_final)
    risk_df = F.compute_risk_ranking(trainval_pd, test_pd)
    risk_df.to_csv(os.path.join(OUT_DIR, 'risk_ranking.csv'), index=False)
    reliability_df = F.compute_reliability_flags(quality_df, test_df_cell)
    reliability_df.to_csv(os.path.join(OUT_DIR, 'reliability_flags.csv'), index=False)
    backtest_df = F.backtest_lead_time(model_final, test_pd, trainval_pd, stride=F.BACKTEST_STRIDE, horizon=F.FORECAST_HORIZON, alert_thresholds=F.ALERT_THRESHOLD_SWEEP)
    backtest_summary = F.summarize_backtest(backtest_df)
    backtest_df.to_csv(os.path.join(OUT_DIR, 'backtest_lead_time_raw.csv'), index=False)
    backtest_summary.to_csv(os.path.join(OUT_DIR, 'backtest_lead_time_summary.csv'), index=False)
    pooled_by_thr = backtest_summary[backtest_summary['level'] == 'pooled_by_threshold']
    for (_, r) in pooled_by_thr.iterrows():
        pass
    (all_forecast_rows, driver_rows) = ([], [])
    rng_master = np.random.default_rng(F.FS_RANDOM_SEED)
    for (prov, pd_t) in test_pd.items():
        (s_obs, weeks, cid) = (pd_t['s_obs'], pd_t['weeks'], pd_t['cluster'])
        if len(s_obs) <= model_final['n_lags'] + 1:
            continue
        meets_quality_gate = prov in included_provinces
        beta_p = F._beta_for(model_final, prov, cid)
        q_diag_p = model_final['q_diag'].copy()
        if prov in model_final.get('q_inc_by_prov', {}):
            q_diag_p[F.INC_COL] = model_final['q_inc_by_prov'][prov]
        threshold_raw = float(np.percentile(F._inc_scaled_to_raw(trainval_pd[prov]['s_obs'][:, F.INC_COL]), F.OUTBREAK_PERCENTILE)) if prov in trainval_pd else float('nan')
        (paths_raw, fc_weeks) = F.simulate_forecast_paths(model_final, s_obs, weeks, beta_p, q_diag_p, horizon=F.FORECAST_HORIZON, n_sims=F.N_SIM_PATHS, rng=rng_master)
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
