import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import forecast
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, 'results')
os.makedirs(OUT_DIR, exist_ok=True)
N_RUNS = forecast.N_STABILITY_RUNS
HORIZON = forecast.FORECAST_HORIZON
STRIDE = forecast.BACKTEST_STRIDE
N_SIMS_HORIZON = 150

def horizon_predictions_for_run(model, test_pd, trainval_pd, included_provinces):
    n_lags = model['n_lags']
    rows = []
    for (prov, pd_t) in test_pd.items():
        (s_obs, weeks, cid) = (pd_t['s_obs'], pd_t['weeks'], pd_t['cluster'])
        T = len(s_obs)
        if T <= n_lags + HORIZON + 2:
            continue
        raw_inc_full = forecast._inc_scaled_to_raw(s_obs[:, forecast.INC_COL])
        beta_p = forecast._beta_for(model, prov, cid)
        q_diag_p = model['q_diag'].copy()
        if prov in model.get('q_inc_by_prov', {}):
            q_diag_p[forecast.INC_COL] = model['q_inc_by_prov'][prov]
        rng = np.random.default_rng(abs(hash(prov)) % 2 ** 31)
        for t0 in range(n_lags, T - HORIZON - 1, STRIDE):
            (s_hist, w_hist) = (s_obs[:t0 + 1], weeks[:t0 + 1])
            (paths_raw, _) = forecast.simulate_forecast_paths(model, s_hist, w_hist, beta_p, q_diag_p, horizon=HORIZON, n_sims=N_SIMS_HORIZON, rng=rng)
            point_est = np.median(paths_raw, axis=0)
            actual_future = raw_inc_full[t0 + 1:t0 + 1 + HORIZON]
            for h in range(HORIZON):
                rows.append({'province': prov, 'included_in_headline': prov in included_provinces, 'horizon_weeks': h + 1, 'y_true': float(actual_future[h]), 'y_hat': float(point_est[h])})
    return rows

def r2_by_horizon(rows_df: pd.DataFrame, provinces_filter=None) -> pd.DataFrame:
    df = rows_df if provinces_filter is None else rows_df[rows_df['province'].isin(provinces_filter)]
    out = []
    for (h, g) in df.groupby('horizon_weeks'):
        pooled = float(r2_score(g['y_true'], g['y_hat']))
        per_prov = g.groupby('province').apply(lambda gg: r2_score(gg['y_true'], gg['y_hat']) if len(gg) > 1 else float('nan'))
        out.append({'horizon_weeks': h, 'pooled_r2': pooled, 'mean_per_province_r2': float(per_prov.mean())})
    return pd.DataFrame(out).sort_values('horizon_weeks').reset_index(drop=True)
if __name__ == '__main__':
    train_pd = forecast._build_province_dict(forecast.latent, forecast.TRAIN_YEARS)
    val_pd = forecast._build_province_dict(forecast.latent, forecast.VAL_YEARS)
    trainval_pd = forecast._build_province_dict(forecast.latent, forecast.TRAINVAL_YEARS)
    test_pd = forecast._build_province_dict(forecast.latent, forecast.TEST_YEARS)
    quality_df = forecast.compute_province_quality_scores(train_pd)
    included_provinces = set(quality_df.loc[quality_df['quality_score'] >= forecast.QUALITY_THRESHOLD, 'province'])
    all_run_rows = []
    for run_idx in range(1, N_RUNS + 1):
        model_final = forecast.fit_winning_model(train_pd, val_pd, trainval_pd)
        pred_rows = horizon_predictions_for_run(model_final, test_pd, trainval_pd, included_provinces)
        pred_df = pd.DataFrame(pred_rows)
        r2_all = r2_by_horizon(pred_df)
        r2_filt = r2_by_horizon(pred_df, provinces_filter=included_provinces)
        merged = r2_all.merge(r2_filt, on='horizon_weeks', suffixes=('_all28', '_filtered26'))
        merged.insert(0, 'run', run_idx)
        all_run_rows.append(merged)
        for (_, r) in merged.iterrows():
            pass
    raw_df = pd.concat(all_run_rows, ignore_index=True)
    raw_df.to_csv(os.path.join(OUT_DIR, 'horizon_skill_runs_raw.csv'), index=False)
    metric_cols = [c for c in raw_df.columns if c not in ('run', 'horizon_weeks')]
    summary_rows = []
    for (h, g) in raw_df.groupby('horizon_weeks'):
        agg = g[metric_cols].agg(['mean', 'std', 'min', 'max'])
        for metric in metric_cols:
            summary_rows.append({'horizon_weeks': h, 'metric': metric, 'mean': agg.loc['mean', metric], 'std': agg.loc['std', metric], 'min': agg.loc['min', metric], 'max': agg.loc['max', metric]})
    summary_df = pd.DataFrame(summary_rows).sort_values(['horizon_weeks', 'metric']).reset_index(drop=True)
    summary_df.to_csv(os.path.join(OUT_DIR, 'horizon_skill_summary.csv'), index=False)
    for (_, r) in summary_df.iterrows():
        pass
