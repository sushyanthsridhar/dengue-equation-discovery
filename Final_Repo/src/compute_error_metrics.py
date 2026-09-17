import os
import numpy as np
import pandas as pd
import forecast
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, 'results')
os.makedirs(OUT_DIR, exist_ok=True)
N_RUNS = forecast.N_STABILITY_RUNS

def _pooled_errors(prov_preds: dict, provinces_to_use) -> dict:
    (y_true_all, y_hat_all) = ([], [])
    for (prov, (y_true, y_hat)) in prov_preds.items():
        if prov not in provinces_to_use:
            continue
        y_true_all.append(y_true)
        y_hat_all.append(y_hat)
    if not y_true_all:
        return {'mae': float('nan'), 'rmse': float('nan'), 'nrmse': float('nan'), 'mean_actual_incidence': float('nan')}
    y_true_all = np.concatenate(y_true_all)
    y_hat_all = np.concatenate(y_hat_all)
    mean_actual = float(np.mean(y_true_all))
    mae = float(np.mean(np.abs(y_true_all - y_hat_all)))
    rmse = float(np.sqrt(np.mean((y_true_all - y_hat_all) ** 2)))
    nrmse = rmse / mean_actual if mean_actual > 0 else float('nan')
    return {'mae': mae, 'rmse': rmse, 'nrmse': nrmse, 'mean_actual_incidence': mean_actual}
if __name__ == '__main__':
    train_pd = forecast._build_province_dict(forecast.latent, forecast.TRAIN_YEARS)
    val_pd = forecast._build_province_dict(forecast.latent, forecast.VAL_YEARS)
    trainval_pd = forecast._build_province_dict(forecast.latent, forecast.TRAINVAL_YEARS)
    test_pd = forecast._build_province_dict(forecast.latent, forecast.TEST_YEARS)
    quality_df = forecast.compute_province_quality_scores(train_pd)
    included_provinces = set(quality_df.loc[quality_df['quality_score'] >= forecast.QUALITY_THRESHOLD, 'province'])
    all_provinces = set(quality_df['province'])
    per_province_rows = []
    run_rows = []
    for run_idx in range(1, N_RUNS + 1):
        model_final = forecast.fit_winning_model(train_pd, val_pd, trainval_pd)
        (test_r2_mean_per_prov, test_df_cell, prov_preds) = forecast.evaluate_on_split(test_pd, model_final)
        for (_, prow) in test_df_cell.iterrows():
            prov = prow['province']
            (y_true, y_hat) = prov_preds[prov]
            mean_actual = float(np.mean(y_true))
            rmse = float(np.sqrt(prow['mse']))
            nrmse = rmse / mean_actual if mean_actual > 0 else float('nan')
            per_province_rows.append({'run': run_idx, 'province': prov, 'cluster_id': prow['cluster_id'], 'r2': prow['r2'], 'mae': prow['mae'], 'rmse': rmse, 'nrmse': nrmse, 'mean_actual_incidence': mean_actual, 'included_in_headline': prov in included_provinces})
        per_province_df_this_run = pd.DataFrame([r for r in per_province_rows if r['run'] == run_idx])
        incl_this_run = per_province_df_this_run[per_province_df_this_run['included_in_headline']]
        pooled_all28 = _pooled_errors(prov_preds, all_provinces)
        pooled_filt26 = _pooled_errors(prov_preds, included_provinces)
        run_rows.append({'run': run_idx, 'mean_per_province_mae_all28': float(per_province_df_this_run['mae'].mean()), 'mean_per_province_rmse_all28': float(per_province_df_this_run['rmse'].mean()), 'mean_per_province_nrmse_all28': float(per_province_df_this_run['nrmse'].mean()), 'mean_per_province_mae_filtered26': float(incl_this_run['mae'].mean()), 'mean_per_province_rmse_filtered26': float(incl_this_run['rmse'].mean()), 'mean_per_province_nrmse_filtered26': float(incl_this_run['nrmse'].mean()), 'pooled_mae_all28': pooled_all28['mae'], 'pooled_rmse_all28': pooled_all28['rmse'], 'pooled_nrmse_all28': pooled_all28['nrmse'], 'pooled_mean_actual_incidence_all28': pooled_all28['mean_actual_incidence'], 'pooled_mae_filtered26': pooled_filt26['mae'], 'pooled_rmse_filtered26': pooled_filt26['rmse'], 'pooled_nrmse_filtered26': pooled_filt26['nrmse'], 'pooled_mean_actual_incidence_filtered26': pooled_filt26['mean_actual_incidence']})
    per_province_df = pd.DataFrame(per_province_rows)
    per_province_df.to_csv(os.path.join(OUT_DIR, 'error_metrics_per_province_raw.csv'), index=False)
    run_df = pd.DataFrame(run_rows)
    run_df.to_csv(os.path.join(OUT_DIR, 'error_metrics_runs_raw.csv'), index=False)
    metric_cols = [c for c in run_df.columns if c != 'run']
    summary = run_df[metric_cols].agg(['mean', 'std', 'min', 'max']).T
    summary.index.name = 'metric'
    summary = summary.reset_index()
    summary.to_csv(os.path.join(OUT_DIR, 'error_metrics_summary.csv'), index=False)
    for (_, r) in summary.iterrows():
        pass
