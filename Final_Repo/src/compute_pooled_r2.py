import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import forecast
HERE = __import__('os').path.dirname(__import__('os').path.abspath(__file__))
OUT_DIR = __import__('os').path.join(HERE, 'results')
__import__('os').makedirs(OUT_DIR, exist_ok=True)
N_RUNS = forecast.N_STABILITY_RUNS

def pooled_r2_from_prov_preds(prov_preds: dict, provinces_to_use) -> float:
    (y_true_all, y_hat_all) = ([], [])
    for (prov, (y_true, y_hat)) in prov_preds.items():
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
    included_provinces = set(quality_df.loc[quality_df['quality_score'] >= forecast.QUALITY_THRESHOLD, 'province'])
    all_provinces = set(quality_df['province'])
    rows = []
    for run_idx in range(1, N_RUNS + 1):
        model_final = forecast.fit_winning_model(train_pd, val_pd, trainval_pd)
        (test_r2_mean_per_prov, test_df_cell, prov_preds) = forecast.evaluate_on_split(test_pd, model_final)
        test_df_incl = test_df_cell[test_df_cell['province'].isin(included_provinces)]
        test_r2_filt_mean_per_prov = float(test_df_incl['r2'].mean()) if len(test_df_incl) else float('nan')
        pooled_all28 = pooled_r2_from_prov_preds(prov_preds, all_provinces)
        pooled_filt26 = pooled_r2_from_prov_preds(prov_preds, included_provinces)
        rows.append({'run': run_idx, 'pooled_r2_all28': pooled_all28, 'pooled_r2_filtered26': pooled_filt26, 'mean_per_province_r2_filtered26_crosscheck': test_r2_filt_mean_per_prov, 'n_llm_terms_surviving': len(model_final['llm_formulas_used'])})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, 'pooled_r2_runs_raw.csv'), index=False)
    metric_cols = [c for c in df.columns if c != 'run']
    summary = df[metric_cols].agg(['mean', 'std', 'min', 'max']).T
    summary.index.name = 'metric'
    summary = summary.reset_index()
    summary.to_csv(os.path.join(OUT_DIR, 'pooled_r2_summary.csv'), index=False)
    for (_, r) in summary.iterrows():
        pass
