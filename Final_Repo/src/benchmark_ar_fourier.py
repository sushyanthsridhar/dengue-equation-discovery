import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
(_stdout_before, _stderr_before) = (sys.stdout, sys.stderr)
try:
    import forecast as F
except ModuleNotFoundError:
    import forecast_temporal_fix as F
(sys.stdout, sys.stderr) = (_stdout_before, _stderr_before)
OUT_DIR = F.OUT_DIR
os.makedirs(OUT_DIR, exist_ok=True)
train_pd = F._build_province_dict(F.latent, F.TRAIN_YEARS)
val_pd = F._build_province_dict(F.latent, F.VAL_YEARS)
trainval_pd = F._build_province_dict(F.latent, F.TRAINVAL_YEARS)
test_pd = F._build_province_dict(F.latent, F.TEST_YEARS)
quality_df = F.compute_province_quality_scores(train_pd)
included_provinces = set(quality_df.loc[quality_df['quality_score'] >= F.QUALITY_THRESHOLD, 'province'])
with open(F.SAVED_FS_JSON) as fh:
    _fs_best = F.Program.from_dict(json.load(fh))
N_LAGS = max(_fs_best.n_lags, 3)
MEAN_INC = float(F.SCALER.mean_[F.INC_COL])
STD_INC = float(F.SCALER.scale_[F.INC_COL])

def _inverse_incidence(y_scaled):
    y_log = y_scaled * STD_INC + MEAN_INC
    return np.expm1(y_log) if F.LOG_INCIDENCE else y_log

def build_ar_fourier_xy(province_data, n_lags):
    (ar_parts, four_parts, y_parts, prov_parts) = ([], [], [], [])
    for (prov, pd_t) in province_data.items():
        s_obs = pd_t['s_obs']
        weeks = pd_t['weeks']
        inc = s_obs[:, F.INC_COL]
        T = len(inc)
        if T <= n_lags + 1:
            continue
        (rows, targets, weeks_used) = ([], [], [])
        for t in range(n_lags, T - 1):
            rows.append([inc[t - lag] for lag in range(n_lags + 1)])
            targets.append(inc[t + 1])
            weeks_used.append(weeks[t])
        if not rows:
            continue
        ar_parts.append(np.array(rows, dtype=np.float32))
        y_parts.append(np.array(targets, dtype=np.float32))
        w = np.array(weeks_used, dtype=np.float64)
        t_rad52 = 2.0 * np.pi * w / 52.0
        t_rad26 = 2.0 * np.pi * w / 26.0
        four_parts.append(np.stack([np.sin(t_rad52), np.cos(t_rad52), np.sin(t_rad26), np.cos(t_rad26)], axis=1).astype(np.float32))
        prov_parts.append(np.array([prov] * len(targets)))
    if not ar_parts:
        return (np.zeros((0, n_lags + 1), dtype=np.float32), np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.array([]))
    return (np.concatenate(ar_parts, axis=0), np.concatenate(four_parts, axis=0), np.concatenate(y_parts, axis=0), np.concatenate(prov_parts, axis=0))

def per_province_r2(y_true_raw, y_pred_raw, prov_labels, included):
    df = pd.DataFrame({'province': prov_labels, 'y_true': y_true_raw, 'y_pred': y_pred_raw})
    rows = []
    for (prov, g) in df.groupby('province'):
        if len(g) < 2:
            continue
        rows.append({'province': prov, 'r2': float(r2_score(g['y_true'], g['y_pred'])), 'mae': float(np.mean(np.abs(g['y_true'] - g['y_pred']))), 'mse': float(np.mean((g['y_true'] - g['y_pred']) ** 2)), 'n': len(g)})
    prov_df = pd.DataFrame(rows).sort_values('r2', ascending=False) if rows else pd.DataFrame(columns=['province', 'r2', 'mae', 'mse', 'n'])
    incl = prov_df[prov_df['province'].isin(included)]
    r2_filt = float(incl['r2'].mean()) if len(incl) else float('nan')
    return (prov_df, r2_filt)
(ar_train, four_train, y_train, prov_train) = build_ar_fourier_xy(train_pd, N_LAGS)
(ar_val, four_val, y_val, prov_val) = build_ar_fourier_xy(val_pd, N_LAGS)
(ar_trainval, four_trainval, y_trainval, prov_trainval) = build_ar_fourier_xy(trainval_pd, N_LAGS)
(ar_test, four_test, y_test, prov_test) = build_ar_fourier_xy(test_pd, N_LAGS)
y_test_raw = _inverse_incidence(y_test)
summary_rows = []
for (variant_name, use_fourier) in (('ar_only', False), ('ar_fourier', True)):
    if use_fourier:
        X_train = np.concatenate([ar_train, four_train], axis=1)
        X_val = np.concatenate([ar_val, four_val], axis=1)
        X_trainval = np.concatenate([ar_trainval, four_trainval], axis=1)
        X_test = np.concatenate([ar_test, four_test], axis=1)
    else:
        (X_train, X_val, X_trainval, X_test) = (ar_train, ar_val, ar_trainval, ar_test)
    model_dev = LinearRegression()
    model_dev.fit(X_train, y_train)
    val_pred_raw = _inverse_incidence(model_dev.predict(X_val))
    val_true_raw = _inverse_incidence(y_val)
    val_r2 = float(r2_score(val_true_raw, val_pred_raw)) if len(y_val) else float('nan')
    model_final = LinearRegression()
    model_final.fit(X_trainval, y_trainval)
    test_pred_raw = _inverse_incidence(model_final.predict(X_test))
    test_r2_pooled = float(r2_score(y_test_raw, test_pred_raw))
    (prov_df, test_r2_filt) = per_province_r2(y_test_raw, test_pred_raw, prov_test, included_provinces)
    prov_df.to_csv(os.path.join(OUT_DIR, f'{variant_name}_per_province.csv'), index=False)
    summary_rows.append({'variant': variant_name, 'val_r2': val_r2, 'test_r2_pooled': test_r2_pooled, 'test_r2_mean_per_province': test_r2_filt, 'n_features': X_trainval.shape[1]})
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(OUT_DIR, 'ar_fourier_benchmark.csv'), index=False)
