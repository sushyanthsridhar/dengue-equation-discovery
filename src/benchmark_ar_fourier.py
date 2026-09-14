"""
REFEREE REPORT addendum: "strongly recommended if inexpensive" item under
Section 5 -- "A conventional autoregression + Fourier/dynamic harmonic
baseline."

This adds the classic epidemiological forecasting baseline the referee asks
for: a plain linear model of next-week incidence built only from its own
past values (autoregression) plus fixed annual/semi-annual seasonal terms
(Fourier / dynamic harmonic regression) -- no latent representation, no
FunSearch-discovered terms, no LLM, no hierarchy. This is the simplest,
most standard comparison point in the epidemic-forecasting literature, and
it was the one baseline family missing from the benchmark table (naive
persistence/seasonal-naive, Gradient Boosting, LSTM, and the fixed-library
SINDy pooled model are already covered by earlier scripts).

Reuses forecast.py's data loading, scaler, and temporal split exactly like
benchmark_lstm_gb_sindy.py and llm_vs_random_matched_experiment.py do, so
the comparison is apples to apples: train 2015-2018, FunSearch inner
validation 2018, outer validation 2019, refit 2015-2019, test once on
2022-2023. Same one-step-ahead information-set discipline as every other
benchmark here: only incidence and week-of-year through week t are used to
predict week t+1, nothing from t+1 onward.

Two variants are fit and reported, since the seasonal-vs-autoregressive
split is itself informative:

  ar_only          -- next-week incidence regressed on its own lags
                       (n_lags back, same n_lags as the winning sparse
                       model) only. No seasonal information at all.
  ar_fourier       -- the same autoregressive lags, plus fixed annual
                       (sin/cos at 52 weeks) and semi-annual (sin/cos at
                       26 weeks) Fourier terms -- a standard "dynamic
                       harmonic regression" specification.

Both are pooled (one fit across all provinces, not per-province, not
hierarchical) plain linear regressions (ordinary least squares), fit on
TRAIN+VAL and evaluated once on TEST, with a val_r2 also reported from a
TRAIN-only fit evaluated on VAL for completeness.

Run this AFTER forecast.py (or forecast_temporal_fix.py) has been run at
least once, for the same reason the other benchmark scripts need it: it
picks n_lags from the winning FunSearch program.

This file works unchanged whether it sits next to forecast_temporal_fix.py
(in referee_report/) or has been promoted next to forecast.py (in
Final_Repo/) -- same fallback-import pattern as the other benchmark
scripts.

Usage:
    python3 benchmark_ar_fourier.py

Writes to outputs/:
    ar_fourier_benchmark.csv            -- val/test R2 (pooled + mean per
                                            province) for both variants
    ar_only_per_province.csv            -- per-province R2/MAE/MSE, ar_only
    ar_fourier_per_province.csv         -- per-province R2/MAE/MSE, ar_fourier
"""
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

_stdout_before, _stderr_before = sys.stdout, sys.stderr
try:
    import forecast as F
except ModuleNotFoundError:
    import forecast_temporal_fix as F
# forecast.py / forecast_temporal_fix.py redirects stdout/stderr to its own
# log file as a side effect of import; put ours back for this script's own
# console output.
sys.stdout, sys.stderr = _stdout_before, _stderr_before

print('[ar-fourier] imported', F.__name__, '-- data, scaler and splits loaded from it')

OUT_DIR = F.OUT_DIR
os.makedirs(OUT_DIR, exist_ok=True)

train_pd = F._build_province_dict(F.latent, F.TRAIN_YEARS)
val_pd = F._build_province_dict(F.latent, F.VAL_YEARS)
trainval_pd = F._build_province_dict(F.latent, F.TRAINVAL_YEARS)
test_pd = F._build_province_dict(F.latent, F.TEST_YEARS)

quality_df = F.compute_province_quality_scores(train_pd)
included_provinces = set(quality_df.loc[quality_df['quality_score'] >= F.QUALITY_THRESHOLD, 'province'])
print(f'[ar-fourier] {len(included_provinces)} of {len(quality_df)} provinces pass the quality gate')

with open(F.SAVED_FS_JSON) as fh:
    _fs_best = F.Program.from_dict(json.load(fh))
N_LAGS = max(_fs_best.n_lags, 3)
print(f'[ar-fourier] using n_lags={N_LAGS} (matches the winning sparse model)')

MEAN_INC = float(F.SCALER.mean_[F.INC_COL])
STD_INC = float(F.SCALER.scale_[F.INC_COL])


def _inverse_incidence(y_scaled):
    y_log = y_scaled * STD_INC + MEAN_INC
    return np.expm1(y_log) if F.LOG_INCIDENCE else y_log


# ---------------------------------------------------------------------------
# Feature construction: pooled, direct one-step-ahead, incidence-only
# ---------------------------------------------------------------------------
def build_ar_fourier_xy(province_data, n_lags):
    """Returns pooled AR-lag columns (n_lags+1 of them), Fourier columns
    (annual + semi-annual sin/cos, 4 of them), scaled next-step incidence,
    and a per-row province label, across every province in province_data.
    Callers slice out the AR-only columns for the ar_only variant."""
    ar_parts, four_parts, y_parts, prov_parts = [], [], [], []
    for prov, pd_t in province_data.items():
        s_obs = pd_t['s_obs']
        weeks = pd_t['weeks']
        inc = s_obs[:, F.INC_COL]
        T = len(inc)
        if T <= n_lags + 1:
            continue
        rows, targets, weeks_used = [], [], []
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
        return (np.zeros((0, n_lags + 1), dtype=np.float32), np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32), np.array([]))
    return (np.concatenate(ar_parts, axis=0), np.concatenate(four_parts, axis=0),
            np.concatenate(y_parts, axis=0), np.concatenate(prov_parts, axis=0))


def per_province_r2(y_true_raw, y_pred_raw, prov_labels, included):
    df = pd.DataFrame({'province': prov_labels, 'y_true': y_true_raw, 'y_pred': y_pred_raw})
    rows = []
    for prov, g in df.groupby('province'):
        if len(g) < 2:
            continue
        rows.append({
            'province': prov,
            'r2': float(r2_score(g['y_true'], g['y_pred'])),
            'mae': float(np.mean(np.abs(g['y_true'] - g['y_pred']))),
            'mse': float(np.mean((g['y_true'] - g['y_pred']) ** 2)),
            'n': len(g),
        })
    prov_df = pd.DataFrame(rows).sort_values('r2', ascending=False) if rows else pd.DataFrame(
        columns=['province', 'r2', 'mae', 'mse', 'n'])
    incl = prov_df[prov_df['province'].isin(included)]
    r2_filt = float(incl['r2'].mean()) if len(incl) else float('nan')
    return prov_df, r2_filt


ar_train, four_train, y_train, prov_train = build_ar_fourier_xy(train_pd, N_LAGS)
ar_val, four_val, y_val, prov_val = build_ar_fourier_xy(val_pd, N_LAGS)
ar_trainval, four_trainval, y_trainval, prov_trainval = build_ar_fourier_xy(trainval_pd, N_LAGS)
ar_test, four_test, y_test, prov_test = build_ar_fourier_xy(test_pd, N_LAGS)
y_test_raw = _inverse_incidence(y_test)

print(f'[ar-fourier] pooled rows: train={len(ar_train)}  val={len(ar_val)}  trainval={len(ar_trainval)}  test={len(ar_test)}')

summary_rows = []

for variant_name, use_fourier in (('ar_only', False), ('ar_fourier', True)):
    print(f'\n[STAGE] {variant_name}')
    if use_fourier:
        X_train = np.concatenate([ar_train, four_train], axis=1)
        X_val = np.concatenate([ar_val, four_val], axis=1)
        X_trainval = np.concatenate([ar_trainval, four_trainval], axis=1)
        X_test = np.concatenate([ar_test, four_test], axis=1)
    else:
        X_train, X_val, X_trainval, X_test = ar_train, ar_val, ar_trainval, ar_test

    # val_r2: fit on TRAIN only, evaluate on VAL (development-data-only selection, per referee's protocol)
    model_dev = LinearRegression()
    model_dev.fit(X_train, y_train)
    val_pred_raw = _inverse_incidence(model_dev.predict(X_val))
    val_true_raw = _inverse_incidence(y_val)
    val_r2 = float(r2_score(val_true_raw, val_pred_raw)) if len(y_val) else float('nan')

    # final: refit on TRAIN+VAL, evaluate once on TEST
    model_final = LinearRegression()
    model_final.fit(X_trainval, y_trainval)
    test_pred_raw = _inverse_incidence(model_final.predict(X_test))
    test_r2_pooled = float(r2_score(y_test_raw, test_pred_raw))
    prov_df, test_r2_filt = per_province_r2(y_test_raw, test_pred_raw, prov_test, included_provinces)
    prov_df.to_csv(os.path.join(OUT_DIR, f'{variant_name}_per_province.csv'), index=False)

    print(f'  {variant_name}  val_R2={val_r2:.4f}  test_R2_pooled={test_r2_pooled:.4f}  test_R2_filt={test_r2_filt:.4f}')
    summary_rows.append({
        'variant': variant_name,
        'val_r2': val_r2,
        'test_r2_pooled': test_r2_pooled,
        'test_r2_mean_per_province': test_r2_filt,
        'n_features': X_trainval.shape[1],
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(OUT_DIR, 'ar_fourier_benchmark.csv'), index=False)

print('\n[DONE] conventional autoregression + Fourier/dynamic-harmonic baseline')
print(summary_df.to_string(index=False))
print('\nWrote ar_fourier_benchmark.csv, ar_only_per_province.csv, ar_fourier_per_province.csv to', OUT_DIR)
