"""
Paper figure: global equation performance -- predicted vs. actual incidence
over time.

This is not a referee-report item; it produces the figure Sushi needs for
the results section showing how well the discovered, hierarchically fit
equation tracks real dengue incidence, aggregated across all 28 provinces,
across the whole modeled record (2015-2019 fit period, then the 2022-2023
held-out test period).

WHAT IT DOES:
  1. Runs forecast.py's fit_winning_model() to produce the actual final
     production model: the saved winning FunSearch structure plus a fresh
     live LLM term-proposal round loop (MAX_LLM_ROUNDS rounds, real Ollama
     calls -- same as every real run of this pipeline; not cached, so this
     takes a few minutes and each run's exact LLM-proposed terms can vary
     slightly, same caveat noted throughout results_report.txt).
  2. Refits that structure on TRAIN+VAL (2015-2019) -- this is the "fit"
     segment of the plot, i.e. in-sample.
  3. Evaluates the same fitted model, one-step-ahead and open-loop, on
     TRAIN+VAL (in-sample) and on TEST (2022-2023, genuinely held out),
     using forecast.py's own forecast_sequential -- the exact same
     sequential rollout the paper's accuracy numbers come from.
  4. Aligns every prediction back to its (year, week) and province, then
     averages actual and predicted incidence across all provinces for each
     (year, week) to get one national aggregate curve per split. This is
     an unweighted mean across provinces, not population-weighted --
     called out explicitly in the CSV and the plot caption text.
  5. Plots two panels: the full 2015-2019 + 2022-2023 timeline, and a
     zoomed-in view of just the 2022-2023 test period (the genuinely
     out-of-sample evaluation the paper's headline R2 numbers describe).
     Both panels show actual vs. predicted mean incidence, with the R2 for
     each split annotated.

This script does not touch the ablation/benchmark experiments already in
results_report.txt -- it is purely a visualization of the already-reported
final model's fit and forecast quality, at the aggregate level.

Usage:
    python3 plot_global_equation_performance.py

Writes:
    figures/global_equation_performance.png   -- the two-panel figure
    results/global_equation_performance_per_province.csv   -- one row per
        (province, split, year, week): y_true, y_hat
    results/global_equation_performance_national_mean.csv  -- one row per
        (split, year, week): mean y_true, mean y_hat across provinces
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root: this script now lives one level down, in src/
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

print('[plot-global-eq] imported', F.__name__)

FIG_DIR = os.path.join(HERE, 'figures')
RES_DIR = os.path.join(HERE, 'results')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

MEAN_INC = float(F.SCALER.mean_[F.INC_COL])
STD_INC = float(F.SCALER.scale_[F.INC_COL])


def _inverse_incidence(y_scaled):
    y_log = y_scaled * STD_INC + MEAN_INC
    return np.expm1(y_log) if F.LOG_INCIDENCE else y_log


def evaluate_with_time(province_data, model, split_name):
    """Same math as forecast.py's evaluate_on_split, but also returns the
    (year, week) each prediction belongs to, so predictions can be plotted
    against real time rather than just scored."""
    n_lags, fs_terms, llm_formulas = model['n_lags'], model['fs_terms'], model['llm_formulas']
    q_inc_by_prov = model.get('q_inc_by_prov', {})
    rows = []
    for prov, pd_t in province_data.items():
        cid = pd_t['cluster']
        s_test = pd_t['s_obs']
        weeks = pd_t['weeks']
        years = pd_t['years']
        T = len(s_test)
        if T <= n_lags + 1:
            continue
        beta_p = model['g'] + model['h'].get(cid, 0) + model['u'].get(prov, 0)
        q_diag_p = model['q_diag'].copy()
        if prov in q_inc_by_prov:
            q_diag_p[F.INC_COL] = q_inc_by_prov[prov]
        yhat_sc = F.forecast_sequential(s_test, weeks, beta_p, q_diag_p, model['r_diag'], n_lags, fs_terms, llm_formulas)
        n_pred = len(yhat_sc)
        y_true_sc = s_test[n_lags + 1:n_lags + 1 + n_pred, F.INC_COL]
        weeks_used = weeks[n_lags + 1:n_lags + 1 + n_pred]
        years_used = years[n_lags + 1:n_lags + 1 + n_pred]
        yhat_cnt = _inverse_incidence(yhat_sc)
        ytrue_cnt = _inverse_incidence(y_true_sc)
        for wk, yr, yt, yp in zip(weeks_used, years_used, ytrue_cnt, yhat_cnt):
            rows.append({'split': split_name, 'province': prov, 'cluster_id': cid,
                         'year': int(yr), 'week': int(wk), 'y_true': float(yt), 'y_hat': float(yp)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. Build the same data splits forecast.py uses, and fit the actual
#    production model (fresh LLM rounds -- see module docstring).
# ---------------------------------------------------------------------------
train_pd = F._build_province_dict(F.latent, F.TRAIN_YEARS)
val_pd = F._build_province_dict(F.latent, F.VAL_YEARS)
trainval_pd = F._build_province_dict(F.latent, F.TRAINVAL_YEARS)
test_pd = F._build_province_dict(F.latent, F.TEST_YEARS)

print('[plot-global-eq] fitting the winning model (this reruns the live LLM round loop -- a few minutes) ...')
model_final = F.fit_winning_model(train_pd, val_pd, trainval_pd)
print(f"[plot-global-eq] final model fit, {len(model_final.get('llm_formulas_used', []))} LLM terms survived")

# ---------------------------------------------------------------------------
# 2. Evaluate with time labels on both the in-sample fit period and the
#    genuinely held-out test period.
# ---------------------------------------------------------------------------
fit_df = evaluate_with_time(trainval_pd, model_final, 'fit_2015_2019')
test_df = evaluate_with_time(test_pd, model_final, 'test_2022_2023')
per_province_df = pd.concat([fit_df, test_df], ignore_index=True)
per_province_df.to_csv(os.path.join(RES_DIR, 'global_equation_performance_per_province.csv'), index=False)
print(f'[plot-global-eq] wrote per-province time series ({len(per_province_df)} rows)')

r2_fit = float(r2_score(fit_df['y_true'], fit_df['y_hat']))
r2_test = float(r2_score(test_df['y_true'], test_df['y_hat']))
print(f'[plot-global-eq] pooled R2, fit (in-sample, 2015-2019)  = {r2_fit:.4f}')
print(f'[plot-global-eq] pooled R2, test (out-of-sample, 2022-2023) = {r2_test:.4f}')

# ---------------------------------------------------------------------------
# 3. Aggregate to a national mean curve per (split, year, week). Unweighted
#    mean across provinces -- not population-weighted.
# ---------------------------------------------------------------------------
national_df = (per_province_df.groupby(['split', 'year', 'week'], as_index=False)
               .agg(y_true=('y_true', 'mean'), y_hat=('y_hat', 'mean'), n_provinces=('province', 'nunique')))
national_df['time_value'] = national_df['year'] + (national_df['week'] - 1) / 52.0
national_df = national_df.sort_values('time_value').reset_index(drop=True)
national_df.to_csv(os.path.join(RES_DIR, 'global_equation_performance_national_mean.csv'), index=False)
print(f'[plot-global-eq] wrote national mean time series ({len(national_df)} rows)')

# ---------------------------------------------------------------------------
# 4. Plot: full timeline (top) + zoomed test period (bottom).
# ---------------------------------------------------------------------------
fig, (ax_full, ax_test) = plt.subplots(2, 1, figsize=(11, 8))

fit_nat = national_df[national_df['split'] == 'fit_2015_2019']
test_nat = national_df[national_df['split'] == 'test_2022_2023']

ax_full.plot(fit_nat['time_value'], fit_nat['y_true'], color='#333333', lw=1.3, label='Actual (national mean)')
ax_full.plot(fit_nat['time_value'], fit_nat['y_hat'], color='#1f77b4', lw=1.3, ls='--', label='Global equation (fit, in-sample)')
ax_full.plot(test_nat['time_value'], test_nat['y_true'], color='#333333', lw=1.3)
ax_full.plot(test_nat['time_value'], test_nat['y_hat'], color='#d62728', lw=1.5, ls='--', label='Global equation (test, out-of-sample)')
ax_full.axvspan(test_nat['time_value'].min(), test_nat['time_value'].max(), color='#d62728', alpha=0.06)
ax_full.set_title(f'Global equation performance, full modeled record  (fit R2={r2_fit:.3f}, test R2={r2_test:.3f})')
ax_full.set_ylabel('Mean incidence across provinces')
ax_full.legend(loc='upper left', fontsize=9)
ax_full.set_xlabel('Year')

ax_test.plot(test_nat['time_value'], test_nat['y_true'], color='#333333', lw=1.6, marker='o', ms=3, label='Actual')
ax_test.plot(test_nat['time_value'], test_nat['y_hat'], color='#d62728', lw=1.6, ls='--', marker='o', ms=3, label='Global equation (predicted)')
ax_test.set_title(f'Held-out test period, 2022-2023 only  (R2={r2_test:.3f})')
ax_test.set_ylabel('Mean incidence across provinces')
ax_test.set_xlabel('Year')
ax_test.legend(loc='upper left', fontsize=9)

fig.tight_layout()
out_path = os.path.join(FIG_DIR, 'global_equation_performance.png')
fig.savefig(out_path, dpi=200)
print('\n[DONE] wrote', out_path)
print('Wrote global_equation_performance_per_province.csv and global_equation_performance_national_mean.csv to', RES_DIR)
