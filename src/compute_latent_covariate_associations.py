"""
Correlation of the LLM-selected latent dimensions against real named covariates,
computed on training years only (no leakage).

WHY THIS SCRIPT EXISTS:
forecast.py's compute_latent_covariate_meaning() already does exactly this
computation internally, once per stability run, to ground the LLM prompt (see
its docstring and Section 1 of results_report.txt). It returns only the top-2
correlations per z-dimension as an inline string for the prompt, and never
writes them anywhere -- there is no persisted table of these numbers. This
script reuses forecast.py's own correlation logic unmodified and just saves
the result to a CSV, restricted to the four latent dimensions (z4, z6, z7, z9)
that actually survived into the final discovered equation across the five
stability runs (results/global_equation_run1.csv through run5.csv).

WHAT IT DOES NOT DO:
It does not fit, retrain, or touch the model in any way. It is a pure
correlation computation on the existing autoencoder latent CSV and the raw
covariate CSV, both already produced upstream of this script.

Usage:
    python3 compute_latent_covariate_associations.py

Writes:
    results/latent_covariate_associations.csv -- one row per (z, covariate,
        Pearson r), top 3 covariates by |r| for each of z4, z6, z7, z9.
"""
import os

import pandas as pd

import forecast as fc  # noqa: E402  (importing runs forecast.py's module-level data/setup code)

SELECTED_Z = [4, 6, 7, 9]  # latent dims that appear as active LLM-proposed terms in at least one of the 5 stability runs
COV_COLS = ['avg_temp', 'avg_rainfall', 'avg_humidity', 'nino34_anom', 'soi_index', 'tna_sst_anom',
            'solar_radiation', 'wind_speed', 'dew_point_temp', 'neighbor_incidence_lag0',
            'consecutive_dry_weeks', 'consecutive_wet_weeks', 'rainy_season_phase', 'school_calendar_active']

if __name__ == '__main__':
    raw = pd.read_csv(fc.DATA_CSV, usecols=['province', 'year', 'week'] + COV_COLS)
    raw = raw[raw['year'].isin(fc.TRAIN_YEARS)]
    z_cols = [f'z{i}' for i in range(1, fc.LATENT_DIM + 1)]
    lat = pd.read_csv(fc.LATENT_CSV, usecols=['province', 'year', 'week'] + z_cols)
    lat = lat[lat['year'].isin(fc.TRAIN_YEARS)]
    merged = pd.merge(lat, raw, on=['province', 'year', 'week'], how='inner')

    rows = []
    for zi in SELECTED_Z:
        zcol = f'z{zi}'
        corrs = merged[COV_COLS].corrwith(merged[zcol]).dropna()
        corrs = corrs.reindex(corrs.abs().sort_values(ascending=False).index)
        for rank, (name, val) in enumerate(corrs.head(3).items(), start=1):
            rows.append({'z': zcol, 'rank': rank, 'covariate': name, 'pearson_r': round(float(val), 4)})

    df = pd.DataFrame(rows)
    OUT_PATH = os.path.join(fc.HERE, 'results', 'latent_covariate_associations.csv')  # fc.HERE resolves to the repo root regardless of cwd
    df.to_csv(OUT_PATH, index=False)
    print(df.to_string(index=False))
    print(f"\nWrote {OUT_PATH}")
