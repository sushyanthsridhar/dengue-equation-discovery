import pandas as pd
import forecast as fc
SELECTED_Z = [4, 6, 7, 9]
COV_COLS = ['avg_temp', 'avg_rainfall', 'avg_humidity', 'nino34_anom', 'soi_index', 'tna_sst_anom', 'solar_radiation', 'wind_speed', 'dew_point_temp', 'neighbor_incidence_lag0', 'consecutive_dry_weeks', 'consecutive_wet_weeks', 'rainy_season_phase', 'school_calendar_active']
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
        for (rank, (name, val)) in enumerate(corrs.head(3).items(), start=1):
            rows.append({'z': zcol, 'rank': rank, 'covariate': name, 'pearson_r': round(float(val), 4)})
    df = pd.DataFrame(rows)
    df.to_csv('results/latent_covariate_associations.csv', index=False)
