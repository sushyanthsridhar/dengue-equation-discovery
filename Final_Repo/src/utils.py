import numpy as np
import pandas as pd

def load_data(data_csv: str, train_years: list):
    raw = pd.read_csv(data_csv)
    raw = raw.sort_values(['province', 'year', 'week']).reset_index(drop=True)
    ID_COLS = ['province', 'year', 'month', 'week']
    feature_cols = [c for c in raw.columns if c not in ID_COLS]
    raw['time_sin'] = np.sin(2.0 * np.pi * raw['week'] / 52.0)
    raw['time_cos'] = np.cos(2.0 * np.pi * raw['week'] / 52.0)
    raw['incidence_lag1'] = raw.groupby('province')['incidence'].shift(1)
    raw['incidence_lag2'] = raw.groupby('province')['incidence'].shift(2)
    raw['inc_momentum'] = raw.groupby('province')['incidence'].diff()
    input_features = feature_cols + ['time_sin', 'time_cos', 'incidence_lag1', 'incidence_lag2', 'inc_momentum']
    train_mask = raw['year'].isin(train_years)
    col_means = raw.loc[train_mask, input_features].mean()
    raw[input_features] = raw[input_features].fillna(col_means)
    return (raw, input_features)
