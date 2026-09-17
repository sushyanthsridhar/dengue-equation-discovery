import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingRegressor
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
STATE_DIM = F.STATE_DIM

def _inverse_incidence(y_scaled):
    y_log = y_scaled * STD_INC + MEAN_INC
    return np.expm1(y_log) if F.LOG_INCIDENCE else y_log

def build_pooled_xy(province_data, n_lags):
    (X_parts, y_parts, prov_parts) = ([], [], [])
    for (prov, pd_t) in province_data.items():
        s_obs = pd_t['s_obs']
        if len(s_obs) <= n_lags + 1:
            continue
        X_aug = F._build_aug(s_obs, n_lags)
        X_aug = X_aug[:-1]
        y_next = s_obs[n_lags + 1:, F.INC_COL]
        n = min(len(X_aug), len(y_next))
        if n <= 0:
            continue
        X_parts.append(X_aug[:n])
        y_parts.append(y_next[:n])
        prov_parts.append(np.array([prov] * n))
    if not X_parts:
        return (np.zeros((0, (n_lags + 1) * STATE_DIM), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.array([]))
    return (np.concatenate(X_parts, axis=0), np.concatenate(y_parts, axis=0), np.concatenate(prov_parts, axis=0))

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
(X_train, y_train, prov_train) = build_pooled_xy(train_pd, N_LAGS)
(X_val, y_val, prov_val) = build_pooled_xy(val_pd, N_LAGS)
(X_trainval, y_trainval, prov_trainval) = build_pooled_xy(trainval_pd, N_LAGS)
(X_test, y_test, prov_test) = build_pooled_xy(test_pd, N_LAGS)
y_test_raw = _inverse_incidence(y_test)
rows_summary = []
gb = GradientBoostingRegressor(n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.8, random_state=0, validation_fraction=0.15, n_iter_no_change=20, tol=0.0001)
gb.fit(X_trainval, y_trainval)
gb_pred_test_raw = _inverse_incidence(gb.predict(X_test))
gb_test_r2_pooled = float(r2_score(y_test_raw, gb_pred_test_raw))
(gb_prov_df, gb_test_r2_filt) = per_province_r2(y_test_raw, gb_pred_test_raw, prov_test, included_provinces)
gb_prov_df.to_csv(os.path.join(OUT_DIR, 'gb_benchmark_per_province.csv'), index=False)
rows_summary.append({'variant': 'gradient_boosting', 'test_r2_pooled': gb_test_r2_pooled, 'test_r2_mean_per_province': gb_test_r2_filt})
torch.manual_seed(0)

def to_sequence(X_flat):
    n = X_flat.shape[0]
    blocks = X_flat.reshape(n, N_LAGS + 1, STATE_DIM)
    return blocks[:, ::-1, :].copy()

class LSTMForecaster(nn.Module):

    def __init__(self, input_dim, hidden=48):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        (out, _) = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)
device = 'cpu'
BATCH = 256
MAX_EPOCHS = 150
PATIENCE = 12
Xtr_seq = torch.tensor(to_sequence(X_train), dtype=torch.float32, device=device)
ytr = torch.tensor(y_train, dtype=torch.float32, device=device)
Xva_seq = torch.tensor(to_sequence(X_val), dtype=torch.float32, device=device)
yva_np = y_val
model_lstm = LSTMForecaster(STATE_DIM).to(device)
opt = torch.optim.Adam(model_lstm.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
(best_val_r2, best_epoch, bad_epochs) = (-1000000000.0, 0, 0)
n_train = Xtr_seq.shape[0]
for epoch in range(1, MAX_EPOCHS + 1):
    model_lstm.train()
    perm = torch.randperm(n_train)
    total_loss = 0.0
    for i in range(0, n_train, BATCH):
        idx = perm[i:i + BATCH]
        opt.zero_grad()
        pred = model_lstm(Xtr_seq[idx])
        loss = loss_fn(pred, ytr[idx])
        loss.backward()
        opt.step()
        total_loss += float(loss) * len(idx)
    model_lstm.eval()
    with torch.no_grad():
        val_pred = model_lstm(Xva_seq).cpu().numpy()
    val_r2 = r2_score(yva_np, val_pred) if len(yva_np) else float('-inf')
    if val_r2 > best_val_r2:
        (best_val_r2, best_epoch, bad_epochs) = (val_r2, epoch, 0)
    else:
        bad_epochs += 1
    if epoch == 1 or epoch % 10 == 0:
        pass
    if bad_epochs >= PATIENCE:
        break
torch.manual_seed(0)
model_lstm_final = LSTMForecaster(STATE_DIM).to(device)
opt2 = torch.optim.Adam(model_lstm_final.parameters(), lr=0.001)
Xtv_seq = torch.tensor(to_sequence(X_trainval), dtype=torch.float32, device=device)
ytv = torch.tensor(y_trainval, dtype=torch.float32, device=device)
n_tv = Xtv_seq.shape[0]
for epoch in range(1, max(best_epoch, 1) + 1):
    model_lstm_final.train()
    perm = torch.randperm(n_tv)
    for i in range(0, n_tv, BATCH):
        idx = perm[i:i + BATCH]
        opt2.zero_grad()
        pred = model_lstm_final(Xtv_seq[idx])
        loss = loss_fn(pred, ytv[idx])
        loss.backward()
        opt2.step()
model_lstm_final.eval()
with torch.no_grad():
    Xte_seq = torch.tensor(to_sequence(X_test), dtype=torch.float32, device=device)
    lstm_pred_test_scaled = model_lstm_final(Xte_seq).cpu().numpy()
lstm_pred_test_raw = _inverse_incidence(lstm_pred_test_scaled)
lstm_test_r2_pooled = float(r2_score(y_test_raw, lstm_pred_test_raw))
(lstm_prov_df, lstm_test_r2_filt) = per_province_r2(y_test_raw, lstm_pred_test_raw, prov_test, included_provinces)
lstm_prov_df.to_csv(os.path.join(OUT_DIR, 'lstm_benchmark_per_province.csv'), index=False)
rows_summary.append({'variant': 'lstm', 'test_r2_pooled': lstm_test_r2_pooled, 'test_r2_mean_per_province': lstm_test_r2_filt})
pd.DataFrame(rows_summary).to_csv(os.path.join(OUT_DIR, 'lstm_gb_benchmark.csv'), index=False)
fixed_model_train = F.run_hierarchical_em(train_pd, N_LAGS, [], [], label='fixed_library_sindy_pooled_TRAIN', use_cluster=False, use_province=False)
fixed_model_train['q_inc_by_prov'] = F.estimate_province_q_inc(fixed_model_train, train_pd)
(fixed_val_r2, _, _) = F.evaluate_on_split(val_pd, fixed_model_train)
fixed_model_final = F.run_hierarchical_em(trainval_pd, N_LAGS, [], [], label='fixed_library_sindy_pooled_FINAL', use_cluster=False, use_province=False)
fixed_model_final['q_inc_by_prov'] = F.estimate_province_q_inc(fixed_model_final, trainval_pd)
(fixed_test_r2, fixed_test_df, _) = F.evaluate_on_split(test_pd, fixed_model_final)
fixed_test_incl = fixed_test_df[fixed_test_df['province'].isin(included_provinces)]
fixed_test_r2_filt = float(fixed_test_incl['r2'].mean()) if len(fixed_test_incl) else float('nan')
fixed_test_df.to_csv(os.path.join(OUT_DIR, 'fixed_library_sindy_per_province.csv'), index=False)
(active_terms, pruned_terms) = F.get_active_and_pruned(fixed_model_final)
eq_rows = [{'term_name': nm, 'coef_incidence': coef, 'active': True} for (nm, coef) in active_terms]
eq_rows += [{'term_name': nm, 'coef_incidence': 0.0, 'active': False} for nm in pruned_terms]
pd.DataFrame(eq_rows).sort_values('coef_incidence', key=lambda s: s.abs(), ascending=False).to_csv(os.path.join(OUT_DIR, 'fixed_library_sindy_equation.csv'), index=False)
pd.DataFrame([{'variant': 'fixed_library_sindy_pooled', 'val_r2': fixed_val_r2, 'test_r2': fixed_test_r2, 'test_r2_filt': fixed_test_r2_filt, 'n_active_terms': len(active_terms)}]).to_csv(os.path.join(OUT_DIR, 'fixed_library_sindy_pooled.csv'), index=False)
combined_rows = list(rows_summary)
combined_rows.append({'variant': 'fixed_library_sindy_pooled', 'test_r2_pooled': fixed_test_r2, 'test_r2_mean_per_province': fixed_test_r2_filt})
main_table_path = os.path.join(OUT_DIR, 'main_benchmark_table.csv')
if os.path.exists(main_table_path):
    existing = pd.read_csv(main_table_path)
    combined_df = pd.concat([existing, pd.DataFrame(combined_rows)], ignore_index=True)
else:
    combined_df = pd.DataFrame(combined_rows)
combined_df.to_csv(os.path.join(OUT_DIR, 'benchmark_summary_with_lstm_gb_sindy.csv'), index=False)
