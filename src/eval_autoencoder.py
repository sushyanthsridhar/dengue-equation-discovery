import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from utils import load_data
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(HERE, 'extended_input_normalized.csv')
CFG_PATH = os.path.join(HERE, 'best_config_autoencoder.txt')
MODELS_DIR = os.path.join(HERE, 'models')
PLOTS_DIR = os.path.join(HERE, 'plots')
LATENT_DIR = os.path.join(HERE, 'latents')
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(LATENT_DIR, exist_ok=True)
TRAIN_YEARS = list(range(2015, 2019))
TEST_YEARS = [2022, 2023]

def load_config(path):
    cfg = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('val_mse') or line.startswith('val_r2'):
                continue
            (k, v) = line.split('=', 1)
            cfg[k.strip()] = v.strip()
    return cfg
ACTS = {'relu': nn.ReLU, 'leakyrelu': nn.LeakyReLU, 'elu': nn.ELU}

class WeeklyAutoencoder(nn.Module):

    def __init__(self, input_dim, latent_dim, hidden_dims, dropout, activation, use_batchnorm):
        super().__init__()
        act = ACTS[activation]
        enc_dims = [input_dim] + hidden_dims + [latent_dim]
        enc_layers = []
        for i in range(len(enc_dims) - 1):
            enc_layers.append(nn.Linear(enc_dims[i], enc_dims[i + 1]))
            if i < len(enc_dims) - 2:
                if use_batchnorm:
                    enc_layers.append(nn.BatchNorm1d(enc_dims[i + 1]))
                enc_layers.append(act())
                if dropout > 0.0:
                    enc_layers.append(nn.Dropout(dropout))
        self.encoder = nn.Sequential(*enc_layers)
        dec_dims = list(reversed(enc_dims))
        dec_layers = []
        for i in range(len(dec_dims) - 1):
            dec_layers.append(nn.Linear(dec_dims[i], dec_dims[i + 1]))
            if i < len(dec_dims) - 2:
                if use_batchnorm:
                    dec_layers.append(nn.BatchNorm1d(dec_dims[i + 1]))
                dec_layers.append(act())
                if dropout > 0.0:
                    dec_layers.append(nn.Dropout(dropout))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x):
        return self.encoder(x)
cfg = load_config(CFG_PATH)
LATENT_DIM = int(cfg['latent_dim'])
HIDDEN_DIMS = eval(cfg['hidden_dims'])
DROPOUT = float(cfg['dropout'])
LR = float(cfg['lr'])
ACTIVATION = cfg['activation']
USE_BATCHNORM = cfg['use_batchnorm'] == 'True'
(raw, INPUT_FEATURES) = load_data(DATA_CSV, TRAIN_YEARS)
INPUT_DIM = len(INPUT_FEATURES)
test_mask = raw['year'].isin(TEST_YEARS)
X_test = torch.tensor(raw.loc[test_mask, INPUT_FEATURES].values.astype(np.float32))
meta_test = raw.loc[test_mask, ['province', 'year', 'week']].reset_index(drop=True)
INCIDENCE_WEIGHT = 5.0
feat_weights = torch.ones(INPUT_DIM)
for (i, f) in enumerate(INPUT_FEATURES):
    if 'incidence' in f or 'inc_momentum' in f:
        feat_weights[i] = INCIDENCE_WEIGHT
model = WeeklyAutoencoder(INPUT_DIM, LATENT_DIM, HIDDEN_DIMS, DROPOUT, ACTIVATION, USE_BATCHNORM)
model.load_state_dict(torch.load(os.path.join(MODELS_DIR, f'best_model_dim{LATENT_DIM}.pt'), weights_only=True))
model.eval()
with torch.no_grad():
    X_hat = model(X_test)
    Z_test = model.encode(X_test).detach().numpy()
mse = nn.functional.mse_loss(X_hat, X_test).item()
ss_res = ((X_test - X_hat) ** 2).sum()
ss_tot = ((X_test - X_test.mean()) ** 2).sum()
r2 = (1.0 - ss_res / (ss_tot + 1e-08)).item()
z_cols = [f'z{i + 1}' for i in range(LATENT_DIM)]
latent_df = pd.DataFrame(Z_test, columns=z_cols)
latent_df['province'] = meta_test['province'].values
latent_df['year'] = meta_test['year'].values
latent_df['week'] = meta_test['week'].values
latent_df['split'] = 'test'
latent_path = os.path.join(LATENT_DIR, f'latent_test_dim{LATENT_DIM}.csv')
latent_df.to_csv(latent_path, index=False)
X_test_np = X_test.numpy()
X_hat_np = X_hat.detach().numpy()
per_feature_r2 = []
for i in range(INPUT_DIM):
    ss_r = ((X_test_np[:, i] - X_hat_np[:, i]) ** 2).sum()
    ss_t = ((X_test_np[:, i] - X_test_np[:, i].mean()) ** 2).sum()
    per_feature_r2.append(1.0 - ss_r / (ss_t + 1e-08))
feat_r2_df = pd.DataFrame({'feature': INPUT_FEATURES, 'r2': per_feature_r2}).sort_values('r2')
(fig, ax) = plt.subplots(figsize=(12, 5))
colors = ['#d62728' if r < 0.9 else '#2ca02c' for r in feat_r2_df['r2']]
ax.barh(feat_r2_df['feature'], feat_r2_df['r2'], color=colors)
ax.axvline(0.9, color='black', linestyle='--', linewidth=0.8, label='R2=0.9')
ax.set_xlabel('R2')
ax.set_title(f'Per-feature reconstruction R2 on test set  (overall R2={r2:.4f})')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'test_per_feature_r2.png'), dpi=150)
plt.close()
