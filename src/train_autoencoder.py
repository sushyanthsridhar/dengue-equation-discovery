import os
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from utils import load_data
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(HERE, 'extended_input_normalized.csv')
CFG_PATH = os.path.join(HERE, 'best_config_autoencoder.txt')
MODELS_DIR = os.path.join(HERE, 'models')
LATENT_DIR = os.path.join(HERE, 'latents')
PLOTS_DIR = os.path.join(HERE, 'plots')
for d in [MODELS_DIR, LATENT_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)
TRAIN_YEARS = list(range(2015, 2019))
VAL_YEARS = [2019]
EPOCHS = 500
PATIENCE = 80

def load_config(path):
    cfg = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('val_mse') or line.startswith('val_r2'):
                continue
            k, v = line.split('=', 1)
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
WEIGHT_DECAY = float(cfg['weight_decay'])
BATCH_SIZE = int(cfg['batch_size'])
ACTIVATION = cfg['activation']
USE_BATCHNORM = cfg['use_batchnorm'] == 'True'
raw, INPUT_FEATURES = load_data(DATA_CSV, TRAIN_YEARS)
INPUT_DIM = len(INPUT_FEATURES)
train_mask = raw['year'].isin(TRAIN_YEARS)
val_mask = raw['year'].isin(VAL_YEARS)
trainval_mask = train_mask | val_mask
X_train = torch.tensor(raw.loc[train_mask, INPUT_FEATURES].values.astype(np.float32))
X_trainval = torch.tensor(raw.loc[trainval_mask, INPUT_FEATURES].values.astype(np.float32))
meta_trainval = raw.loc[trainval_mask, ['province', 'year', 'week']].reset_index(drop=True)
meta_trainval['split'] = meta_trainval['year'].apply(lambda y: 'train' if y in TRAIN_YEARS else 'val')
INCIDENCE_WEIGHT = 5.0
feat_weights = torch.ones(INPUT_DIM)
for i, f in enumerate(INPUT_FEATURES):
    if 'incidence' in f or 'inc_momentum' in f:
        feat_weights[i] = INCIDENCE_WEIGHT

def weighted_mse(x_hat, x):
    return (feat_weights * (x_hat - x) ** 2).mean()
model = WeeklyAutoencoder(INPUT_DIM, LATENT_DIM, HIDDEN_DIMS, DROPOUT, ACTIVATION, USE_BATCHNORM)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-06)
trainval_loader = DataLoader(TensorDataset(X_trainval, X_trainval), batch_size=BATCH_SIZE, shuffle=True)
model_path = os.path.join(MODELS_DIR, f'best_model_dim{LATENT_DIM}.pt')
train_losses = []
best_train_loss = float('inf')
patience_count = 0
n_params = sum((p.numel() for p in model.parameters()))
for epoch in range(1, EPOCHS + 1):
    model.train()
    batch_losses = []
    for xb, _ in trainval_loader:
        optimizer.zero_grad()
        loss = weighted_mse(model(xb), xb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        batch_losses.append(loss.item())
    train_loss = float(np.mean(batch_losses))
    scheduler.step(train_loss)
    train_losses.append(train_loss)
    if train_loss < best_train_loss:
        best_train_loss = train_loss
        patience_count = 0
        torch.save(model.state_dict(), model_path)
    else:
        patience_count += 1
    if epoch % 50 == 0 or epoch == 1:
        cur_lr = optimizer.param_groups[0]['lr']
    if patience_count >= PATIENCE:
        break
model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()
with torch.no_grad():
    final_mse = nn.functional.mse_loss(model(X_trainval), X_trainval).item()
    ss_res = ((X_trainval - model(X_trainval)) ** 2).sum()
    ss_tot = ((X_trainval - X_trainval.mean()) ** 2).sum()
    final_r2 = (1.0 - ss_res / (ss_tot + 1e-08)).item()
with torch.no_grad():
    Z = model.encode(X_trainval).detach().numpy()
z_cols = [f'z{i + 1}' for i in range(LATENT_DIM)]
latent_df = pd.DataFrame(Z, columns=z_cols)
latent_df['province'] = meta_trainval['province'].values
latent_df['year'] = meta_trainval['year'].values
latent_df['week'] = meta_trainval['week'].values
latent_df['split'] = meta_trainval['split'].values
latent_path = os.path.join(LATENT_DIR, f'latent_dim{LATENT_DIM}.csv')
latent_df.to_csv(latent_path, index=False)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(train_losses, label='Train+Val')
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE Loss')
ax.set_title(f'Autoencoder  latent_dim={LATENT_DIM}  fit_r2={final_r2:.4f}')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, f'training_loss_dim{LATENT_DIM}.png'), dpi=150)
plt.close()
plot_cols = z_cols[:min(LATENT_DIM, 6)]
pairs = list(itertools.combinations(plot_cols, 2))
if pairs:
    ncols = min(3, len(pairs))
    nrows = (len(pairs) + ncols - 1) // ncols
    years = sorted(latent_df['year'].unique())
    yc = {y: plt.cm.plasma(i / max(len(years) - 1, 1)) for i, y in enumerate(years)}
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten() if len(pairs) > 1 else [axes]
    for idx, (za, zb) in enumerate(pairs):
        ax = axes[idx]
        for yr in years:
            m = latent_df['year'] == yr
            ax.scatter(latent_df.loc[m, za], latent_df.loc[m, zb], color=yc[yr], s=4, alpha=0.4, label=str(yr))
        ax.set_xlabel(za, fontsize=8)
        ax.set_ylabel(zb, fontsize=8)
        ax.set_title(f'{za} vs {zb}', fontsize=9)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(title='Year', fontsize=5, markerscale=2)
    for idx in range(len(pairs), len(axes)):
        axes[idx].set_visible(False)
    plt.suptitle(f'Autoencoder  latent_dim={LATENT_DIM}  fit_r2={final_r2:.4f}', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'latent_pairs_dim{LATENT_DIM}.png'), dpi=150)
    plt.close()
