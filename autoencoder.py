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
import optuna
from optuna.samplers import TPESampler
from utils import load_data
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(HERE, 'extended_input_normalized.csv')
MODELS_DIR = os.path.join(HERE, 'models')
LATENT_DIR = os.path.join(HERE, 'latents')
PLOTS_DIR = os.path.join(HERE, 'plots')
for d in [MODELS_DIR, LATENT_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)
TRAIN_YEARS = list(range(2015, 2019))
VAL_YEARS = [2019]
TEST_YEARS = [2022, 2023]
N_TRIALS = 100
EPOCHS = 200
PATIENCE = 40
FINAL_EPOCHS = 500
FINAL_PATIENCE = 80
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

def r2(y, yhat):
    ss_res = ((y - yhat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return (1.0 - ss_res / (ss_tot + 1e-08)).item()
raw, INPUT_FEATURES = load_data(DATA_CSV, TRAIN_YEARS)
INPUT_DIM = len(INPUT_FEATURES)
train_mask = raw['year'].isin(TRAIN_YEARS)
val_mask = raw['year'].isin(VAL_YEARS)
trainval_mask = train_mask | val_mask
test_mask = raw['year'].isin(TEST_YEARS)
X_train = torch.tensor(raw.loc[train_mask, INPUT_FEATURES].values.astype(np.float32))
X_val = torch.tensor(raw.loc[val_mask, INPUT_FEATURES].values.astype(np.float32))
X_trainval = torch.tensor(raw.loc[trainval_mask, INPUT_FEATURES].values.astype(np.float32))
X_test = torch.tensor(raw.loc[test_mask, INPUT_FEATURES].values.astype(np.float32))
meta_trainval = raw.loc[trainval_mask, ['province', 'year', 'week']].reset_index(drop=True)
meta_trainval['split'] = meta_trainval['year'].apply(lambda y: 'train' if y in TRAIN_YEARS else 'val')

def objective(trial):
    latent_dim = trial.suggest_int('latent_dim', 3, 10)
    hidden_choice = trial.suggest_categorical('hidden_dims', ['[256,128]', '[512,256]', '[512,256,128]', '[256,256]'])
    hidden_dims = eval(hidden_choice)
    dropout = trial.suggest_float('dropout', 0.0, 0.4)
    lr = trial.suggest_float('lr', 0.0005, 0.005, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-06, 0.001, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    activation = trial.suggest_categorical('activation', ['relu', 'leakyrelu', 'elu'])
    use_batchnorm = trial.suggest_categorical('use_batchnorm', [True, False])
    model = WeeklyAutoencoder(INPUT_DIM, latent_dim, hidden_dims, dropout, activation, use_batchnorm)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-06)
    train_loader = DataLoader(TensorDataset(X_train, X_train), batch_size=batch_size, shuffle=True)
    best_val_loss = float('inf')
    patience_count = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, _ in train_loader:
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(xb), xb)
            if torch.isnan(loss):
                return float('inf')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = nn.functional.mse_loss(model(X_val), X_val).item()
        if torch.isnan(torch.tensor(val_loss)):
            return float('inf')
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
        else:
            patience_count += 1
        if patience_count >= PATIENCE:
            break
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    model.eval()
    with torch.no_grad():
        val_r2 = r2(X_val, model(X_val))
    return best_val_loss
sampler = TPESampler(seed=42)
pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=30)
study = optuna.create_study(direction='minimize', sampler=sampler, pruner=pruner)
optuna.logging.set_verbosity(optuna.logging.WARNING)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
best = study.best_trial
LATENT_DIM = best.params['latent_dim']
HIDDEN_DIMS = eval(best.params['hidden_dims'])
DROPOUT = best.params['dropout']
LR = best.params['lr']
WEIGHT_DECAY = best.params['weight_decay']
BATCH_SIZE = best.params['batch_size']
ACTIVATION = best.params['activation']
USE_BATCHNORM = best.params['use_batchnorm']
for k, v in best.params.items():
    pass
config_path = os.path.join(HERE, 'best_config_autoencoder.txt')
with open(config_path, 'w') as f:
    f.write(f'val_mse={best.value:.6f}\n')
    for k, v in best.params.items():
        f.write(f'{k}={v}\n')
final_model = WeeklyAutoencoder(INPUT_DIM, LATENT_DIM, HIDDEN_DIMS, DROPOUT, ACTIVATION, USE_BATCHNORM)
optimizer = torch.optim.AdamW(final_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-06)
trainval_loader = DataLoader(TensorDataset(X_trainval, X_trainval), batch_size=BATCH_SIZE, shuffle=True)
model_path = os.path.join(MODELS_DIR, f'best_model_dim{LATENT_DIM}.pt')
train_losses = []
best_val_loss = float('inf')
patience_count = 0
n_params = sum((p.numel() for p in final_model.parameters()))
for epoch in range(1, FINAL_EPOCHS + 1):
    final_model.train()
    batch_losses = []
    for xb, _ in trainval_loader:
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(final_model(xb), xb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=1.0)
        optimizer.step()
        batch_losses.append(loss.item())
    train_loss = float(np.mean(batch_losses))
    scheduler.step(train_loss)
    train_losses.append(train_loss)
    if train_loss < best_val_loss:
        best_val_loss = train_loss
        patience_count = 0
        torch.save(final_model.state_dict(), model_path)
    else:
        patience_count += 1
    if epoch % 50 == 0 or epoch == 1:
        cur_lr = optimizer.param_groups[0]['lr']
    if patience_count >= FINAL_PATIENCE:
        break
final_model.load_state_dict(torch.load(model_path, weights_only=True))
final_model.eval()
with torch.no_grad():
    trainval_r2 = r2(X_trainval, final_model(X_trainval))
    test_r2 = r2(X_test, final_model(X_test))
    test_mse = nn.functional.mse_loss(final_model(X_test), X_test).item()
with torch.no_grad():
    Z = final_model.encode(X_trainval).detach().numpy()
z_cols = [f'z{i + 1}' for i in range(LATENT_DIM)]
latent_df = pd.DataFrame(Z, columns=z_cols)
latent_df['province'] = meta_trainval['province'].values
latent_df['year'] = meta_trainval['year'].values
latent_df['week'] = meta_trainval['week'].values
latent_df['split'] = meta_trainval['split'].values
latent_path = os.path.join(LATENT_DIR, f'latent_dim{LATENT_DIM}.csv')
latent_df.to_csv(latent_path, index=False)
X_test_np = X_test.numpy()
with torch.no_grad():
    X_hat_np = final_model(X_test).numpy()
per_feature_r2 = []
for i in range(INPUT_DIM):
    ss_r = ((X_test_np[:, i] - X_hat_np[:, i]) ** 2).sum()
    ss_t = ((X_test_np[:, i] - X_test_np[:, i].mean()) ** 2).sum()
    per_feature_r2.append(1.0 - ss_r / (ss_t + 1e-08))
feat_r2_df = pd.DataFrame({'feature': INPUT_FEATURES, 'r2': per_feature_r2}).sort_values('r2')
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(train_losses, label='Train+Val')
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE Loss')
ax.set_title(f'Final Autoencoder  latent_dim={LATENT_DIM}  test_r2={test_r2:.4f}')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, f'training_loss_dim{LATENT_DIM}.png'), dpi=150)
plt.close()
fig, ax = plt.subplots(figsize=(12, 5))
colors = ['#d62728' if r < 0.9 else '#2ca02c' for r in feat_r2_df['r2']]
ax.barh(feat_r2_df['feature'], feat_r2_df['r2'], color=colors)
ax.axvline(0.9, color='black', linestyle='--', linewidth=0.8, label='R2=0.9')
ax.set_xlabel('R2')
ax.set_title(f'Per-feature reconstruction R2 on test  (overall R2={test_r2:.4f})')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, f'test_per_feature_r2_dim{LATENT_DIM}.png'), dpi=150)
plt.close()
