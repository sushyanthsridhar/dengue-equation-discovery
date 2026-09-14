import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import optuna
from optuna.samplers import TPESampler
from utils import load_data
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(HERE, 'extended_input_normalized.csv')
TRAIN_YEARS = list(range(2015, 2019))
VAL_YEARS = [2019]
N_TRIALS = 100
EPOCHS = 200
PATIENCE = 40
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
raw, INPUT_FEATURES = load_data(DATA_CSV, TRAIN_YEARS)
INPUT_DIM = len(INPUT_FEATURES)
train_mask = raw['year'].isin(TRAIN_YEARS)
val_mask = raw['year'].isin(VAL_YEARS)
X_train = torch.tensor(raw.loc[train_mask, INPUT_FEATURES].values.astype(np.float32))
X_val = torch.tensor(raw.loc[val_mask, INPUT_FEATURES].values.astype(np.float32))

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
    return best_val_loss
sampler = TPESampler(seed=42)
pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=30)
study = optuna.create_study(direction='minimize', sampler=sampler, pruner=pruner)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
best = study.best_trial
for k, v in best.params.items():
    pass
config_path = os.path.join(HERE, 'best_config_autoencoder.txt')
with open(config_path, 'w') as f:
    f.write(f'val_mse={best.value:.6f}\n')
    for k, v in best.params.items():
        f.write(f'{k}={v}\n')
