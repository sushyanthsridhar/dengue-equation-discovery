"""
FunSearch's inner validation year (2018) is kept strictly separate from the
outer model-selection validation year (2019): 2019 is seen for the first
time only at the outer model-selection stage, never inside the FunSearch
search itself, so the outer validation evidence is independent of the term
structure the search already chose. See Section 4 (LLM-Guided Sparse
Dynamics Discovery) and Table 4 (year ranges per stage) in the paper.
"""
import os, sys, json, copy, random, uuid, time, hashlib, warnings
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Lasso, Ridge, ElasticNet, BayesianRidge
from sklearn.metrics import r2_score
from joblib import Parallel, delayed
from typing import Dict, List, Tuple, Any, Optional
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root: this script now lives one level down, in src/
PARENT = HERE  # repo root; HERE already resolved two levels up from src/, so PARENT is the same location
sys.path.insert(0, PARENT)
from utils import load_data
DATA_CSV = os.path.join(PARENT, 'extended_input_normalized.csv')
LATENT_CSV = os.path.join(PARENT, 'latents', 'latent_dim10.csv')
MODEL_PATH = os.path.join(PARENT, 'models', 'best_model_dim10.pt')
CLUSTER_CSV = os.path.join(PARENT, 'province_clusters.csv')
LOG_PATH = os.path.join(HERE, 'discover_equation_temporal_fix_log.txt')
LATENT_DIM = 10
TRAIN_YEARS = list(range(2015, 2019))
VAL_YEARS = [2019]
TEST_YEARS = [2022, 2023]
LOG_INCIDENCE = True
N_JOBS = -1
EXCLUDE_PROVINCES = ['16 Pedernales', '07 Elías Piña', '05 Dajabón', '03 Baoruco']
WEAK_PROVINCES = {'02 Azua', '08 El Seibo', '15 Monte Cristi', '17 Peravia', '26 Santiago Rodríguez', '20 Samaná'}
QUALITY_THRESHOLD = 0.2
QUALITY_W_COMPLETE = 0.2
QUALITY_W_SMOOTH = 0.35
QUALITY_W_STABLE = 0.2
QUALITY_W_CONSIST = 0.25
FS_N_ISLANDS = 4
FS_ISLAND_SIZE = 5
FS_N_ITERATIONS = 80
FS_MIGRATE_EVERY = 10
FS_MAX_EXTRA = 8
FS_DOUBLE_MUT = 0.15
FS_CROSSOVER = 0.2
FS_RANDOM_SEED = 42
FS_INNER_TRAIN = [2015, 2016, 2017]
FS_INNER_VAL = [2018]
THRESH_GLOBAL = 0.05
THRESH_CLUSTER = 0.1
THRESH_PROVINCE = 0.08
THRESH_INC_G = 0.04
THRESH_INC_C = 0.1
THRESH_INC_P = 100.0
STLSQ_ITERS = 30
POLY_DEG = 1
TAU0 = 10.0
HORSESHOE_ITERS = 5
SIGMA_H_FIXED = 0.05
SIGMA_U_FIXED = 0.1
SIGMA_FIXED = True
NOISE_FREEZE = True
Q_INIT_Z = 0.001
Q_INIT_INC = 0.05
R_INIT_Z = 0.0001
R_INIT_INC = 0.03
N_EM = 50
KAPPA = 5.0
ADAPT_LASSO_ITERS = 5
ADAPT_LASSO_GAMMA = 0.5
LASSO_HARD_ZERO = 0.005
ACTIVE_THRESH = 0.01
Q_INC_CLIP_LO = Q_INIT_INC * 0.1
Q_INC_CLIP_HI = Q_INIT_INC * 20.0
OLLAMA_URL = 'http://localhost:11434/api/chat'
OLLAMA_MODEL_PRIMARY = 'llama3.1:8b-instruct'  # referee_report change: try a stronger instruct model first
OLLAMA_MODEL_FALLBACK = 'mistral:latest'  # used automatically if the primary model is not pulled locally
FS_JSON_PATH = os.path.join(HERE, 'fsl_best.json')
FS_HISTORY_PATH = os.path.join(HERE, 'fsl_history.csv')
SKIP_FUNSEARCH = True
FS_GRID_TAU = [2, 3, 4, 5, 6, 7, 8, 9, 10]
FS_GRID_ALGOS = ['ridge', 'lasso', 'elasticnet', 'bayesian_ridge']
FS_EN_L1_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9]
FS_GRID_DIVERSITY = {'spread': {0: (range(3, 5), 0), 1: (range(5, 6), 0), 2: (range(6, 9), 0), 3: (range(4, 6), 3)}, 'narrow': {0: (range(3, 5), 0), 1: (range(4, 6), 0), 2: (range(5, 7), 0), 3: (range(6, 8), 0)}}
FS_DIR = os.path.join(HERE, 'fs_programs')
GRID_CSV = os.path.join(HERE, 'discover_equation_grid_results.csv')
GRID_LOG = os.path.join(HERE, 'discover_equation_grid_log.txt')
MAX_LLM_ROUNDS = 1  # referee_report note: left at 1 deliberately, this runs inside all 72 grid cells, raising it here would multiply grid search runtime; the richer multi-round loop lives in forecast_temporal_fix.py where only the single winning cell is refit
PLATEAU_DELTA = 0.002
PLATEAU_PATIENCE = 3
R2_WEIGHT = 0.65
SPECTRAL_WEIGHT = 0.35
KEY_PERIODS = [52, 26]
SPECTRAL_BIN_WINDOW = 2
random.seed(FS_RANDOM_SEED)
np.random.seed(FS_RANDOM_SEED)

class _Tee:

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
        else:
            pass

    def flush(self):
        for s in self.streams:
            s.flush()
        else:
            pass
_log_file = open(LOG_PATH, 'w', buffering=1)
sys.stdout = _Tee(sys.__stdout__, _log_file)
sys.stderr = _Tee(sys.__stderr__, _log_file)
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
                else:
                    pass
                enc_layers.append(act())
                if dropout > 0.0:
                    enc_layers.append(nn.Dropout(dropout))
                else:
                    pass
            else:
                pass
        else:
            pass
        self.encoder = nn.Sequential(*enc_layers)
        dec_dims = list(reversed(enc_dims))
        dec_layers = []
        for i in range(len(dec_dims) - 1):
            dec_layers.append(nn.Linear(dec_dims[i], dec_dims[i + 1]))
            if i < len(dec_dims) - 2:
                if use_batchnorm:
                    dec_layers.append(nn.BatchNorm1d(dec_dims[i + 1]))
                else:
                    pass
                dec_layers.append(act())
                if dropout > 0.0:
                    dec_layers.append(nn.Dropout(dropout))
                else:
                    pass
            else:
                pass
        else:
            pass
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x):
        return self.encoder(x)
raw, INPUT_FEATURES = load_data(DATA_CSV, TRAIN_YEARS)
INPUT_DIM = len(INPUT_FEATURES)
ae = WeeklyAutoencoder(INPUT_DIM, LATENT_DIM, [512, 256], 0.0087, 'elu', True)
ae.load_state_dict(torch.load(MODEL_PATH, map_location='cpu', weights_only=True))
ae.eval()
z_cols = [f'z{i + 1}' for i in range(LATENT_DIM)]
test_mask = raw['year'].isin(TEST_YEARS)
X_test_t = torch.tensor(raw.loc[test_mask, INPUT_FEATURES].values.astype(np.float32))
with torch.no_grad():
    Z_test = ae.encode(X_test_t).numpy()
test_lat = pd.DataFrame(Z_test, columns=z_cols)
test_lat['province'] = raw.loc[test_mask, 'province'].values
test_lat['year'] = raw.loc[test_mask, 'year'].values
test_lat['week'] = raw.loc[test_mask, 'week'].values
saved = pd.read_csv(LATENT_CSV)
latent = pd.concat([saved, test_lat], ignore_index=True)
latent = latent.merge(raw[['province', 'year', 'week', 'incidence']], on=['province', 'year', 'week'], how='left')
latent = latent.sort_values(['province', 'year', 'week']).reset_index(drop=True)
if LOG_INCIDENCE:
    latent['incidence'] = np.log1p(latent['incidence'].clip(lower=0))
else:
    pass
clusters = pd.read_csv(CLUSTER_CSV)[['province', 'cluster_id']]
latent = latent.merge(clusters, on='province', how='left')
latent = latent[~latent['province'].isin(EXCLUDE_PROVINCES)].reset_index(drop=True)
CLUSTER_IDS = sorted(latent['cluster_id'].dropna().unique().astype(int))
STATE_COLS = z_cols + ['incidence']
STATE_DIM = len(STATE_COLS)
INC_COL = STATE_DIM - 1
TRAINVAL_YEARS = TRAIN_YEARS + VAL_YEARS
trainval_mask = latent['year'].isin(TRAINVAL_YEARS)
SCALER = StandardScaler()
SCALER.fit(latent.loc[trainval_mask, STATE_COLS].values.astype(np.float32))
FS_SCALER = StandardScaler()
FS_SCALER.fit(latent.loc[latent['year'].isin(FS_INNER_TRAIN), STATE_COLS].values.astype(np.float32))
ISLAND_BIAS_TABLE = {0: (range(3, 5), 0, 'lasso'), 1: (range(5, 6), 0, 'ridge'), 2: (range(6, 9), 0, 'elasticnet'), 3: (range(4, 6), 3, 'bayesian_ridge'), 4: (range(3, 5), 0, 'ridge'), 5: (range(5, 6), 0, 'lasso'), 6: (range(6, 9), 2, 'elasticnet'), 7: (range(4, 6), 3, 'ridge'), 8: (range(3, 5), 0, 'lasso'), 9: (range(5, 7), 1, 'bayesian_ridge')}
TERM_TYPES = ['inc_pow', 'inc_z_prod', 'cross_inc', 'z_sq', 'z_prod', 'inc_diff', 'z_cube', 'z_inc_ratio']

@dataclass
class ExtraTerm:
    term_type: str
    params: Dict[str, Any] = field(default_factory=dict)

    def compute(self, X_lag: np.ndarray, n_lags: int) -> Optional[np.ndarray]:
        try:
            tt = self.term_type
            p = self.params
            ml = n_lags
            if tt == 'inc_pow':
                k = min(p.get('lag', 0), ml)
                pw = p.get('power', 2)
                v = X_lag[:, STATE_DIM * k + INC_COL] ** pw
            elif tt == 'inc_z_prod':
                ki = min(p.get('lag_inc', 0), ml)
                kz = min(p.get('lag_z', 0), ml)
                j = min(p.get('z_idx', 0), LATENT_DIM - 1)
                v = X_lag[:, STATE_DIM * ki + INC_COL] * X_lag[:, STATE_DIM * kz + j]
            elif tt == 'cross_inc':
                k1 = min(p.get('lag1', 0), ml)
                k2 = min(p.get('lag2', 1), ml)
                v = X_lag[:, STATE_DIM * k1 + INC_COL] * X_lag[:, STATE_DIM * k2 + INC_COL]
            elif tt == 'z_sq':
                k = min(p.get('lag', 0), ml)
                j = min(p.get('z_idx', 0), LATENT_DIM - 1)
                v = X_lag[:, STATE_DIM * k + j] ** 2
            elif tt == 'z_prod':
                k = min(p.get('lag', 0), ml)
                i = min(p.get('z_idx1', 0), LATENT_DIM - 1)
                j = min(p.get('z_idx2', 1), LATENT_DIM - 1)
                v = X_lag[:, STATE_DIM * k + i] * X_lag[:, STATE_DIM * k + j]
            elif tt == 'inc_diff':
                k1 = min(p.get('lag1', 0), ml)
                k2 = min(p.get('lag2', 1), ml)
                v = X_lag[:, STATE_DIM * k1 + INC_COL] - X_lag[:, STATE_DIM * k2 + INC_COL]
            elif tt == 'z_cube':
                k = min(p.get('lag', 0), ml)
                j = min(p.get('z_idx', 0), LATENT_DIM - 1)
                v = X_lag[:, STATE_DIM * k + j] ** 3
            elif tt == 'z_inc_ratio':
                k = min(p.get('lag', 0), ml)
                j = min(p.get('z_idx', 0), LATENT_DIM - 1)
                inc = X_lag[:, STATE_DIM * k + INC_COL]
                z = X_lag[:, STATE_DIM * k + j]
                v = z / (np.abs(inc) + 1e-06)
            else:
                return None
            return v.astype(np.float32).reshape(-1, 1)
        except Exception:
            return None
        else:
            pass
        finally:
            pass

    def describe(self) -> str:
        return f'{self.term_type}({self.params})'

    def to_dict(self):
        return {'term_type': self.term_type, 'params': dict(self.params)}

    @staticmethod
    def from_dict(d):
        return ExtraTerm(d['term_type'], d['params'])

@dataclass
class Program:
    program_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    n_lags: int = 5
    extra_terms: List[ExtraTerm] = field(default_factory=list)
    score: float = float('-inf')
    generation: int = 0
    parent_id: str = 'seed'

    def fingerprint(self) -> str:
        spec = json.dumps({'n_lags': self.n_lags, 'extra_terms': sorted([t.to_dict() for t in self.extra_terms], key=lambda x: json.dumps(x, sort_keys=True))}, sort_keys=True)
        return hashlib.md5(spec.encode()).hexdigest()[:12]

    def to_dict(self) -> Dict:
        return {'program_id': self.program_id, 'n_lags': self.n_lags, 'extra_terms': [t.to_dict() for t in self.extra_terms], 'score': self.score, 'generation': self.generation, 'parent_id': self.parent_id, 'spec': self.fingerprint()}

def _build_aug(x_sc: np.ndarray, n_lags: int) -> np.ndarray:
    T, d = x_sc.shape
    rows = []
    for i in range(n_lags, T):
        lags = [x_sc[i - lag] for lag in range(n_lags + 1)]
        rows.append(np.concatenate(lags))
    else:
        pass
    return np.array(rows, dtype=np.float32) if rows else np.zeros((0, (n_lags + 1) * d), dtype=np.float32)

def build_library(X_aug: np.ndarray, week_arr: np.ndarray, n_lags: int, fs_terms: List[ExtraTerm], llm_formulas: List[Tuple[str, str]]) -> Tuple[np.ndarray, List[str]]:
    d = STATE_DIM
    parts = [X_aug]
    names = []
    for lag in range(n_lags + 1):
        for dim in range(d):
            names.append(f's{dim}_lag{lag}')
        else:
            pass
    else:
        pass
    y_lag1 = X_aug[:, d + INC_COL:d + INC_COL + 1]
    z1_lag2 = X_aug[:, 2 * d + 0:2 * d + 1]
    parts.append(y_lag1 * z1_lag2)
    names.append('base_y1_z1_2')
    y_t = X_aug[:, INC_COL:INC_COL + 1]
    parts.append(y_t * (1.0 - y_t / KAPPA))
    names.append('logistic')
    for m in range(3):
        z_m = X_aug[:, d + m:d + m + 1]
        parts.append(y_lag1 * z_m)
        names.append(f'y1_zm{m}')
    else:
        pass
    t_rad = 2.0 * np.pi * week_arr.astype(np.float64) / 52.0
    parts.append(np.sin(t_rad).reshape(-1, 1).astype(np.float32))
    parts.append(np.cos(t_rad).reshape(-1, 1).astype(np.float32))
    names.extend(['sin52', 'cos52'])
    t_rad26 = 2.0 * np.pi * week_arr.astype(np.float64) / 26.0
    parts.append(np.sin(t_rad26).reshape(-1, 1).astype(np.float32))
    parts.append(np.cos(t_rad26).reshape(-1, 1).astype(np.float32))
    names.extend(['sin26', 'cos26'])
    I_cumul = sum((X_aug[:, d * lag + INC_COL:d * lag + INC_COL + 1] for lag in range(n_lags + 1))).astype(np.float32)
    parts.append(I_cumul)
    names.append('I_cumul')
    for term in fs_terms:
        col = term.compute(X_aug, n_lags)
        if col is not None and col.shape[0] == X_aug.shape[0]:
            if np.all(np.isfinite(col)):
                parts.append(col)
                names.append(f'fs_{term.describe()}')
            else:
                pass
        else:
            pass
    else:
        pass
    var_map = {}
    for lag in range(n_lags + 1):
        var_map[f'y_lag{lag}'] = X_aug[:, d * lag + INC_COL]
        for zi in range(LATENT_DIM):
            var_map[f'z{zi + 1}_lag{lag}'] = X_aug[:, d * lag + zi]
        else:
            pass
    else:
        pass
    var_map['week_arr'] = week_arr
    var_map['np'] = np
    for fname, formula in llm_formulas:
        try:
            local = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in var_map.items()}
            result = eval(formula, {'__builtins__': {}}, local)
            result = np.array(result, dtype=np.float32).reshape(-1)
            if result.shape[0] == X_aug.shape[0] and np.all(np.isfinite(result)):
                parts.append(result.reshape(-1, 1))
                names.append(f'llm_{fname}')
            else:
                pass
        except Exception:
            pass
        else:
            pass
        finally:
            pass
    else:
        pass
    return (np.concatenate(parts, axis=1).astype(np.float32), names)

def build_library_and_targets(x_sc: np.ndarray, weeks: np.ndarray, n_lags: int, fs_terms: List[ExtraTerm], llm_formulas: List[Tuple[str, str]]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    X_aug = _build_aug(x_sc, n_lags)
    Phi_full, nms = build_library(X_aug, weeks[n_lags:], n_lags, fs_terms, llm_formulas)
    targets = (x_sc[n_lags + 1:] - x_sc[n_lags:-1]).astype(np.float32)
    n_rows = min(len(Phi_full) - 1, len(targets))
    return (Phi_full[:n_rows].copy(), targets[:n_rows].copy(), nms)

def kalman_rts(s_obs: np.ndarray, u_seq: np.ndarray, q_diag: np.ndarray, r_diag: np.ndarray) -> np.ndarray:
    T, d = s_obs.shape
    I = np.eye(d)
    Q = np.diag(q_diag)
    R = np.diag(r_diag)
    x_f = np.zeros((T, d))
    P_f = np.zeros((T, d, d))
    x_a = np.zeros((T, d))
    P_a = np.zeros((T, d, d))
    x_a[0] = s_obs[0]
    P_a[0] = R.copy()
    for t in range(1, T):
        u = u_seq[t - 1]
        x_f[t] = x_a[t - 1] + u
        P_f[t] = P_a[t - 1] + Q
        S = P_f[t] + R
        K = np.linalg.solve(S.T, P_f[t].T).T
        x_a[t] = x_f[t] + K @ (s_obs[t] - x_f[t])
        P_a[t] = (I - K) @ P_f[t]
    else:
        pass
    x_s = x_a.copy()
    for t in range(T - 2, -1, -1):
        G = np.linalg.solve(P_f[t + 1].T, P_a[t].T).T
        x_s[t] = x_a[t] + G @ (x_s[t + 1] - x_f[t + 1])
    else:
        pass
    return x_s

def _smooth_one(prov, s_obs, weeks, beta_p, x_prev, q_diag, r_diag, n_lags, fs_terms, llm_formulas):
    T = len(s_obs)
    if T <= n_lags + 1:
        return (prov, s_obs.copy())
    else:
        pass
    X_aug = _build_aug(x_prev, n_lags)
    Phi_full, _ = build_library(X_aug, weeks[n_lags:], n_lags, fs_terms, llm_formulas)
    u_seq = np.zeros((T - 1, s_obs.shape[1]), dtype=np.float32)
    n_u = min(Phi_full.shape[0], T - 1)
    u_seq[:n_u] = Phi_full[:n_u] @ beta_p
    return (prov, kalman_rts(s_obs, u_seq, q_diag, r_diag))

def mstep_global(Phi_all: np.ndarray, res_all: np.ndarray) -> np.ndarray:
    M = Phi_all.shape[1]
    PtP = (Phi_all.T @ Phi_all).astype(np.float64)
    Pty = (Phi_all.T @ res_all).astype(np.float64)
    lam = TAU0 ** 2
    W = np.linalg.solve(PtP + lam * np.eye(M), Pty)
    for _ in range(ADAPT_LASSO_ITERS):
        inc_mag = np.abs(W[:, INC_COL]) ** ADAPT_LASSO_GAMMA
        w_adapt = lam / (inc_mag + 0.001)
        pen_adapt = np.diag(w_adapt)
        W[:, INC_COL] = np.linalg.solve(PtP + pen_adapt, Pty[:, INC_COL])
    else:
        pass
    W[:, INC_COL] = np.where(np.abs(W[:, INC_COL]) < LASSO_HARD_ZERO, 0.0, W[:, INC_COL])
    return W.astype(np.float32)

def mstep_cluster(Phi_c: np.ndarray, res_c: np.ndarray, sigma_h: float) -> np.ndarray:
    M = Phi_c.shape[1]
    PtP = (Phi_c.T @ Phi_c).astype(np.float64)
    Pty = (Phi_c.T @ res_c).astype(np.float64)
    pen = 1.0 / max(sigma_h, 1e-06) ** 2 * np.eye(M)
    return np.linalg.solve(PtP + pen, Pty).astype(np.float32)

def mstep_province(Phi_p: np.ndarray, res_p: np.ndarray, sigma_u: float) -> np.ndarray:
    M = Phi_p.shape[1]
    PtP = (Phi_p.T @ Phi_p).astype(np.float64)
    Pty = (Phi_p.T @ res_p).astype(np.float64)
    pen = 1.0 / max(sigma_u, 1e-06) ** 2 * np.eye(M)
    return np.linalg.solve(PtP + pen, Pty).astype(np.float32)

def update_sigma_inc(offsets: List[np.ndarray]) -> float:
    inc_vals = np.concatenate([o[:, INC_COL] for o in offsets])
    return float(np.std(inc_vals)) + 1e-06

def compute_province_quality_scores(province_data: Dict) -> pd.DataFrame:
    mean_inc_sc = float(SCALER.mean_[INC_COL])
    std_inc_sc = float(SCALER.scale_[INC_COL])
    records = []
    for prov, pd_t in province_data.items():
        inc_sc = pd_t['s_obs'][:, INC_COL]
        inc_log = inc_sc * std_inc_sc + mean_inc_sc
        inc_raw = np.expm1(inc_log).clip(0)
        T = len(inc_raw)
        zero_rate = float(np.mean(inc_raw < 0.5))
        completeness = 1.0 - zero_rate
        if T > 2:
            ac = float(np.corrcoef(inc_raw[:-1], inc_raw[1:])[0, 1])
            smoothness = max(0.0, ac if np.isfinite(ac) else 0.0)
        else:
            smoothness = 0.0
        mean_inc = float(np.mean(inc_raw))
        std_inc = float(np.std(inc_raw))
        cv = std_inc / (mean_inc + 1e-06)
        stability = 1.0 / (1.0 + cv)
        n_complete = T // 52 * 52
        if n_complete >= 104:
            profiles = inc_raw[:n_complete].reshape(-1, 52)
            n_y = profiles.shape[0]
            corrs = []
            for i in range(n_y):
                for j in range(i + 1, n_y):
                    c = np.corrcoef(profiles[i], profiles[j])[0, 1]
                    if np.isfinite(c):
                        corrs.append(c)
                    else:
                        pass
                else:
                    pass
            else:
                pass
            consistency = max(0.0, float(np.mean(corrs))) if corrs else 0.5
        else:
            consistency = 0.5
        records.append({'province': prov, 'completeness': completeness, 'smoothness': smoothness, 'stability': stability, 'consistency': consistency, 'zero_rate': zero_rate, 'cv': cv})
    else:
        pass
    df = pd.DataFrame(records)
    for col in ['completeness', 'smoothness', 'stability', 'consistency']:
        lo = df[col].min()
        hi = df[col].max()
        if hi > lo:
            df[f'{col}_norm'] = (df[col] - lo) / (hi - lo)
        else:
            df[f'{col}_norm'] = 0.5
    else:
        pass
    df['quality_score'] = QUALITY_W_COMPLETE * df['completeness_norm'] + QUALITY_W_SMOOTH * df['smoothness_norm'] + QUALITY_W_STABLE * df['stability_norm'] + QUALITY_W_CONSIST * df['consistency_norm']
    return df.sort_values('quality_score', ascending=False).reset_index(drop=True)

def run_hierarchical_em(province_data: Dict, n_lags: int, fs_terms: List[ExtraTerm], llm_formulas: List[Tuple[str, str]], label: str='hierarchical', g_init: Optional[np.ndarray]=None, h_init: Optional[Dict]=None, u_init: Optional[Dict]=None) -> Dict:
    provinces = list(province_data.keys())
    q_diag = np.array([Q_INIT_Z] * LATENT_DIM + [Q_INIT_INC], dtype=np.float32)
    r_diag = np.array([R_INIT_Z] * LATENT_DIM + [R_INIT_INC], dtype=np.float32)
    sigma_h = SIGMA_H_FIXED
    sigma_u = SIGMA_U_FIXED
    sample_s = province_data[provinces[0]]['s_obs']
    sample_w = province_data[provinces[0]]['weeks']
    Phi_samp, _, term_names = build_library_and_targets(sample_s, sample_w, n_lags, fs_terms, llm_formulas)
    M = Phi_samp.shape[1]

    def _warm_copy(arr_init, shape):
        out = np.zeros(shape, dtype=np.float32)
        if arr_init is not None:
            n = min(arr_init.shape[0], shape[0])
            out[:n] = arr_init[:n]
        else:
            pass
        return out
    g = _warm_copy(g_init, (M, STATE_DIM))
    h = {cid: _warm_copy(h_init.get(cid) if h_init else None, (M, STATE_DIM)) for cid in CLUSTER_IDS}
    u = {prov: _warm_copy(u_init.get(prov) if u_init else None, (M, STATE_DIM)) for prov in provinces}
    x_smooth_all = {prov: province_data[prov]['s_obs'].copy() for prov in provinces}
    for em_iter in range(N_EM):
        if em_iter == 0 or (em_iter + 1) % 10 == 0 or em_iter == N_EM - 1:
            print(f'      [EM] {label}: iteration {em_iter + 1}/{N_EM}')
        else:
            pass
        beta_by_prov = {prov: g + h[province_data[prov]['cluster']] + u[prov] for prov in provinces}
        results = Parallel(n_jobs=N_JOBS)((delayed(_smooth_one)(prov, province_data[prov]['s_obs'], province_data[prov]['weeks'], beta_by_prov[prov], x_smooth_all[prov], q_diag, r_diag, n_lags, fs_terms, llm_formulas) for prov in provinces))
        for prov, x_s in results:
            x_smooth_all[prov] = x_s
        else:
            pass
        Phi_by_prov = {}
        tgt_by_prov = {}
        for prov in provinces:
            Phi_p, tgt_p, _ = build_library_and_targets(x_smooth_all[prov], province_data[prov]['weeks'], n_lags, fs_terms, llm_formulas)
            Phi_by_prov[prov] = Phi_p
            tgt_by_prov[prov] = tgt_p
        else:
            pass
        Phi_pool, res_pool = ([], [])
        for prov in provinces:
            cid = province_data[prov]['cluster']
            Phi_p = Phi_by_prov[prov]
            tgt_p = tgt_by_prov[prov]
            res_p = tgt_p - Phi_p @ h[cid] - Phi_p @ u[prov]
            Phi_pool.append(Phi_p)
            res_pool.append(res_p)
        else:
            pass
        g = mstep_global(np.vstack(Phi_pool), np.vstack(res_pool))
        for cid in CLUSTER_IDS:
            c_provs = [pv for pv in provinces if province_data[pv]['cluster'] == cid]
            if not c_provs:
                continue
            else:
                pass
            Phi_c = np.vstack([Phi_by_prov[pv] for pv in c_provs])
            res_c = np.vstack([tgt_by_prov[pv] - Phi_by_prov[pv] @ g - Phi_by_prov[pv] @ u[pv] for pv in c_provs])
            h[cid] = mstep_cluster(Phi_c, res_c, sigma_h)
        else:
            pass
        for prov in provinces:
            cid = province_data[prov]['cluster']
            Phi_p = Phi_by_prov[prov]
            res_p = tgt_by_prov[prov] - Phi_p @ g - Phi_p @ h[cid]
            u[prov] = mstep_province(Phi_p, res_p, sigma_u)
        else:
            pass
        if em_iter % 5 == 0 or em_iter == N_EM - 1:
            active = int((np.abs(g) > ACTIVE_THRESH).any(axis=1).sum())
            weak_u_rms = float(np.mean([np.sqrt(np.mean(u[pv][:, INC_COL] ** 2)) for pv in provinces if pv in WEAK_PROVINCES]))
        else:
            pass
    else:
        pass
    return dict(g=g, h=h, u=u, q_diag=q_diag, r_diag=r_diag, term_names=term_names, sigma_h=sigma_h, sigma_u=sigma_u, n_lags=n_lags, fs_terms=fs_terms, llm_formulas=llm_formulas)

def estimate_province_q_inc(model: Dict, province_data: Dict) -> Dict[str, float]:
    n_lags = model['n_lags']
    fs_terms = model['fs_terms']
    llm_formulas = model['llm_formulas']
    q_inc_by_prov = {}
    for prov, pd_t in province_data.items():
        cid = pd_t['cluster']
        s_obs = pd_t['s_obs']
        beta_p = model['g'] + model['h'].get(cid, np.zeros_like(model['g'])) + model['u'].get(prov, np.zeros(model['g'].shape[0]))
        try:
            Phi_p, tgt_p, _ = build_library_and_targets(s_obs, pd_t['weeks'], n_lags, fs_terms, llm_formulas)
            expected_M = beta_p.shape[0]
            if Phi_p.shape[1] != expected_M:
                Phi_p = Phi_p[:, :expected_M] if Phi_p.shape[1] > expected_M else np.pad(Phi_p, ((0, 0), (0, expected_M - Phi_p.shape[1])))
            else:
                pass
            pred_p = Phi_p @ beta_p
            res_inc = tgt_p[:, INC_COL] - pred_p[:, INC_COL]
            q_inc = float(np.var(res_inc)) if len(res_inc) > 1 else Q_INIT_INC
            q_inc = float(np.clip(q_inc, Q_INC_CLIP_LO, Q_INC_CLIP_HI))
        except Exception:
            q_inc = Q_INIT_INC
        else:
            pass
        finally:
            pass
        q_inc_by_prov[prov] = q_inc
    else:
        pass
    return q_inc_by_prov

def forecast_sequential(s_test: np.ndarray, weeks_test: np.ndarray, beta_p: np.ndarray, q_diag: np.ndarray, r_diag: np.ndarray, n_lags: int, fs_terms: List[ExtraTerm], llm_formulas: List[Tuple[str, str]]) -> np.ndarray:
    T = len(s_test)
    d = STATE_DIM
    I = np.eye(d)
    Q = np.diag(q_diag)
    R = np.diag(r_diag)
    x_a = s_test[n_lags].copy()
    P_a = R.copy()
    x_hist = [s_test[max(0, n_lags - lag)] for lag in range(n_lags + 1)]
    yhat = []
    for t in range(n_lags, T - 1):
        x_aug_t = np.concatenate([x_hist[lag] for lag in range(n_lags + 1)]).reshape(1, -1)
        Phi_t, _ = build_library(x_aug_t.astype(np.float32), np.array([weeks_test[t]]), n_lags, fs_terms, llm_formulas)
        expected_M = beta_p.shape[0]
        if Phi_t.shape[1] < expected_M:
            Phi_t = np.pad(Phi_t, ((0, 0), (0, expected_M - Phi_t.shape[1])), mode='constant')
        elif Phi_t.shape[1] > expected_M:
            Phi_t = Phi_t[:, :expected_M]
        else:
            pass
        u_t = (Phi_t @ beta_p).flatten()
        x_f = x_a + u_t
        P_f = P_a + Q
        yhat.append(float(x_f[INC_COL]))
        if t + 1 < T:
            S = P_f + R
            K = np.linalg.solve(S.T, P_f.T).T
            x_a = x_f + K @ (s_test[t + 1] - x_f)
            P_a = (I - K) @ P_f
            x_hist = [x_a.copy()] + x_hist[:-1]
        else:
            pass
    else:
        pass
    return np.array(yhat, dtype=np.float32)

def evaluate_on_split(province_data: Dict, model: Dict) -> Tuple[float, pd.DataFrame, Dict]:
    n_lags = model['n_lags']
    fs_terms = model['fs_terms']
    llm_formulas = model['llm_formulas']
    mean_inc = float(SCALER.mean_[INC_COL])
    std_inc = float(SCALER.scale_[INC_COL])
    q_inc_by_prov = model.get('q_inc_by_prov', {})
    results = []
    prov_preds = {}
    for prov, pd_t in province_data.items():
        cid = pd_t['cluster']
        s_test = pd_t['s_obs']
        weeks = pd_t['weeks']
        T = len(s_test)
        if T <= n_lags + 1:
            continue
        else:
            pass
        beta_p = model['g'] + model['h'].get(cid, 0) + model['u'].get(prov, 0)
        q_diag_p = model['q_diag'].copy()
        if prov in q_inc_by_prov:
            q_diag_p[INC_COL] = q_inc_by_prov[prov]
        else:
            pass
        yhat_sc = forecast_sequential(s_test, weeks, beta_p, q_diag_p, model['r_diag'], n_lags, fs_terms, llm_formulas)
        y_true_sc = s_test[n_lags + 1:n_lags + 1 + len(yhat_sc), INC_COL]
        yhat_log = yhat_sc * std_inc + mean_inc
        ytrue_log = y_true_sc * std_inc + mean_inc
        if LOG_INCIDENCE:
            yhat_cnt = np.expm1(yhat_log)
            ytrue_cnt = np.expm1(ytrue_log)
        else:
            yhat_cnt = yhat_log
            ytrue_cnt = ytrue_log
        r2 = float(r2_score(ytrue_cnt, yhat_cnt))
        mae = float(np.mean(np.abs(ytrue_cnt - yhat_cnt)))
        mse = float(np.mean((ytrue_cnt - yhat_cnt) ** 2))
        results.append({'cluster_id': cid, 'province': prov, 'r2': r2, 'mae': mae, 'mse': mse})
        prov_preds[prov] = (ytrue_cnt, yhat_cnt)
    else:
        pass
    df = pd.DataFrame(results).sort_values('r2', ascending=False)
    mean = float(df['r2'].mean()) if len(df) > 0 else float('-inf')
    return (mean, df, prov_preds)

def compute_spectral_score(prov_preds: Dict[str, Tuple[np.ndarray, np.ndarray]], key_periods: List[int]=KEY_PERIODS, bin_window: int=SPECTRAL_BIN_WINDOW) -> float:
    lcm_periods = 52
    province_scores = []
    for prov, (ytrue, yhat) in prov_preds.items():
        N = len(ytrue)
        if N < max(key_periods) + 1:
            continue
        else:
            pass
        residuals = ytrue - yhat
        ytrue_dm = ytrue - ytrue.mean()
        resid_dm = residuals - residuals.mean()
        window = np.hanning(N)
        ytrue_w = ytrue_dm * window
        resid_w = resid_dm * window
        N_pad = int(np.ceil(max(8 * N, 4 * lcm_periods) / lcm_periods) * lcm_periods)
        fft_true = np.fft.rfft(ytrue_w, n=N_pad)
        fft_res = np.fft.rfft(resid_w, n=N_pad)
        pwr_true = np.abs(fft_true) ** 2
        pwr_res = np.abs(fft_res) ** 2
        band_scores = []
        for period in key_periods:
            bin_center = N_pad // period
            bin_lo = max(1, bin_center - bin_window)
            bin_hi = min(len(pwr_true) - 1, bin_center + bin_window)
            p_true = float(np.sum(pwr_true[bin_lo:bin_hi + 1]))
            p_res = float(np.sum(pwr_res[bin_lo:bin_hi + 1]))
            if p_true > 1e-12:
                band_scores.append(max(0.0, 1.0 - p_res / p_true))
            else:
                pass
        else:
            pass
        if band_scores:
            province_scores.append(float(np.mean(band_scores)))
        else:
            pass
    else:
        pass
    return float(np.mean(province_scores)) if province_scores else 0.0

def combined_score(r2: float, spectral: float) -> float:
    return R2_WEIGHT * r2 + SPECTRAL_WEIGHT * spectral

def _build_province_dict(data_df: pd.DataFrame, years: List[int]) -> Dict:
    sub = data_df[data_df['year'].isin(years)]
    out = {}
    for prov, grp in sub.groupby('province'):
        grp = grp.sort_values(['year', 'week']).reset_index(drop=True)
        s_sc = SCALER.transform(grp[STATE_COLS].values.astype(np.float32))
        weeks = grp['week'].values.astype(np.float32)
        cid = int(grp['cluster_id'].iloc[0])
        out[prov] = {'s_obs': s_sc, 'weeks': weeks, 'cluster': cid}
    else:
        pass
    return out

_LATENT_MEANING_CACHE = None

def compute_latent_covariate_meaning():
    """Referee_report addition. z1 through z10 are anonymous autoencoder latent
    dimensions, the LLM has no way to know what physical quantity each one stands
    in for. This computes, once per run, the correlation of each z dimension with
    the real named covariates in the raw data, on training years only so nothing
    from validation or test leaks in, and gives the LLM a genuine grounding
    sentence such as z6 correlates with avg_rainfall instead of an invented one."""
    global _LATENT_MEANING_CACHE
    if _LATENT_MEANING_CACHE is not None:
        return _LATENT_MEANING_CACHE
    else:
        pass
    try:
        cov_cols = ['avg_temp', 'avg_rainfall', 'avg_humidity', 'nino34_anom', 'soi_index', 'tna_sst_anom', 'solar_radiation', 'wind_speed', 'dew_point_temp', 'neighbor_incidence_lag0', 'consecutive_dry_weeks', 'consecutive_wet_weeks', 'rainy_season_phase', 'school_calendar_active']
        raw = pd.read_csv(DATA_CSV, usecols=['province', 'year', 'week'] + cov_cols)
        raw = raw[raw['year'].isin(TRAIN_YEARS)]
        z_cols = [f'z{i}' for i in range(1, LATENT_DIM + 1)]
        lat = pd.read_csv(LATENT_CSV, usecols=['province', 'year', 'week'] + z_cols)
        lat = lat[lat['year'].isin(TRAIN_YEARS)]
        merged = pd.merge(lat, raw, on=['province', 'year', 'week'], how='inner')
        meaning = {}
        for zi in range(1, LATENT_DIM + 1):
            zcol = f'z{zi}'
            corrs = merged[cov_cols].corrwith(merged[zcol])
            corrs = corrs.dropna()
            corrs = corrs.reindex(corrs.abs().sort_values(ascending=False).index)
            top = corrs.head(2)
            parts = [f'{name} r={val:+.2f}' for name, val in top.items()]
            meaning[zi] = ', '.join(parts) if parts else 'no strong correlation with observed covariates'
        else:
            pass
        _LATENT_MEANING_CACHE = meaning
        print(f'  [LLM prompt] computed latent to covariate correlation table for {len(meaning)} z dimensions, training years only')
        return meaning
    except Exception as e:
        print(f'  [LLM prompt] could not compute latent covariate meaning, {type(e).__name__}: {e}')
        _LATENT_MEANING_CACHE = {}
        return {}
    else:
        pass
    finally:
        pass

def build_llm_prompt(fs_best: Program, history_rows: List[Dict], all_programs: List[Dict], val_r2: float, spectral_sc: float, spectral_delta: float, active_terms: List[Tuple[str, float]], pruned_terms: List[str], cluster_r2: Dict[int, float], llm_round: int, prev_suggestions: List[Dict], prev_survived: List[str], val_r2_history: List[float], latent_meaning: Dict[int, str] = None) -> str:
    if val_r2_history:
        trend_parts = [f'Round {i + 1}: {r:.4f}' for i, r in enumerate(val_r2_history)]
        trend_str = '  ' + '  |  '.join(trend_parts)
        if len(val_r2_history) >= 2:
            delta_overall = val_r2_history[-1] - val_r2_history[0]
            trend_str += f'\n  Overall change from round 1: {delta_overall:+.4f}'
        else:
            pass
    else:
        trend_str = '  (first round, no history yet)'
    prev_sug_str = ''
    if prev_suggestions:
        active_coef_map = {nm: coef for nm, coef in active_terms}
        survived_set = set(prev_survived)
        prev_sug_str = '\nPREVIOUS LLM SUGGESTIONS AND OUTCOMES:\n'
        prev_sug_str += '  (terms with |coef| < 0.01 after fitting are dropped as noise)\n'
        for s in prev_suggestions:
            name = s.get('name', '?')
            formula = s.get('formula', '?')
            key = f'llm_{name}'
            if key in survived_set:
                coef = active_coef_map.get(key, 0.0)
                if abs(coef) >= 0.02:
                    outcome = f'KEPT  coef={coef:+.5f}  (meaningful contribution)'
                else:
                    outcome = f'KEPT  coef={coef:+.5f}  (weak, near noise floor)'
            else:
                outcome = 'DROPPED  coef below 0.01 threshold'
            prev_sug_str += f'  {name}: {formula} -> {outcome}\n'
        else:
            pass
    else:
        pass
    llm_active = [(nm, c) for nm, c in active_terms if nm.startswith('llm_')]
    base_active = [(nm, c) for nm, c in active_terms if not nm.startswith('llm_')]
    active_str = '  BASE terms (top 10 by magnitude):\n'
    active_str += '\n'.join((f'    {nm:35s}  coef={coef:+.5f}' for nm, coef in base_active[:10]))
    if llm_active:
        active_str += '\n  LLM terms currently in model:\n'
        active_str += '\n'.join((f'    {nm:35s}  coef={coef:+.5f}' for nm, coef in llm_active))
    else:
        pass
    n_pruned = len(pruned_terms)
    pruned_llm = [nm for nm in pruned_terms if nm.startswith('llm_')]
    pruned_str = f'{n_pruned} terms total dropped (coef < {ACTIVE_THRESH})'
    if pruned_llm:
        pruned_str += f", including LLM terms: {', '.join(pruned_llm)}"
    else:
        pass
    cluster_str = '\n'.join((f'  Cluster {cid}: R2={r2:.4f}' for cid, r2 in sorted(cluster_r2.items())))
    if latent_meaning:
        meaning_str = '\n'.join((f'  z{zi}: {desc}' for zi, desc in sorted(latent_meaning.items())))
    else:
        meaning_str = '  (correlation table unavailable this run, treat z dimensions as unlabelled climate and mobility factors)'
    return f"""You are an expert epidemiologist and dynamical systems researcher collaborating on a SINDy (Sparse Identification of Nonlinear Dynamics) model for dengue forecasting.\n\nYOUR ROLE IN THE PIPELINE:\nYou are proposing candidate library terms for the dI/dt equation in a hierarchical SINDy model. The model structure is W_p = g + h_c + u_p where g is a global SINDy coefficient matrix shared by all 28 provinces, h_c is a small cluster correction, and u_p is a tiny province correction. You are improving g specifically for the incidence (dI/dt) dimension. After you propose terms, they are added to the regression library and the EM solver refits g on training data 2015-2018. Validation is then run on 2019 data only. Terms whose fitted coefficient falls below {ACTIVE_THRESH} in absolute value are dropped automatically. Only propose terms that have a strong biological or mathematical justification for a non-trivial coefficient.\n\nWHAT z1 THROUGH z10 ACTUALLY ARE:\nThese are not raw climate series, they are latent dimensions learned by an autoencoder compressing 36 raw covariates per province per week. The list below is the closest real world reading of what each latent dimension stands in for, computed as its correlation with the original named covariates on training years only. Ground your proposals in this instead of guessing which z is which.\n{meaning_str}\n\nBIOLOGICAL CONTEXT FOR TERM DESIGN:\nDengue transmission in the Dominican Republic follows a seasonal pattern driven by:\n1. Aedes aegypti breeding cycle: rainfall and temperature drive mosquito abundance with roughly a 2 to 4 week lag. Interaction terms like season * z_lag2 or z_lag3 capture this.\n2. Extrinsic incubation period: the virus takes 8 to 12 days to develop in the mosquito. A 1 to 2 week lag between climate forcing and incidence change is biologically correct.\n3. Herd immunity and susceptible depletion: after a large outbreak, incidence self-suppresses. A logistic saturation term like y_lag1 * (1 - y_lag1) or I_cumulative already exists in the base library. Products of incidence with lagged incidence can also capture refractory dynamics.\n4. Outbreak threshold: dengue tends to either grow explosively or fade. A threshold-like nonlinearity can be captured by y_lag1^2 or y_lag1 * z_climate without needing explicit threshold constants.\n5. Semi-annual cycle: some provinces show a secondary peak around week 26. A 26-week seasonal interaction on top of the 52-week base already in the model can capture this.\n\nVAL R2 TREND ACROSS ROUNDS (training 2015-2018, validation 2019):\n{trend_str}\nIf val R2 is declining across rounds, your previous suggestions introduced terms that overfit 2019. Propose fewer and more biologically specific terms this round.\n\nCURRENT MODEL STATE (round {llm_round}):\nValidation R2 = {val_r2:.4f}  |  Spectral score = {spectral_sc:.4f}  (annual + semi-annual periodicity)\nSpectral trend this round = {spectral_delta:+.4f}  ({('improving' if spectral_delta >= 0 else 'DECLINING: prioritise seasonal structure terms')})\n\nPer-cluster validation R2:\n{cluster_str}\nNote: Cluster 3 (Azua, Peravia, San Jose de Ocoa, mountain provinces) is the hardest. These provinces have irregular year-to-year outbreak timing. Consider threshold or saturation terms for these dynamics.\n\nACTIVE TERMS IN THE CURRENT INCIDENCE EQUATION (sorted by coefficient magnitude):\n{active_str}\n\nTerms dropped this round (|coef| < {ACTIVE_THRESH}):\n  {pruned_str}\n{prev_sug_str}\nYOUR TASK:\nSuggest exactly 8 new candidate library terms grounded in dengue biology. Each formula must be a valid numpy expression using ONLY the variables below. Do NOT use any named constants (threshold, beta, K, R0, mu) or any variable not in the list. Every formula must evaluate to a numpy array of the same shape as the input arrays.\n\nBiological priorities in order:\n1. Climate-lag interactions: seasonal signal (sin52 or cos52) multiplied by a lagged z-dimension, e.g. np.sin(2*np.pi*week_arr/52) * z3_lag2.\n2. Incidence-environment couplings at biologically correct lags (1 to 3 weeks), e.g. y_lag1 * z2_lag2.\n3. Susceptible depletion proxies: product of current incidence with a lagged incidence difference, e.g. y_lag0 * (y_lag1 - y_lag3).\n4. Secondary seasonality: 26-week interaction, e.g. np.sin(2*np.pi*week_arr/26) * y_lag1.\n5. Outbreak nonlinearity: squared incidence or incidence times a climate z, e.g. y_lag1**2 or y_lag1 * z5_lag2.\nDo NOT propose pure polynomial interactions of z-dimensions with no incidence involvement, such as z3*z7 or z4^2. These have no direct biological pathway to dI/dt and have consistently produced near-zero coefficients in previous rounds.\n\nRespond ONLY with a valid JSON array. No text before or after. Example:\n[\n  {{"name": "sin52_z3_lag2", "formula": "np.sin(2*np.pi*week_arr/52) * z3_lag2", "reason": "seasonal modulation by climate lag 2 weeks, matching extrinsic incubation period"}},\n  {{"name": "y_z2_lag2", "formula": "y_lag1 * z2_lag2", "reason": "incidence times climate forcing at 2-week lag, vectorial capacity proxy"}},\n  {{"name": "y_depletion", "formula": "y_lag0 * (y_lag1 - y_lag3)", "reason": "susceptible depletion: high incidence suppresses future growth"}},\n  ...\n]\n\nCOMPLETE LIST of available variables. Use ONLY these:\ny_lag0, y_lag1, y_lag2, y_lag3\nz1_lag0, z2_lag0, z3_lag0, z4_lag0, z5_lag0, z6_lag0, z7_lag0, z8_lag0, z9_lag0, z10_lag0\nz1_lag1, z2_lag1, z3_lag1, z4_lag1, z5_lag1, z6_lag1, z7_lag1, z8_lag1, z9_lag1, z10_lag1\nz1_lag2, z2_lag2, z3_lag2, z4_lag2, z5_lag2, z6_lag2, z7_lag2, z8_lag2, z9_lag2, z10_lag2\nz1_lag3, z2_lag3, z3_lag3, z4_lag3, z5_lag3, z6_lag3, z7_lag3, z8_lag3, z9_lag3, z10_lag3\nweek_arr, np\n\nDo NOT use: threshold, beta, K, K1, K2, alpha, gamma, R0, mu, or any name not in the list."""

def query_llm(prompt: str) -> str:
    for model_name in (OLLAMA_MODEL_PRIMARY, OLLAMA_MODEL_FALLBACK):
        payload = {'model': model_name, 'messages': [{'role': 'user', 'content': prompt}], 'stream': False, 'options': {'temperature': 0.4, 'num_predict': 1200}}
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
            resp.raise_for_status()
            result = resp.json().get('message', {}).get('content', '')
            if model_name != OLLAMA_MODEL_PRIMARY:
                print(f'  [LLM] primary model unavailable, used fallback model {model_name} instead')
            else:
                pass
            return result
        except Exception as e:
            print(f'  [LLM] Ollama call with {model_name} FAILED, {type(e).__name__}: {e}')
            continue
        else:
            pass
        finally:
            pass
    else:
        pass
    print('  [LLM] all configured models failed, returning empty response for this round')
    return ''

def parse_llm_suggestions(raw_text: str) -> List[Dict]:
    try:
        start = raw_text.find('[')
        end = raw_text.rfind(']') + 1
        if start == -1 or end == 0:
            return []
        else:
            pass
        json_str = raw_text[start:end]
        suggestions = json.loads(json_str)
        return suggestions
    except Exception as e:
        return []
    else:
        pass
    finally:
        pass

def suggestions_to_formulas(suggestions: List[Dict], X_aug_sample: np.ndarray, n_lags: int, week_arr: np.ndarray) -> List[Tuple[str, str]]:
    d = STATE_DIM
    var_map = {}
    for lag in range(n_lags + 1):
        var_map[f'y_lag{lag}'] = X_aug_sample[:, d * lag + INC_COL]
        for zi in range(LATENT_DIM):
            var_map[f'z{zi + 1}_lag{lag}'] = X_aug_sample[:, d * lag + zi]
        else:
            pass
    else:
        pass
    var_map['week_arr'] = week_arr
    var_map['np'] = np
    single_map = {}
    for lag in range(n_lags + 1):
        single_map[f'y_lag{lag}'] = X_aug_sample[:1, d * lag + INC_COL]
        for zi in range(LATENT_DIM):
            single_map[f'z{zi + 1}_lag{lag}'] = X_aug_sample[:1, d * lag + zi]
        else:
            pass
    else:
        pass
    single_map['week_arr'] = week_arr[:1]
    single_map['np'] = np
    valid = []
    for sug in suggestions:
        name = sug.get('name', '')
        formula = sug.get('formula', '')
        reason = sug.get('reason', '')
        if not name or not formula:
            continue
        else:
            pass
        try:
            local = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in var_map.items()}
            result = eval(formula, {'__builtins__': {}}, local)
            result = np.array(result, dtype=np.float32).reshape(-1)
            if result.shape[0] != X_aug_sample.shape[0] or not np.all(np.isfinite(result)):
                continue
            else:
                pass
            local1 = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in single_map.items()}
            result1 = eval(formula, {'__builtins__': {}}, local1)
            result1 = np.array(result1, dtype=np.float32).reshape(-1)
            if result1.shape[0] != 1 or not np.all(np.isfinite(result1)):
                continue
            else:
                pass
            valid.append((name, formula))
        except Exception as e:
            pass
        else:
            pass
        finally:
            pass
    else:
        pass
    return valid

def get_active_and_pruned(model: Dict) -> Tuple[List[Tuple[str, float]], List[str]]:
    g = model['g']
    names = model['term_names']
    g_inc = g[:, INC_COL] if g.ndim == 2 else g
    active = [(nm, float(coef)) for nm, coef in zip(names, g_inc) if abs(float(coef)) > ACTIVE_THRESH]
    pruned = [nm for nm, coef in zip(names, g_inc) if abs(float(coef)) <= ACTIVE_THRESH]
    active.sort(key=lambda x: abs(x[1]), reverse=True)
    return (active, pruned)

def _tune_elasticnet_l1(fs_train_pd, alpha):
    from sklearn.linear_model import ElasticNet
    from sklearn.model_selection import KFold
    seed_prog = Program(n_lags=3)
    Phi_list, tgt_list = ([], [])
    for prov, pd_t in fs_train_pd.items():
        try:
            Phi_p, tgt_p, _ = build_library_and_targets(pd_t['s_obs'], pd_t['weeks'], 3, [], [])
            Phi_list.append(Phi_p)
            tgt_list.append(tgt_p[:, INC_COL:INC_COL + 1])
        except Exception:
            pass
        else:
            pass
        finally:
            pass
    else:
        pass
    if not Phi_list:
        return 0.5
    else:
        pass
    Phi_all = np.vstack(Phi_list)
    tgt_all = np.vstack(tgt_list).ravel()
    best_l1, best_score = (0.5, float('-inf'))
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    for l1 in FS_EN_L1_RATIOS:
        fold_scores = []
        for tr_idx, va_idx in kf.split(Phi_all):
            try:
                reg = ElasticNet(alpha=alpha, l1_ratio=l1, fit_intercept=False, max_iter=5000)
                reg.fit(Phi_all[tr_idx], tgt_all[tr_idx])
                pred = Phi_all[va_idx] @ reg.coef_
                fold_scores.append(float(r2_score(tgt_all[va_idx], pred)))
            except Exception:
                fold_scores.append(float('-inf'))
            else:
                pass
            finally:
                pass
        else:
            pass
        mean_sc = float(np.mean(fold_scores))
        if mean_sc > best_score:
            best_score, best_l1 = (mean_sc, l1)
        else:
            pass
    else:
        pass
    return best_l1

def run_funsearch(fs_tau=None, score_algo='ridge', bias_table=None, save_path=None):
    from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge
    _tau = fs_tau if fs_tau is not None else TAU0
    _alpha = _tau ** 2
    _bias = bias_table if bias_table is not None else {isl: (v[0], v[1]) for isl, v in ISLAND_BIAS_TABLE.items()}
    fs_train_pd = _build_province_dict(latent, FS_INNER_TRAIN)
    fs_val_pd = _build_province_dict(latent, FS_INNER_VAL)
    print(f'  [FunSearch] starting, {FS_N_ISLANDS} islands x {FS_ISLAND_SIZE} programs, {FS_N_ITERATIONS} iterations, tau={fs_tau}, score_algo={score_algo}')
    _l1_ratio = 0.5
    if score_algo == 'elasticnet':
        _l1_ratio = _tune_elasticnet_l1(fs_train_pd, _alpha)
    else:
        pass

    def _make_reg():
        if score_algo == 'ridge':
            return Ridge(alpha=_alpha, fit_intercept=False)
        elif score_algo == 'lasso':
            return Lasso(alpha=_alpha, fit_intercept=False, max_iter=10000)
        elif score_algo == 'elasticnet':
            return ElasticNet(alpha=_alpha, l1_ratio=_l1_ratio, fit_intercept=False, max_iter=10000)
        elif score_algo == 'bayesian_ridge':
            return BayesianRidge(fit_intercept=False)
        else:
            return Ridge(alpha=_alpha, fit_intercept=False)
    score_cache: Dict[str, float] = {}

    def score_program(prog: Program) -> float:
        fp = prog.fingerprint()
        if fp in score_cache:
            return score_cache[fp]
        else:
            pass
        try:
            n_lags = max(prog.n_lags, 1)
            Phi_list, tgt_list = ([], [])
            for prov, pd_t in fs_train_pd.items():
                Phi_p, tgt_p, _ = build_library_and_targets(pd_t['s_obs'], pd_t['weeks'], n_lags, prog.extra_terms, [])
                Phi_list.append(Phi_p)
                tgt_list.append(tgt_p[:, INC_COL:INC_COL + 1])
            else:
                pass
            Phi_all = np.vstack(Phi_list)
            tgt_all = np.vstack(tgt_list)
            reg = _make_reg()
            reg.fit(Phi_all, tgt_all.ravel())
            r2_vals = []
            for prov, pd_t in fs_val_pd.items():
                Phi_v, tgt_v, _ = build_library_and_targets(pd_t['s_obs'], pd_t['weeks'], n_lags, prog.extra_terms, [])
                if score_algo == 'bayesian_ridge':
                    pred = reg.predict(Phi_v)
                else:
                    pred = (Phi_v @ reg.coef_).ravel()
                pop = 1.0
                r2_vals.append((float(r2_score(tgt_v[:, INC_COL], pred)), pop))
            else:
                pass
            score = sum((r * p for r, p in r2_vals)) / sum((p for _, p in r2_vals))
        except Exception:
            score = float('-inf')
        else:
            pass
        finally:
            pass
        score_cache[fp] = score
        return score

    def mutate(prog: Program, island_id: int) -> Program:
        bias = _bias.get(island_id, (range(3, 6), 0))
        lag_range = bias[0]
        n_extra_bias = bias[1]
        child = copy.deepcopy(prog)
        child.program_id = str(uuid.uuid4())[:8]
        child.generation += 1
        child.parent_id = prog.program_id
        ops = ['add_term', 'remove_term', 'perturb_term', 'replace_term', 'retype_term', 'n_lags_up', 'n_lags_down', 'n_lags_jump']
        if random.random() < FS_DOUBLE_MUT:
            chosen = random.sample(ops, min(2, len(ops)))
        else:
            chosen = [random.choice(ops)]
        for op in chosen:
            if op == 'add_term' and len(child.extra_terms) < FS_MAX_EXTRA + n_extra_bias:
                child.extra_terms.append(ExtraTerm(random.choice(TERM_TYPES), {'lag': random.randint(0, child.n_lags), 'power': random.choice([2, 3]), 'z_idx': random.randint(0, LATENT_DIM - 1), 'z_idx1': random.randint(0, LATENT_DIM - 1), 'z_idx2': random.randint(0, LATENT_DIM - 1), 'lag_inc': random.randint(0, child.n_lags), 'lag_z': random.randint(0, child.n_lags), 'lag1': random.randint(0, child.n_lags), 'lag2': random.randint(0, child.n_lags)}))
            elif op == 'remove_term' and child.extra_terms:
                child.extra_terms.pop(random.randint(0, len(child.extra_terms) - 1))
            elif op == 'perturb_term' and child.extra_terms:
                idx = random.randint(0, len(child.extra_terms) - 1)
                t = child.extra_terms[idx]
                key = random.choice(list(t.params.keys()))
                if key in ('lag', 'lag_inc', 'lag_z', 'lag1', 'lag2'):
                    t.params[key] = random.randint(0, child.n_lags)
                elif key in ('z_idx', 'z_idx1', 'z_idx2'):
                    t.params[key] = random.randint(0, LATENT_DIM - 1)
                elif key == 'power':
                    t.params[key] = random.choice([2, 3])
                else:
                    pass
            elif op == 'replace_term' and child.extra_terms:
                idx = random.randint(0, len(child.extra_terms) - 1)
                child.extra_terms[idx] = ExtraTerm(random.choice(TERM_TYPES), {'lag': random.randint(0, child.n_lags), 'power': random.choice([2, 3]), 'z_idx': random.randint(0, LATENT_DIM - 1), 'z_idx1': random.randint(0, LATENT_DIM - 1), 'z_idx2': random.randint(0, LATENT_DIM - 1), 'lag_inc': random.randint(0, child.n_lags), 'lag_z': random.randint(0, child.n_lags), 'lag1': random.randint(0, child.n_lags), 'lag2': random.randint(0, child.n_lags)})
            elif op == 'retype_term' and child.extra_terms:
                idx = random.randint(0, len(child.extra_terms) - 1)
                child.extra_terms[idx].term_type = random.choice(TERM_TYPES)
            elif op == 'n_lags_up':
                child.n_lags = min(child.n_lags + 1, max(lag_range))
            elif op == 'n_lags_down':
                child.n_lags = max(child.n_lags - 1, min(lag_range))
            elif op == 'n_lags_jump':
                child.n_lags = random.choice(list(lag_range))
            else:
                pass
        else:
            pass
        return child
    islands = [[Program(n_lags=random.randint(3, 5)) for _ in range(FS_ISLAND_SIZE)] for _ in range(FS_N_ISLANDS)]
    for isl in islands:
        for prog in isl:
            prog.score = score_program(prog)
        else:
            pass
    else:
        pass
    history_rows = []
    all_programs = [copy.deepcopy(prog).to_dict() for isl in islands for prog in isl]
    for iteration in range(FS_N_ITERATIONS):
        if iteration == 0 or (iteration + 1) % 10 == 0 or iteration == FS_N_ITERATIONS - 1:
            _best_so_far = max((p.score for isl in islands for p in isl))
            print(f'    [FunSearch] iteration {iteration + 1}/{FS_N_ITERATIONS}  best_score={_best_so_far:+.4f}')
        else:
            pass
        for isl_idx, island in enumerate(islands):
            _weights = [max(0, p.score + 1) for p in island]
            if sum(_weights) <= 0:
                parent = random.choice(island)
            else:
                parent = random.choices(island, weights=_weights)[0]
            child = mutate(parent, isl_idx)
            if random.random() < FS_CROSSOVER and len(island) > 1:
                donor = random.choice([p for p in island if p.program_id != parent.program_id])
                n_swap = random.randint(1, max(1, len(donor.extra_terms)))
                donor_terms = copy.deepcopy(donor.extra_terms[:n_swap])
                child.extra_terms = child.extra_terms + donor_terms
                if len(child.extra_terms) > FS_MAX_EXTRA:
                    child.extra_terms = child.extra_terms[:FS_MAX_EXTRA]
                else:
                    pass
                history_rows.append({'event': 'crossover', 'island': isl_idx, 'iteration': iteration})
            else:
                pass
            child.score = score_program(child)
            history_rows.append({'event': 'mutate', 'island': isl_idx, 'iteration': iteration, 'score': child.score})
            all_programs.append(child.to_dict())
            worst = min(island, key=lambda p: p.score)
            if child.score > worst.score:
                island[island.index(worst)] = child
            else:
                pass
        else:
            pass
        if (iteration + 1) % FS_MIGRATE_EVERY == 0:
            print(f'    [FunSearch] iteration {iteration + 1}, migration round')
            for isl_idx in range(FS_N_ISLANDS):
                best = max(islands[isl_idx], key=lambda p: p.score)
                target = (isl_idx + 1) % FS_N_ISLANDS
                worst_in_target = min(islands[target], key=lambda p: p.score)
                if best.score > worst_in_target.score:
                    islands[target][islands[target].index(worst_in_target)] = copy.deepcopy(best)
                else:
                    pass
                history_rows.append({'event': 'migration', 'from': isl_idx, 'to': target, 'iteration': iteration})
            else:
                pass
        else:
            pass
    else:
        pass
    best = max((prog for isl in islands for prog in isl), key=lambda p: p.score)
    print(f'  [FunSearch] done, best score={best.score:+.4f}  n_lags={best.n_lags}  n_extra_terms={len(best.extra_terms)}')
    if save_path is not None:
        fs_json_out = save_path + '_best.json'
        fs_hist_out = save_path + '_history.csv'
        with open(fs_json_out, 'w') as f:
            json.dump(best.to_dict(), f, indent=2)
        hist_df = pd.DataFrame(history_rows)
        hist_df.to_csv(fs_hist_out, index=False)
    else:
        pass
    return (best, history_rows, all_programs)

def _run_one_grid_cell(run_label, fs_tau, score_algo, diversity_key, train_pd, val_pd, trainval_pd, test_pd, included_provinces, excluded_provinces, quality_df, grid_log_fh):
    import copy as _copy
    os.makedirs(FS_DIR, exist_ok=True)
    fs_save = os.path.join(FS_DIR, run_label.replace(' ', '_'))
    bias_table = {isl: (lr, nb) for isl, (lr, nb) in FS_GRID_DIVERSITY[diversity_key].items()}

    def _log(msg):
        print(msg)
        if grid_log_fh is not None:
            grid_log_fh.write(str(msg) + '\n')
            grid_log_fh.flush()
        else:
            pass
    _log(f"\n{'=' * 60}")
    _log(f'  GRID CELL: {run_label}')
    _log(f'  fs_tau={fs_tau}  score_algo={score_algo}  diversity={diversity_key}')
    _log(f"{'=' * 60}")
    fs_best, history_rows, all_programs = run_funsearch(fs_tau=fs_tau, score_algo=score_algo, bias_table=bias_table, save_path=fs_save)
    n_lags = max(fs_best.n_lags, 3)
    fs_terms = fs_best.extra_terms
    _log(f'  FS done: n_lags={n_lags}  n_extra={len(fs_terms)}  fs_score={fs_best.score:+.4f}')
    sample_prov = list(train_pd.keys())[0]
    sample_s = train_pd[sample_prov]['s_obs']
    sample_w = train_pd[sample_prov]['weeks']
    sample_aug = _build_aug(sample_s, n_lags)
    sample_weeks = sample_w[n_lags:]
    llm_formulas: list = []
    best_model_so_far = None
    best_combined = float('-inf')
    warm_g = warm_h = warm_u = None
    val_r2_last = float('nan')
    spectral_sc_last = float('nan')
    for llm_round in range(MAX_LLM_ROUNDS):
        model_train = run_hierarchical_em(train_pd, n_lags, fs_terms, llm_formulas, label=f'{run_label} Round {llm_round + 1}', g_init=warm_g, h_init=warm_h, u_init=warm_u)
        warm_g = model_train['g'].copy()
        warm_h = {cid: model_train['h'][cid].copy() for cid in CLUSTER_IDS}
        warm_u = {prov: model_train['u'][prov].copy() for prov in train_pd}
        val_r2, val_df, val_preds = evaluate_on_split(val_pd, model_train)
        spectral_sc = compute_spectral_score(val_preds)
        combined_sc = combined_score(val_r2, spectral_sc)
        val_r2_last = val_r2
        spectral_sc_last = spectral_sc
        active_terms, pruned_terms = get_active_and_pruned(model_train)
        cluster_r2 = {int(cid): float(val_df[val_df['cluster_id'] == cid]['r2'].mean()) for cid in val_df['cluster_id'].unique()}
        if combined_sc > best_combined:
            best_combined = combined_sc
            best_model_so_far = _copy.deepcopy(model_train)
        else:
            pass
        _log(f'  Round {llm_round + 1}: val_R2={val_r2:.4f}  spectral={spectral_sc:.4f}  combined={combined_sc:.4f}')
        prompt = build_llm_prompt(fs_best, history_rows, all_programs, val_r2, spectral_sc, 0.0, active_terms, pruned_terms, cluster_r2, llm_round + 1, [], [], [val_r2], compute_latent_covariate_meaning())
        raw_resp = query_llm(prompt)
        suggestions = parse_llm_suggestions(raw_resp)
        if not suggestions:
            _log(f'  No LLM suggestions at round {llm_round + 1}, stopping early.')
            break
        else:
            pass
        new_formulas = suggestions_to_formulas(suggestions, sample_aug, n_lags, sample_weeks)
        existing_names = {nm for nm, _ in llm_formulas}
        for nm, formula in new_formulas:
            if nm not in existing_names:
                llm_formulas.append((nm, formula))
            else:
                pass
        else:
            pass
    else:
        pass
    if best_model_so_far is None:
        best_model_so_far = model_train
    else:
        pass
    survived_names = {t for t, _ in get_active_and_pruned(best_model_so_far)[0] if t.startswith('llm_')}
    final_llm_formulas = [(nm, formula) for nm, formula in llm_formulas if f'llm_{nm}' in survived_names]
    if not final_llm_formulas:
        final_llm_formulas = llm_formulas
    else:
        pass
    model_final = run_hierarchical_em(trainval_pd, n_lags, fs_terms, final_llm_formulas, label=f'{run_label} FINAL on TRAIN+VAL')
    q_inc_by_prov = estimate_province_q_inc(model_final, trainval_pd)
    model_final['q_inc_by_prov'] = q_inc_by_prov
    test_r2, test_df_cell, test_preds_cell = evaluate_on_split(test_pd, model_final)
    test_spectral = compute_spectral_score(test_preds_cell)
    test_combined = combined_score(test_r2, test_spectral)
    test_df_incl = test_df_cell[test_df_cell['province'].isin(included_provinces)]
    test_r2_filt = float(test_df_incl['r2'].mean()) if len(test_df_incl) > 0 else float('nan')
    _log(f'\n  PER-PROVINCE TEST R2 ({run_label}):')
    for _, row in test_df_cell.sort_values('r2', ascending=False).iterrows():
        excl = '  EXCL' if row['province'] in excluded_provinces else ''
        _log(f"    {row['province']:35s}  R2={row['r2']:+.4f}{excl}")
    else:
        pass
    _log(f'\n  SUMMARY [{run_label}]:')
    _log(f'    test_R2        = {test_r2:.4f}')
    _log(f'    test_R2_filt   = {test_r2_filt:.4f}')
    _log(f'    test_spectral  = {test_spectral:.4f}')
    _log(f'    test_combined  = {test_combined:.4f}')
    _log(f'    val_R2 (best)  = {val_r2_last:.4f}')
    _log(f'    FS score       = {fs_best.score:+.4f}  n_lags={n_lags}')
    return {'run_label': run_label, 'fs_tau': fs_tau, 'score_algo': score_algo, 'diversity': diversity_key, 'fs_score': float(fs_best.score), 'fs_n_lags': int(n_lags), 'fs_n_extra': len(fs_terms), 'val_r2': float(val_r2_last), 'val_spectral': float(spectral_sc_last), 'val_combined': float(best_combined), 'test_r2': float(test_r2), 'test_spectral': float(test_spectral), 'test_combined': float(test_combined), 'test_r2_filt': float(test_r2_filt), 'llm_terms': len(final_llm_formulas)}
if __name__ == '__main__':
    train_pd = _build_province_dict(latent, TRAIN_YEARS)
    val_pd = _build_province_dict(latent, VAL_YEARS)
    trainval_pd = _build_province_dict(latent, TRAINVAL_YEARS)
    test_pd = _build_province_dict(latent, TEST_YEARS)
    quality_df = compute_province_quality_scores(train_pd)
    quality_df.to_csv(os.path.join(HERE, 'quality_scores.csv'), index=False)
    included_provinces = set()
    excluded_provinces = set()
    print('\n[STAGE] province quality gate')
    for _, row in quality_df.iterrows():
        prov = row['province']
        qs = row['quality_score']
        status = 'INCLUDE' if qs >= QUALITY_THRESHOLD else 'EXCLUDE'
        weak_tag = ' WEAK' if prov in WEAK_PROVINCES else ''
        print(f'  [quality] {prov:35s} Q={qs:.4f}  {status}{weak_tag}')
        if qs >= QUALITY_THRESHOLD:
            included_provinces.add(prov)
        else:
            excluded_provinces.add(prov)
    else:
        pass
    if excluded_provinces:
        print(f'  [quality] excluded from headline R2: {sorted(excluded_provinces)}')
    else:
        print('  [quality] no provinces excluded')
    overlap_weak_excluded = WEAK_PROVINCES & excluded_provinces
    _cluster_prov: Dict[int, List[str]] = {}
    for prov, pd_ in train_pd.items():
        cid = int(pd_.get('cluster', -1))
        _cluster_prov.setdefault(cid, []).append(prov)
    else:
        pass
    print('\n[STAGE] province clusters')
    for cid in sorted(_cluster_prov):
        _rows = sum((len(train_pd[p]['s_obs']) for p in _cluster_prov[cid]))
        print(f'  [cluster {cid}] {len(_cluster_prov[cid])} provinces, {_rows} training rows')
        for p in sorted(_cluster_prov[cid]):
            _n = len(train_pd[p]['s_obs'])
            _mean_inc = float(np.mean(np.expm1(train_pd[p]['s_obs'][:, INC_COL])))
            qs_val = float(quality_df[quality_df['province'] == p]['quality_score'].values[0]) if p in quality_df['province'].values else -1.0
            _is_weak = 'WEAK ' if p in WEAK_PROVINCES else '     '
            _excl = 'EXCL' if p in excluded_provinces else '    '
            print(f'      {p:35s} n={_n:4d}  mean_inc={_mean_inc:8.4f}  Q={qs_val:.4f}  {_is_weak}{_excl}')
        else:
            pass
    else:
        pass
    print(f'\n[STAGE] grid search starting, {len(FS_GRID_DIVERSITY) * len(FS_GRID_ALGOS) * len(FS_GRID_TAU)} cells total')
    grid_results = []
    os.makedirs(FS_DIR, exist_ok=True)
    total_cells = len(FS_GRID_DIVERSITY) * len(FS_GRID_ALGOS) * len(FS_GRID_TAU)
    done = 0
    with open(GRID_LOG, 'w', buffering=1) as _grid_log:
        for div_key in FS_GRID_DIVERSITY:
            for algo in FS_GRID_ALGOS:
                for tau in FS_GRID_TAU:
                    done += 1
                    label = f'tau{tau}_{algo}_{div_key}'
                    print(f'\n[GRID {done}/{total_cells}] {label}')
                    try:
                        row = _run_one_grid_cell(label, tau, algo, div_key, train_pd, val_pd, trainval_pd, test_pd, included_provinces, excluded_provinces, quality_df, _grid_log)
                    except Exception as _err:
                        import traceback
                        traceback.print_exc()
                        row = {'run_label': label, 'fs_tau': tau, 'score_algo': algo, 'diversity': div_key, 'fs_score': float('nan'), 'fs_n_lags': -1, 'fs_n_extra': -1, 'val_r2': float('nan'), 'val_spectral': float('nan'), 'val_combined': float('nan'), 'test_r2': float('nan'), 'test_spectral': float('nan'), 'test_combined': float('nan'), 'test_r2_filt': float('nan'), 'llm_terms': -1, 'error': str(_err)}
                    else:
                        pass
                    finally:
                        pass
                    grid_results.append(row)
                    pd.DataFrame(grid_results).to_csv(GRID_CSV, index=False)
                    print(f"[GRID {done}/{total_cells}] {label} done  val_r2={row.get('val_r2', float('nan')):.4f}  test_r2_filt={row.get('test_r2_filt', float('nan')):.4f}")
                else:
                    pass
            else:
                pass
        else:
            pass
    results_df = pd.DataFrame(grid_results)
    try:
        results_df_sorted = results_df.dropna(subset=['val_r2']).sort_values('val_r2', ascending=False)
    except Exception:
        results_df_sorted = results_df
    else:
        pass
    finally:
        pass
    header = f"  {'Label':30s}  {'tau':>4}  {'algo':>14}  {'div':>7}  {'FS_sc':>7}  {'val_R2':>7}  {'test_R2':>8}  {'test_R2_filt':>13}"
    print('\n' + '=' * 90)
    print('  GRID SEARCH COMPLETE, ranked by validation R2')
    print('=' * 90)
    print(header)
    for _, row in results_df_sorted.iterrows():
        print(f"  {row['run_label']:30s}  {row['fs_tau']:>4}  {row['score_algo']:>14}  {row['diversity']:>7}  {row['fs_score']:>7.4f}  {row['val_r2']:>7.4f}  {row['test_r2']:>8.4f}  {row['test_r2_filt']:>13.4f}")
    else:
        pass
    if len(results_df_sorted) > 0:
        _winner = results_df_sorted.iloc[0]
        print(f"\n  WINNER: {_winner['run_label']}  val_R2={_winner['val_r2']:.4f}  test_R2_filt={_winner['test_r2_filt']:.4f}")
        print("  Update forecast.py's WINNING_FS_TAU / WINNING_SCORE_ALGO / WINNING_DIVERSITY / WINNING_LABEL to match this cell.")
    else:
        pass
else:
    pass