import os, sys, json, copy, random, uuid, time, hashlib, warnings, math
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
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from utils import load_data
DATA_CSV = os.path.join(HERE, 'extended_input_normalized.csv')
LATENT_CSV = os.path.join(HERE, 'latents', 'latent_dim10.csv')
MODEL_PATH = os.path.join(HERE, 'models', 'best_model_dim10.pt')
CLUSTER_CSV = os.path.join(HERE, 'province_clusters.csv')
LOG_PATH = os.path.join(HERE, 'forecast_log.txt')
OUT_DIR = os.path.join(HERE, 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)
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
WINNING_FS_TAU = 4
WINNING_SCORE_ALGO = 'bayesian_ridge'
WINNING_DIVERSITY = 'spread'
WINNING_LABEL = 'tau4_bayesian_ridge_spread'
SAVED_FS_JSON = os.path.join(HERE, 'fs_programs', f'{WINNING_LABEL}_best.json')
USE_SAVED_FS_PROGRAM = True
FS_N_ISLANDS = 4
FS_ISLAND_SIZE = 5
FS_N_ITERATIONS = 80
FS_MIGRATE_EVERY = 10
FS_MAX_EXTRA = 8
FS_DOUBLE_MUT = 0.15
FS_CROSSOVER = 0.2
FS_RANDOM_SEED = 42
FS_INNER_TRAIN = [2015, 2016, 2017]
FS_INNER_VAL = [2018, 2019]
FS_EN_L1_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9]
FS_GRID_DIVERSITY = {'spread': {0: (range(3, 5), 0), 1: (range(5, 6), 0), 2: (range(6, 9), 0), 3: (range(4, 6), 3)}, 'narrow': {0: (range(3, 5), 0), 1: (range(4, 6), 0), 2: (range(5, 7), 0), 3: (range(6, 8), 0)}}
TAU0 = 10.0
SIGMA_H_FIXED = 0.05
SIGMA_U_FIXED = 0.1
N_EM = 50
KAPPA = 5.0
Q_INIT_Z = 0.001
Q_INIT_INC = 0.05
R_INIT_Z = 0.0001
R_INIT_INC = 0.03
ADAPT_LASSO_ITERS = 5
ADAPT_LASSO_GAMMA = 0.5
LASSO_HARD_ZERO = 0.005
ACTIVE_THRESH = 0.01
Q_INC_CLIP_LO = Q_INIT_INC * 0.1
Q_INC_CLIP_HI = Q_INIT_INC * 20.0
OLLAMA_URL = 'http://localhost:11434/api/chat'
OLLAMA_MODEL = 'mistral:latest'
MAX_LLM_ROUNDS = 1
R2_WEIGHT, SPECTRAL_WEIGHT = (0.65, 0.35)
KEY_PERIODS, SPECTRAL_BIN_WINDOW = ([52, 26], 2)
ISLAND_BIAS_TABLE = {0: (range(3, 5), 0, 'lasso'), 1: (range(5, 6), 0, 'ridge'), 2: (range(6, 9), 0, 'elasticnet'), 3: (range(4, 6), 3, 'bayesian_ridge'), 4: (range(3, 5), 0, 'ridge'), 5: (range(5, 6), 0, 'lasso'), 6: (range(6, 9), 2, 'elasticnet'), 7: (range(4, 6), 3, 'ridge'), 8: (range(3, 5), 0, 'lasso'), 9: (range(5, 7), 1, 'bayesian_ridge')}
TERM_TYPES = ['inc_pow', 'inc_z_prod', 'cross_inc', 'z_sq', 'z_prod', 'inc_diff', 'z_cube', 'z_inc_ratio']
FORECAST_HORIZON = 8
N_SIM_PATHS = 500
CI_LOWER_PCT = 5.0
CI_UPPER_PCT = 95.0
OUTBREAK_PERCENTILE = 90.0
ALERT_PROB_THRESHOLD = 0.5
BACKTEST_STRIDE = 4
TOP_DRIVER_TERMS = 8
TOP_RISK_PROVINCES = 8
RELIABILITY_HIGH_Q, RELIABILITY_HIGH_R2 = (0.5, 0.4)
RELIABILITY_MED_Q, RELIABILITY_MED_R2 = (0.2, 0.15)
N_STABILITY_RUNS = 5
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

def detect_data_cutoff(df: pd.DataFrame, case_col: str='total_cases') -> Optional[Tuple[int, int]]:
    weekly = df.groupby(['year', 'week'])[case_col].sum().reset_index()
    weekly = weekly.sort_values(['year', 'week']).reset_index(drop=True)
    idx = len(weekly) - 1
    while idx >= 0 and weekly.loc[idx, case_col] == 0:
        idx -= 1
    else:
        pass
    if idx < 0:
        return None
    else:
        pass
    return (int(weekly.loc[idx, 'year']), int(weekly.loc[idx, 'week']))
raw, INPUT_FEATURES = load_data(DATA_CSV, TRAIN_YEARS)
INPUT_DIM = len(INPUT_FEATURES)
DATA_CUTOFF = detect_data_cutoff(raw)
if DATA_CUTOFF is not None:
    CUTOFF_YEAR, CUTOFF_WEEK = DATA_CUTOFF
    _full_weeks = raw.groupby(['year', 'week']).ngroups
    _dropped = raw[(raw['year'] > CUTOFF_YEAR) | (raw['year'] == CUTOFF_YEAR) & (raw['week'] > CUTOFF_WEEK)]
    _n_dropped_weeks = _dropped[['year', 'week']].drop_duplicates().shape[0]
    if _n_dropped_weeks > 0:
        pass
    else:
        pass
else:
    CUTOFF_YEAR, CUTOFF_WEEK = (None, None)
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
if DATA_CUTOFF is not None:
    _keep = (latent['year'] < CUTOFF_YEAR) | (latent['year'] == CUTOFF_YEAR) & (latent['week'] <= CUTOFF_WEEK)
    _n_before = len(latent)
    latent = latent[_keep].reset_index(drop=True)
else:
    pass
CLUSTER_IDS = sorted(latent['cluster_id'].dropna().unique().astype(int))
STATE_COLS = z_cols + ['incidence']
STATE_DIM = len(STATE_COLS)
INC_COL = STATE_DIM - 1
try:
    POP_LOOKUP = raw[['province', 'year', 'population']].drop_duplicates(subset=['province', 'year']).set_index(['province', 'year'])['population'].to_dict()
    HAS_POPULATION = True
except Exception:
    POP_LOOKUP = {}
    HAS_POPULATION = False
else:
    pass
finally:
    pass
TRAINVAL_YEARS = TRAIN_YEARS + VAL_YEARS
trainval_mask = latent['year'].isin(TRAINVAL_YEARS)
SCALER = StandardScaler()
SCALER.fit(latent.loc[trainval_mask, STATE_COLS].values.astype(np.float32))
FS_SCALER = StandardScaler()
FS_SCALER.fit(latent.loc[latent['year'].isin(FS_INNER_TRAIN), STATE_COLS].values.astype(np.float32))

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

    @staticmethod
    def from_dict(d: Dict) -> 'Program':
        return Program(program_id=d.get('program_id', str(uuid.uuid4())[:8]), n_lags=d['n_lags'], extra_terms=[ExtraTerm.from_dict(t) for t in d.get('extra_terms', [])], score=d.get('score', float('-inf')), generation=d.get('generation', 0), parent_id=d.get('parent_id', 'seed'))

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
        lo, hi = (df[col].min(), df[col].max())
        df[f'{col}_norm'] = (df[col] - lo) / (hi - lo) if hi > lo else 0.5
    else:
        pass
    df['quality_score'] = QUALITY_W_COMPLETE * df['completeness_norm'] + QUALITY_W_SMOOTH * df['smoothness_norm'] + QUALITY_W_STABLE * df['stability_norm'] + QUALITY_W_CONSIST * df['consistency_norm']
    return df.sort_values('quality_score', ascending=False).reset_index(drop=True)

def run_hierarchical_em(province_data: Dict, n_lags: int, fs_terms: List[ExtraTerm], llm_formulas: List[Tuple[str, str]], label: str='hierarchical', g_init=None, h_init=None, u_init=None, use_cluster: bool=True, use_province: bool=True) -> Dict:
    provinces = list(province_data.keys())
    q_diag = np.array([Q_INIT_Z] * LATENT_DIM + [Q_INIT_INC], dtype=np.float32)
    r_diag = np.array([R_INIT_Z] * LATENT_DIM + [R_INIT_INC], dtype=np.float32)
    sigma_h, sigma_u = (SIGMA_H_FIXED, SIGMA_U_FIXED)
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
        beta_by_prov = {prov: g + h[province_data[prov]['cluster']] + u[prov] for prov in provinces}
        results = Parallel(n_jobs=N_JOBS)((delayed(_smooth_one)(prov, province_data[prov]['s_obs'], province_data[prov]['weeks'], beta_by_prov[prov], x_smooth_all[prov], q_diag, r_diag, n_lags, fs_terms, llm_formulas) for prov in provinces))
        for prov, x_s in results:
            x_smooth_all[prov] = x_s
        else:
            pass
        Phi_by_prov, tgt_by_prov = ({}, {})
        for prov in provinces:
            Phi_p, tgt_p, _ = build_library_and_targets(x_smooth_all[prov], province_data[prov]['weeks'], n_lags, fs_terms, llm_formulas)
            Phi_by_prov[prov] = Phi_p
            tgt_by_prov[prov] = tgt_p
        else:
            pass
        Phi_pool, res_pool = ([], [])
        for prov in provinces:
            cid = province_data[prov]['cluster']
            Phi_p, tgt_p = (Phi_by_prov[prov], tgt_by_prov[prov])
            res_p = tgt_p - Phi_p @ h[cid] - Phi_p @ u[prov]
            Phi_pool.append(Phi_p)
            res_pool.append(res_p)
        else:
            pass
        g = mstep_global(np.vstack(Phi_pool), np.vstack(res_pool))
        if use_cluster:
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
        else:
            pass
        if use_province:
            for prov in provinces:
                cid = province_data[prov]['cluster']
                Phi_p = Phi_by_prov[prov]
                res_p = tgt_by_prov[prov] - Phi_p @ g - Phi_p @ h[cid]
                u[prov] = mstep_province(Phi_p, res_p, sigma_u)
            else:
                pass
        else:
            pass
        if em_iter % 10 == 0 or em_iter == N_EM - 1:
            active = int((np.abs(g) > ACTIVE_THRESH).any(axis=1).sum())
        else:
            pass
    else:
        pass
    return dict(g=g, h=h, u=u, q_diag=q_diag, r_diag=r_diag, term_names=term_names, sigma_h=sigma_h, sigma_u=sigma_u, n_lags=n_lags, fs_terms=fs_terms, llm_formulas=llm_formulas)

def estimate_province_q_inc(model: Dict, province_data: Dict) -> Dict[str, float]:
    n_lags, fs_terms, llm_formulas = (model['n_lags'], model['fs_terms'], model['llm_formulas'])
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

def forecast_sequential(s_test, weeks_test, beta_p, q_diag, r_diag, n_lags, fs_terms, llm_formulas) -> np.ndarray:
    T, d = (len(s_test), STATE_DIM)
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
    n_lags, fs_terms, llm_formulas = (model['n_lags'], model['fs_terms'], model['llm_formulas'])
    mean_inc = float(SCALER.mean_[INC_COL])
    std_inc = float(SCALER.scale_[INC_COL])
    q_inc_by_prov = model.get('q_inc_by_prov', {})
    results, prov_preds = ([], {})
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
            yhat_cnt, ytrue_cnt = (np.expm1(yhat_log), np.expm1(ytrue_log))
        else:
            yhat_cnt, ytrue_cnt = (yhat_log, ytrue_log)
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

def compute_spectral_score(prov_preds, key_periods=KEY_PERIODS, bin_window=SPECTRAL_BIN_WINDOW) -> float:
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
        ytrue_w, resid_w = (ytrue_dm * window, resid_dm * window)
        N_pad = int(np.ceil(max(8 * N, 4 * lcm_periods) / lcm_periods) * lcm_periods)
        fft_true = np.fft.rfft(ytrue_w, n=N_pad)
        fft_res = np.fft.rfft(resid_w, n=N_pad)
        pwr_true, pwr_res = (np.abs(fft_true) ** 2, np.abs(fft_res) ** 2)
        band_scores = []
        for period in key_periods:
            bin_center = N_pad // period
            bin_lo, bin_hi = (max(1, bin_center - bin_window), min(len(pwr_true) - 1, bin_center + bin_window))
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
        yrs = grp['year'].values.astype(np.int32)
        cid = int(grp['cluster_id'].iloc[0])
        out[prov] = {'s_obs': s_sc, 'weeks': weeks, 'years': yrs, 'cluster': cid}
    else:
        pass
    return out

def get_active_and_pruned(model: Dict) -> Tuple[List[Tuple[str, float]], List[str]]:
    g = model['g']
    names = model['term_names']
    g_inc = g[:, INC_COL] if g.ndim == 2 else g
    active = [(nm, float(coef)) for nm, coef in zip(names, g_inc) if abs(float(coef)) > ACTIVE_THRESH]
    pruned = [nm for nm, coef in zip(names, g_inc) if abs(float(coef)) <= ACTIVE_THRESH]
    active.sort(key=lambda x: abs(x[1]), reverse=True)
    return (active, pruned)

def build_llm_prompt(fs_best, val_r2, spectral_sc, active_terms, pruned_terms, cluster_r2, llm_round, val_r2_history) -> str:
    if val_r2_history:
        trend_parts = [f'Round {i + 1}: {r:.4f}' for i, r in enumerate(val_r2_history)]
        trend_str = '  ' + '  |  '.join(trend_parts)
    else:
        trend_str = '  (first round, no history yet)'
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
    pruned_str = f'{n_pruned} terms total dropped (coef < {ACTIVE_THRESH})'
    cluster_str = '\n'.join((f'  Cluster {cid}: R2={r2:.4f}' for cid, r2 in sorted(cluster_r2.items())))
    return f'You are an expert epidemiologist and dynamical systems researcher collaborating on a SINDy (Sparse Identification of Nonlinear Dynamics) model for dengue forecasting.\n\nYOUR ROLE IN THE PIPELINE:\nYou are proposing candidate library terms for the dI/dt equation in a hierarchical SINDy model. The model structure is W_p = g + h_c + u_p where g is a global SINDy coefficient matrix shared by all 28 provinces, h_c is a small cluster correction, and u_p is a tiny province correction. You are improving g specifically for the incidence (dI/dt) dimension. After you propose terms, they are added to the regression library and the EM solver refits g on training data 2015-2018. Validation is then run on 2019 data only. Terms whose fitted coefficient falls below {ACTIVE_THRESH} in absolute value are dropped automatically. Only propose terms that have a strong biological or mathematical justification for a non-trivial coefficient.\n\nBIOLOGICAL CONTEXT FOR TERM DESIGN:\nDengue transmission in the Dominican Republic follows a seasonal pattern driven by Aedes aegypti breeding cycles tied to rainfall and temperature with a 2 to 4 week lag, an 8 to 12 day extrinsic incubation period, herd immunity and susceptible depletion after large outbreaks, and either explosive growth or fade near outbreak thresholds.\n\nVAL R2 TREND ACROSS ROUNDS (training 2015-2018, validation 2019):\n{trend_str}\n\nCURRENT MODEL STATE (round {llm_round}):\nValidation R2 = {val_r2:.4f}  |  Spectral score = {spectral_sc:.4f}\n\nPer-cluster validation R2:\n{cluster_str}\nNote: Cluster 3 (Azua, Peravia, San Jose de Ocoa, mountain provinces) is the hardest, with irregular year-to-year outbreak timing.\n\nACTIVE TERMS IN THE CURRENT INCIDENCE EQUATION (sorted by coefficient magnitude):\n{active_str}\n\nTerms dropped this round (|coef| < {ACTIVE_THRESH}):\n  {pruned_str}\n\nYOUR TASK:\nSuggest exactly 8 new candidate library terms grounded in dengue biology. Each formula must be a valid numpy expression using ONLY the variables below. Do NOT use any named constants (threshold, beta, K, R0, mu) or any variable not in the list. Every formula must evaluate to a numpy array of the same shape as the input arrays.\n\nBiological priorities in order: climate-lag interactions (seasonal signal times a lagged z-dimension), incidence-environment couplings at 1 to 3 week lags, susceptible depletion proxies, secondary 26-week seasonality, and outbreak nonlinearity (squared incidence or incidence times a climate z). Do NOT propose pure polynomial interactions of z-dimensions with no incidence involvement.\n\nRespond ONLY with a valid JSON array. No text before or after. Example:\n[\n  {{"name": "sin52_z3_lag2", "formula": "np.sin(2*np.pi*week_arr/52) * z3_lag2", "reason": "seasonal modulation by climate lag 2 weeks"}},\n  {{"name": "y_z2_lag2", "formula": "y_lag1 * z2_lag2", "reason": "incidence times climate forcing at 2-week lag"}}\n]\n\nCOMPLETE LIST of available variables. Use ONLY these:\ny_lag0, y_lag1, y_lag2, y_lag3\nz1_lag0, z2_lag0, z3_lag0, z4_lag0, z5_lag0, z6_lag0, z7_lag0, z8_lag0, z9_lag0, z10_lag0\nz1_lag1, z2_lag1, z3_lag1, z4_lag1, z5_lag1, z6_lag1, z7_lag1, z8_lag1, z9_lag1, z10_lag1\nz1_lag2, z2_lag2, z3_lag2, z4_lag2, z5_lag2, z6_lag2, z7_lag2, z8_lag2, z9_lag2, z10_lag2\nz1_lag3, z2_lag3, z3_lag3, z4_lag3, z5_lag3, z6_lag3, z7_lag3, z8_lag3, z9_lag3, z10_lag3\nweek_arr, np\n\nDo NOT use: threshold, beta, K, K1, K2, alpha, gamma, R0, mu, or any name not in the list.'

def query_llm(prompt: str) -> str:
    payload = {'model': OLLAMA_MODEL, 'messages': [{'role': 'user', 'content': prompt}], 'stream': False, 'options': {'temperature': 0.4, 'num_predict': 1200}}
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
        resp.raise_for_status()
        result = resp.json().get('message', {}).get('content', '')
        return result
    except Exception as e:
        return ''
    else:
        pass
    finally:
        pass

def parse_llm_suggestions(raw_text: str) -> List[Dict]:
    try:
        start = raw_text.find('[')
        end = raw_text.rfind(']') + 1
        if start == -1 or end == 0:
            return []
        else:
            pass
        suggestions = json.loads(raw_text[start:end])
        return suggestions
    except Exception as e:
        return []
    else:
        pass
    finally:
        pass

def suggestions_to_formulas(suggestions, X_aug_sample, n_lags, week_arr) -> List[Tuple[str, str]]:
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
        name, formula, reason = (sug.get('name', ''), sug.get('formula', ''), sug.get('reason', ''))
        if not name or not formula:
            continue
        else:
            pass
        try:
            local = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in var_map.items()}
            result = np.array(eval(formula, {'__builtins__': {}}, local), dtype=np.float32).reshape(-1)
            if result.shape[0] != X_aug_sample.shape[0] or not np.all(np.isfinite(result)):
                continue
            else:
                pass
            local1 = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in single_map.items()}
            result1 = np.array(eval(formula, {'__builtins__': {}}, local1), dtype=np.float32).reshape(-1)
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

def _tune_elasticnet_l1(fs_train_pd, alpha):
    from sklearn.linear_model import ElasticNet
    from sklearn.model_selection import KFold
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
    Phi_all, tgt_all = (np.vstack(Phi_list), np.vstack(tgt_list).ravel())
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

def run_funsearch(fs_tau=None, score_algo='ridge', bias_table=None):
    from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge
    _tau = fs_tau if fs_tau is not None else TAU0
    _alpha = _tau ** 2
    _bias = bias_table if bias_table is not None else {isl: (v[0], v[1]) for isl, v in ISLAND_BIAS_TABLE.items()}
    fs_train_pd = _build_province_dict(latent, FS_INNER_TRAIN)
    fs_val_pd = _build_province_dict(latent, FS_INNER_VAL)
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
            pass
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
            Phi_all, tgt_all = (np.vstack(Phi_list), np.vstack(tgt_list))
            reg = _make_reg()
            reg.fit(Phi_all, tgt_all.ravel())
            r2_vals = []
            for prov, pd_t in fs_val_pd.items():
                Phi_v, tgt_v, _ = build_library_and_targets(pd_t['s_obs'], pd_t['weeks'], n_lags, prog.extra_terms, [])
                pred = reg.predict(Phi_v) if score_algo == 'bayesian_ridge' else (Phi_v @ reg.coef_).ravel()
                r2_vals.append(float(r2_score(tgt_v[:, INC_COL], pred)))
            else:
                pass
            score = float(np.mean(r2_vals))
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
        lag_range, n_extra_bias = (bias[0], bias[1])
        child = copy.deepcopy(prog)
        child.program_id = str(uuid.uuid4())[:8]
        child.generation += 1
        child.parent_id = prog.program_id
        ops = ['add_term', 'remove_term', 'perturb_term', 'replace_term', 'retype_term', 'n_lags_up', 'n_lags_down', 'n_lags_jump']
        chosen = random.sample(ops, min(2, len(ops))) if random.random() < FS_DOUBLE_MUT else [random.choice(ops)]
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
    for iteration in range(FS_N_ITERATIONS):
        for isl_idx, island in enumerate(islands):
            parent = random.choices(island, weights=[max(0, p.score + 1) for p in island])[0]
            child = mutate(parent, isl_idx)
            if random.random() < FS_CROSSOVER and len(island) > 1:
                donor = random.choice([p for p in island if p.program_id != parent.program_id])
                n_swap = random.randint(1, max(1, len(donor.extra_terms)))
                child.extra_terms = child.extra_terms + copy.deepcopy(donor.extra_terms[:n_swap])
                if len(child.extra_terms) > FS_MAX_EXTRA:
                    child.extra_terms = child.extra_terms[:FS_MAX_EXTRA]
                else:
                    pass
            else:
                pass
            child.score = score_program(child)
            worst = min(island, key=lambda p: p.score)
            if child.score > worst.score:
                island[island.index(worst)] = child
            else:
                pass
        else:
            pass
        if (iteration + 1) % FS_MIGRATE_EVERY == 0:
            for isl_idx in range(FS_N_ISLANDS):
                best = max(islands[isl_idx], key=lambda p: p.score)
                target = (isl_idx + 1) % FS_N_ISLANDS
                worst_in_target = min(islands[target], key=lambda p: p.score)
                if best.score > worst_in_target.score:
                    islands[target][islands[target].index(worst_in_target)] = copy.deepcopy(best)
                else:
                    pass
            else:
                pass
        else:
            pass
    else:
        pass
    best = max((prog for isl in islands for prog in isl), key=lambda p: p.score)
    return best

def fit_winning_model(train_pd, val_pd, trainval_pd) -> Dict:
    if USE_SAVED_FS_PROGRAM and os.path.exists(SAVED_FS_JSON):
        with open(SAVED_FS_JSON) as f:
            fs_best = Program.from_dict(json.load(f))
    else:
        bias_table = {isl: (lr, nb) for isl, (lr, nb) in FS_GRID_DIVERSITY[WINNING_DIVERSITY].items()}
        fs_best = run_funsearch(fs_tau=WINNING_FS_TAU, score_algo=WINNING_SCORE_ALGO, bias_table=bias_table)
    n_lags, fs_terms = (max(fs_best.n_lags, 3), fs_best.extra_terms)
    sample_prov = list(train_pd.keys())[0]
    sample_aug = _build_aug(train_pd[sample_prov]['s_obs'], n_lags)
    sample_weeks = train_pd[sample_prov]['weeks'][n_lags:]
    llm_formulas: List[Tuple[str, str]] = []
    model_train = None
    for llm_round in range(MAX_LLM_ROUNDS):
        model_train = run_hierarchical_em(train_pd, n_lags, fs_terms, llm_formulas, label=f'{WINNING_LABEL} Round {llm_round + 1}')
        val_r2, val_df, val_preds = evaluate_on_split(val_pd, model_train)
        spectral_sc = compute_spectral_score(val_preds)
        active_terms, pruned_terms = get_active_and_pruned(model_train)
        cluster_r2 = {int(cid): float(val_df[val_df['cluster_id'] == cid]['r2'].mean()) for cid in val_df['cluster_id'].unique()}
        prompt = build_llm_prompt(fs_best, val_r2, spectral_sc, active_terms, pruned_terms, cluster_r2, llm_round + 1, [val_r2])
        raw_resp = query_llm(prompt)
        suggestions = parse_llm_suggestions(raw_resp)
        if not suggestions:
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
    survived_names = {t for t, _ in get_active_and_pruned(model_train)[0] if t.startswith('llm_')}
    final_llm_formulas = [(nm, f) for nm, f in llm_formulas if f'llm_{nm}' in survived_names]
    if not final_llm_formulas and llm_formulas:
        final_llm_formulas = llm_formulas
    else:
        pass
    model_final = run_hierarchical_em(trainval_pd, n_lags, fs_terms, final_llm_formulas, label=f'{WINNING_LABEL} FINAL on TRAIN+VAL')
    model_final['q_inc_by_prov'] = estimate_province_q_inc(model_final, trainval_pd)
    model_final['llm_formulas_used'] = final_llm_formulas
    return model_final
MANUAL_EXPERT_FORMULAS: List[Tuple[str, str]] = [('manual_seasonal_climate_lag2', 'np.sin(2*np.pi*week_arr/52) * z1_lag2'), ('manual_depletion', 'y_lag0 * (1.0 - y_lag1)'), ('manual_climate_coupling_lag3', 'y_lag1 * z4_lag3'), ('manual_semiannual_incidence', 'np.sin(2*np.pi*week_arr/26) * y_lag1'), ('manual_outbreak_nonlinear', 'y_lag1 ** 2'), ('manual_climate_concurrent', 'y_lag0 * z2_lag0')]

def generate_random_valid_terms(n: int, n_lags: int, seed: int) -> List[ExtraTerm]:
    rng = random.Random(seed)
    terms = []
    for _ in range(n):
        terms.append(ExtraTerm(rng.choice(TERM_TYPES), {'lag': rng.randint(0, n_lags), 'power': rng.choice([2, 3]), 'z_idx': rng.randint(0, LATENT_DIM - 1), 'z_idx1': rng.randint(0, LATENT_DIM - 1), 'z_idx2': rng.randint(0, LATENT_DIM - 1), 'lag_inc': rng.randint(0, n_lags), 'lag_z': rng.randint(0, n_lags), 'lag1': rng.randint(0, n_lags), 'lag2': rng.randint(0, n_lags)}))
    else:
        pass
    return terms

def _fit_and_score(label, train_pd, val_pd, trainval_pd, test_pd, n_lags, fs_terms, llm_formulas, included_provinces) -> Tuple[Dict, Dict]:
    model = run_hierarchical_em(trainval_pd, n_lags, fs_terms, llm_formulas, label=label)
    model['q_inc_by_prov'] = estimate_province_q_inc(model, trainval_pd)
    val_r2, _, _ = evaluate_on_split(val_pd, model)
    test_r2, test_df, _ = evaluate_on_split(test_pd, model)
    test_incl = test_df[test_df['province'].isin(included_provinces)]
    test_r2_filt = float(test_incl['r2'].mean()) if len(test_incl) else float('nan')
    n_terms = len(llm_formulas) if llm_formulas else 0
    return (model, {'variant': label, 'n_extra_terms': n_terms, 'val_r2': val_r2, 'test_r2': test_r2, 'test_r2_filt': test_r2_filt})

def run_llm_ablation(model_final: Dict, train_pd, val_pd, trainval_pd, test_pd, included_provinces) -> pd.DataFrame:
    n_lags = model_final['n_lags']
    fs_terms = model_final['fs_terms']
    llm_used = model_final['llm_formulas_used']
    n_llm = len(llm_used) if llm_used else 3
    rows = []
    _, r = _fit_and_score('ablation_no_llm', train_pd, val_pd, trainval_pd, test_pd, n_lags, fs_terms, [], included_provinces)
    rows.append(r)
    val_r2_full, _, _ = evaluate_on_split(val_pd, model_final)
    test_r2_full, test_df_full, _ = evaluate_on_split(test_pd, model_final)
    test_incl_full = test_df_full[test_df_full['province'].isin(included_provinces)]
    rows.append({'variant': 'with_llm', 'n_extra_terms': n_llm, 'val_r2': val_r2_full, 'test_r2': test_r2_full, 'test_r2_filt': float(test_incl_full['r2'].mean()) if len(test_incl_full) else float('nan')})
    random_terms = generate_random_valid_terms(n_llm, n_lags, seed=FS_RANDOM_SEED + 1)
    _, r = _fit_and_score('ablation_random_terms', train_pd, val_pd, trainval_pd, test_pd, n_lags, fs_terms + random_terms, [], included_provinces)
    r['n_extra_terms'] = n_llm
    rows.append(r)
    _, r = _fit_and_score('ablation_manual_terms', train_pd, val_pd, trainval_pd, test_pd, n_lags, fs_terms, MANUAL_EXPERT_FORMULAS, included_provinces)
    rows.append(r)
    df = pd.DataFrame(rows)
    for _, r in df.iterrows():
        pass
    else:
        pass
    return df

def run_hierarchy_ablation(model_final: Dict, train_pd, val_pd, trainval_pd, test_pd, included_provinces) -> pd.DataFrame:
    n_lags = model_final['n_lags']
    fs_terms = model_final['fs_terms']
    llm_formulas = model_final['llm_formulas_used']
    configs = [('global_only', False, False), ('global_plus_cluster', True, False), ('global_plus_province', False, True)]
    rows = []
    for i, (label, use_c, use_p) in enumerate(configs, start=1):
        model = run_hierarchical_em(trainval_pd, n_lags, fs_terms, llm_formulas, label=f'ablation_{label}', use_cluster=use_c, use_province=use_p)
        model['q_inc_by_prov'] = estimate_province_q_inc(model, trainval_pd)
        val_r2, _, _ = evaluate_on_split(val_pd, model)
        test_r2, test_df, _ = evaluate_on_split(test_pd, model)
        test_incl = test_df[test_df['province'].isin(included_provinces)]
        rows.append({'variant': label, 'val_r2': val_r2, 'test_r2': test_r2, 'test_r2_filt': float(test_incl['r2'].mean()) if len(test_incl) else float('nan')})
    else:
        pass
    val_r2_full, _, _ = evaluate_on_split(val_pd, model_final)
    test_r2_full, test_df_full, _ = evaluate_on_split(test_pd, model_final)
    test_incl_full = test_df_full[test_df_full['province'].isin(included_provinces)]
    rows.append({'variant': 'global_plus_cluster_plus_province (full)', 'val_r2': val_r2_full, 'test_r2': test_r2_full, 'test_r2_filt': float(test_incl_full['r2'].mean()) if len(test_incl_full) else float('nan')})
    df = pd.DataFrame(rows)
    for _, r in df.iterrows():
        pass
    else:
        pass
    return df

def build_full_history(latent_df: pd.DataFrame) -> Dict[str, Dict[Tuple[int, int], float]]:
    hist: Dict[str, Dict[Tuple[int, int], float]] = {}
    for prov, grp in latent_df.groupby('province'):
        d = {}
        for _, row in grp.iterrows():
            log_val = float(row['incidence'])
            raw = float(np.expm1(log_val)) if LOG_INCIDENCE else log_val
            d[int(row['year']), int(row['week'])] = max(raw, 0.0)
        else:
            pass
        hist[prov] = d
    else:
        pass
    return hist

def evaluate_naive_baselines(test_pd: Dict, full_history: Dict, n_lags: int, included_provinces) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    all_true_p, all_pred_p, all_true_s, all_pred_s = ([], [], [], [])
    for prov, pd_t in test_pd.items():
        if prov not in included_provinces:
            continue
        else:
            pass
        s_obs, weeks_arr, years_arr = (pd_t['s_obs'], pd_t['weeks'], pd_t['years'])
        T = len(s_obs)
        if T <= n_lags + 1:
            continue
        else:
            pass
        raw_inc = _inc_scaled_to_raw(s_obs[:, INC_COL])
        yt_p, yp_p, yt_s, yp_s = ([], [], [], [])
        for t in range(n_lags, T - 1):
            y_true = float(raw_inc[t + 1])
            yt_p.append(y_true)
            yp_p.append(float(raw_inc[t]))
            key = (int(years_arr[t + 1]) - 1, int(weeks_arr[t + 1]))
            y_season = full_history.get(prov, {}).get(key)
            if y_season is not None:
                yt_s.append(y_true)
                yp_s.append(y_season)
            else:
                pass
        else:
            pass
        r2p = float(r2_score(yt_p, yp_p)) if len(yt_p) > 1 else float('nan')
        r2s = float(r2_score(yt_s, yp_s)) if len(yt_s) > 1 else float('nan')
        rows.append({'province': prov, 'persistence_r2': r2p, 'seasonal_naive_r2': r2s, 'n_points': len(yt_p), 'n_points_seasonal': len(yt_s)})
        all_true_p.extend(yt_p)
        all_pred_p.extend(yp_p)
        all_true_s.extend(yt_s)
        all_pred_s.extend(yp_s)
    else:
        pass
    df = pd.DataFrame(rows)
    pooled_p = float(r2_score(all_true_p, all_pred_p)) if len(all_true_p) > 1 else float('nan')
    pooled_s = float(r2_score(all_true_s, all_pred_s)) if len(all_true_s) > 1 else float('nan')
    summary = pd.DataFrame([{'variant': 'persistence (y_hat_t+1 = y_t)', 'test_r2_pooled': pooled_p, 'test_r2_mean_per_province': float(df['persistence_r2'].mean())}, {'variant': 'seasonal_naive (y_hat_t+1 = y_t+1_minus_52wk)', 'test_r2_pooled': pooled_s, 'test_r2_mean_per_province': float(df['seasonal_naive_r2'].mean())}])
    return (df, summary)

def _beta_for(model: Dict, prov: str, cid: int) -> np.ndarray:
    return model['g'] + model['h'].get(cid, 0) + model['u'].get(prov, 0)

def _inc_scaled_to_raw(x_inc_scaled: np.ndarray) -> np.ndarray:
    mean_inc = float(SCALER.mean_[INC_COL])
    std_inc = float(SCALER.scale_[INC_COL])
    log_val = x_inc_scaled * std_inc + mean_inc
    raw = np.expm1(log_val) if LOG_INCIDENCE else log_val
    return np.clip(raw, 0.0, None)

def _incidence_to_cases(incidence_raw: np.ndarray, prov: str, year: int) -> Optional[np.ndarray]:
    if not HAS_POPULATION:
        return None
    else:
        pass
    pop = POP_LOOKUP.get((prov, year))
    if pop is None:
        return None
    else:
        pass
    return incidence_raw * float(pop)

def label_term(name: str) -> str:
    if name in ('sin52', 'cos52'):
        return 'Annual seasonal cycle (52-week)'
    else:
        pass
    if name in ('sin26', 'cos26'):
        return 'Semi-annual seasonal cycle (26-week)'
    else:
        pass
    if name == 'logistic':
        return 'Susceptible depletion (logistic saturation on current incidence)'
    else:
        pass
    if name == 'I_cumul':
        return 'Cumulative incidence memory (rolling sum over lag window)'
    else:
        pass
    if name == 'base_y1_z1_2':
        return 'Incidence (1wk lag) x climate-latent factor z1 (2wk lag)'
    else:
        pass
    if name.startswith('y1_zm'):
        idx = name.replace('y1_zm', '')
        return f'Incidence (1wk lag) x climate-latent factor z{int(idx) + 1} (concurrent)'
    else:
        pass
    if name.startswith('fs_'):
        return f'FunSearch-discovered nonlinear term: {name[3:]}'
    else:
        pass
    if name.startswith('llm_'):
        return f'LLM-proposed biological term: {name[4:]}'
    else:
        pass
    if name.startswith('s') and '_lag' in name:
        try:
            dim_str, lag_str = name[1:].split('_lag')
            dim = int(dim_str)
            if dim == INC_COL:
                return f'Own incidence, {lag_str}-week lag'
            else:
                pass
            return f'Climate-latent factor z{dim + 1}, {lag_str}-week lag'
        except Exception:
            return name
        else:
            pass
        finally:
            pass
    else:
        pass
    return name

def attribute_drivers(model: Dict, s_recent: np.ndarray, weeks_recent: np.ndarray, beta_p: np.ndarray, top_k: int=TOP_DRIVER_TERMS) -> pd.DataFrame:
    n_lags, fs_terms, llm_formulas = (model['n_lags'], model['fs_terms'], model['llm_formulas'])
    x_aug = np.concatenate([s_recent[-1 - lag] for lag in range(n_lags + 1)]).reshape(1, -1).astype(np.float32)
    Phi_t, names = build_library(x_aug, np.array([weeks_recent[-1]]), n_lags, fs_terms, llm_formulas)
    expected_M = beta_p.shape[0]
    if Phi_t.shape[1] < expected_M:
        Phi_t = np.pad(Phi_t, ((0, 0), (0, expected_M - Phi_t.shape[1])), mode='constant')
    elif Phi_t.shape[1] > expected_M:
        Phi_t = Phi_t[:, :expected_M]
        names = names[:expected_M]
    else:
        pass
    contrib = (Phi_t[0] * beta_p[:, INC_COL]).astype(np.float64)
    df = pd.DataFrame({'term': names, 'label': [label_term(n) for n in names], 'contribution_scaled': contrib})
    df['abs_contribution'] = df['contribution_scaled'].abs()
    return df.sort_values('abs_contribution', ascending=False).head(top_k).reset_index(drop=True)

def simulate_forecast_paths(model: Dict, s_obs: np.ndarray, weeks_obs: np.ndarray, beta_p: np.ndarray, q_diag_p: np.ndarray, horizon: int=FORECAST_HORIZON, n_sims: int=N_SIM_PATHS, rng: Optional[np.random.Generator]=None) -> Tuple[np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng(FS_RANDOM_SEED)
    else:
        pass
    n_lags, fs_terms, llm_formulas = (model['n_lags'], model['fs_terms'], model['llm_formulas'])
    last_week = int(weeks_obs[-1])
    fc_weeks = np.array([(last_week - 1 + h) % 52 + 1 for h in range(1, horizon + 1)], dtype=np.float32)
    x0_hist = [s_obs[-1 - lag].copy() for lag in range(n_lags + 1)]
    std_vec = np.sqrt(np.clip(q_diag_p, 1e-12, None))
    paths_scaled = np.zeros((n_sims, horizon), dtype=np.float64)
    for sim in range(n_sims):
        x_hist = [x.copy() for x in x0_hist]
        for h in range(horizon):
            x_aug_t = np.concatenate([x_hist[lag] for lag in range(n_lags + 1)]).reshape(1, -1).astype(np.float32)
            Phi_t, _ = build_library(x_aug_t, np.array([fc_weeks[h]]), n_lags, fs_terms, llm_formulas)
            expected_M = beta_p.shape[0]
            if Phi_t.shape[1] < expected_M:
                Phi_t = np.pad(Phi_t, ((0, 0), (0, expected_M - Phi_t.shape[1])), mode='constant')
            elif Phi_t.shape[1] > expected_M:
                Phi_t = Phi_t[:, :expected_M]
            else:
                pass
            u_t = (Phi_t @ beta_p).flatten()
            noise = rng.normal(0.0, std_vec)
            x_next = x_hist[0] + u_t + noise
            paths_scaled[sim, h] = x_next[INC_COL]
            x_hist = [x_next] + x_hist[:-1]
        else:
            pass
    else:
        pass
    paths_raw = _inc_scaled_to_raw(paths_scaled)
    return (paths_raw, fc_weeks)

def summarize_forecast(paths_raw: np.ndarray, fc_weeks: np.ndarray, threshold_raw: float) -> pd.DataFrame:
    point = np.median(paths_raw, axis=0)
    lower = np.percentile(paths_raw, CI_LOWER_PCT, axis=0)
    upper = np.percentile(paths_raw, CI_UPPER_PCT, axis=0)
    p_week = np.mean(paths_raw > threshold_raw, axis=0)
    crossed_by = np.zeros(paths_raw.shape, dtype=bool)
    crossed_by[:, 0] = paths_raw[:, 0] > threshold_raw
    for h in range(1, paths_raw.shape[1]):
        crossed_by[:, h] = crossed_by[:, h - 1] | (paths_raw[:, h] > threshold_raw)
    else:
        pass
    p_ever_by = crossed_by.mean(axis=0)
    return pd.DataFrame({'week_ahead': np.arange(1, len(fc_weeks) + 1), 'epi_week': fc_weeks.astype(int), 'point_estimate_incidence': point, 'ci_lower': lower, 'ci_upper': upper, 'p_exceed_threshold_this_week': p_week, 'p_exceed_threshold_by_this_week': p_ever_by})

def compute_risk_ranking(train_pd: Dict, test_pd: Dict, recent_weeks: int=4) -> pd.DataFrame:
    rows = []
    cluster_baseline: Dict[int, List[float]] = {}
    prov_baseline = {}
    for prov, pd_t in train_pd.items():
        raw_inc = _inc_scaled_to_raw(pd_t['s_obs'][:, INC_COL])
        prov_baseline[prov] = (float(np.mean(raw_inc)), float(np.std(raw_inc) + 1e-09), pd_t['cluster'])
        cluster_baseline.setdefault(pd_t['cluster'], []).append(float(np.mean(raw_inc)))
    else:
        pass
    cluster_mean = {cid: float(np.mean(v)) for cid, v in cluster_baseline.items()}
    for prov, pd_t in test_pd.items():
        if prov not in prov_baseline:
            continue
        else:
            pass
        raw_inc = _inc_scaled_to_raw(pd_t['s_obs'][:, INC_COL])
        if len(raw_inc) < recent_weeks:
            continue
        else:
            pass
        recent_mean = float(np.mean(raw_inc[-recent_weeks:]))
        base_mean, base_std, cid = prov_baseline[prov]
        z = (recent_mean - base_mean) / base_std
        rows.append({'province': prov, 'cluster_id': cid, 'recent_mean_incidence': recent_mean, 'own_historical_mean_incidence': base_mean, 'z_score_vs_own_baseline': z, 'cluster_mean_incidence': cluster_mean.get(cid, float('nan'))})
    else:
        pass
    df = pd.DataFrame(rows).sort_values('z_score_vs_own_baseline', ascending=False).reset_index(drop=True)
    return df

def compute_reliability_flags(quality_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    merged = quality_df.merge(test_df[['province', 'r2']], on='province', how='left')

    def _flag(row):
        q, r2 = (row['quality_score'], row.get('r2', float('nan')))
        if pd.isna(r2):
            return 'UNKNOWN'
        else:
            pass
        if q >= RELIABILITY_HIGH_Q and r2 >= RELIABILITY_HIGH_R2:
            return 'HIGH'
        else:
            pass
        if q >= RELIABILITY_MED_Q and r2 >= RELIABILITY_MED_R2:
            return 'MEDIUM'
        else:
            pass
        return 'LOW'
    merged['reliability'] = merged.apply(_flag, axis=1)
    return merged[['province', 'quality_score', 'r2', 'reliability']].sort_values('reliability', key=lambda s: s.map({'HIGH': 0, 'MEDIUM': 1, 'LOW': 2, 'UNKNOWN': 3}))
ALERT_THRESHOLD_SWEEP = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

def backtest_lead_time(model: Dict, test_pd: Dict, trainval_pd: Dict, stride: int=BACKTEST_STRIDE, horizon: int=FORECAST_HORIZON, alert_thresholds: List[float]=ALERT_THRESHOLD_SWEEP, outbreak_pct: float=OUTBREAK_PERCENTILE, n_sims: int=150) -> pd.DataFrame:
    n_lags = model['n_lags']
    threshold_by_prov = {}
    for prov, pd_t in trainval_pd.items():
        raw_inc = _inc_scaled_to_raw(pd_t['s_obs'][:, INC_COL])
        threshold_by_prov[prov] = float(np.percentile(raw_inc, outbreak_pct))
    else:
        pass
    rows = []
    for prov, pd_t in test_pd.items():
        s_obs, weeks, cid = (pd_t['s_obs'], pd_t['weeks'], pd_t['cluster'])
        T = len(s_obs)
        if T <= n_lags + horizon + 2 or prov not in threshold_by_prov:
            continue
        else:
            pass
        raw_inc_full = _inc_scaled_to_raw(s_obs[:, INC_COL])
        threshold_raw = threshold_by_prov[prov]
        beta_p = _beta_for(model, prov, cid)
        q_diag_p = model['q_diag'].copy()
        if prov in model.get('q_inc_by_prov', {}):
            q_diag_p[INC_COL] = model['q_inc_by_prov'][prov]
        else:
            pass
        rng = np.random.default_rng(abs(hash(prov)) % 2 ** 31)
        for t0 in range(n_lags, T - horizon - 1, stride):
            s_hist, w_hist = (s_obs[:t0 + 1], weeks[:t0 + 1])
            paths_raw, _ = simulate_forecast_paths(model, s_hist, w_hist, beta_p, q_diag_p, horizon=horizon, n_sims=n_sims, rng=rng)
            p_week = np.mean(paths_raw > threshold_raw, axis=0)
            actual_future = raw_inc_full[t0 + 1:t0 + 1 + horizon]
            actual_h = next((h for h in range(len(actual_future)) if actual_future[h] > threshold_raw), None)
            for alert_prob_threshold in alert_thresholds:
                alert_h = next((h for h in range(horizon) if p_week[h] >= alert_prob_threshold), None)
                if alert_h is not None and actual_h is not None:
                    outcome, lead_time = ('hit', actual_h - alert_h)
                elif alert_h is not None and actual_h is None:
                    outcome, lead_time = ('false_alarm', None)
                elif alert_h is None and actual_h is not None:
                    outcome, lead_time = ('miss', None)
                else:
                    outcome, lead_time = ('true_negative', None)
                rows.append({'province': prov, 'anchor_idx': t0, 'alert_threshold': alert_prob_threshold, 'outcome': outcome, 'lead_time_weeks': lead_time})
            else:
                pass
        else:
            pass
    else:
        pass
    return pd.DataFrame(rows)

def summarize_backtest(backtest_df: pd.DataFrame) -> pd.DataFrame:
    if len(backtest_df) == 0:
        return pd.DataFrame()
    else:
        pass

    def _agg(g):
        n = len(g)
        hits = (g['outcome'] == 'hit').sum()
        false_alarms = (g['outcome'] == 'false_alarm').sum()
        misses = (g['outcome'] == 'miss').sum()
        true_neg = (g['outcome'] == 'true_negative').sum()
        hit_leads = g.loc[g['outcome'] == 'hit', 'lead_time_weeks'].dropna()
        recall = hits / (hits + misses) if hits + misses > 0 else float('nan')
        precision = hits / (hits + false_alarms) if hits + false_alarms > 0 else float('nan')
        return pd.Series({'n_anchors': n, 'hits': hits, 'false_alarms': false_alarms, 'misses': misses, 'true_negatives': true_neg, 'recall': recall, 'precision': precision, 'mean_lead_time_weeks': float(hit_leads.mean()) if len(hit_leads) else float('nan'), 'outbreak_events_seen': hits + misses})
    by_threshold = backtest_df.groupby('alert_threshold').apply(_agg).reset_index()
    by_threshold.insert(0, 'level', 'pooled_by_threshold')
    default_slice = backtest_df[backtest_df['alert_threshold'] == ALERT_PROB_THRESHOLD]
    by_province = default_slice.groupby('province').apply(_agg).reset_index()
    by_province.insert(0, 'level', f'per_province_at_threshold_{ALERT_PROB_THRESHOLD}')
    by_province.insert(1, 'alert_threshold', ALERT_PROB_THRESHOLD)
    return pd.concat([by_threshold, by_province], ignore_index=True, sort=False)
if __name__ == '__main__':
    train_pd = _build_province_dict(latent, TRAIN_YEARS)
    val_pd = _build_province_dict(latent, VAL_YEARS)
    trainval_pd = _build_province_dict(latent, TRAINVAL_YEARS)
    test_pd = _build_province_dict(latent, TEST_YEARS)
    quality_df = compute_province_quality_scores(train_pd)
    included_provinces = set(quality_df.loc[quality_df['quality_score'] >= QUALITY_THRESHOLD, 'province'])
    excluded_provinces = set(quality_df['province']) - included_provinces
    stability_rows = []
    province_r2_rows = []
    model_final = test_r2 = test_df_cell = test_r2_filt = None
    llm_ablation_df = hierarchy_ablation_df = None
    for run_idx in range(1, N_STABILITY_RUNS + 1):
        model_final = fit_winning_model(train_pd, val_pd, trainval_pd)
        test_r2, test_df_cell, test_preds_cell = evaluate_on_split(test_pd, model_final)
        test_df_incl = test_df_cell[test_df_cell['province'].isin(included_provinces)]
        test_r2_filt = float(test_df_incl['r2'].mean()) if len(test_df_incl) else float('nan')
        active_inc, pruned_inc = get_active_and_pruned(model_final)
        eq_rows = [{'run': run_idx, 'term_name': nm, 'coef_incidence': coef, 'active': True} for nm, coef in active_inc]
        eq_rows += [{'run': run_idx, 'term_name': nm, 'coef_incidence': 0.0, 'active': False} for nm in pruned_inc]
        pd.DataFrame(eq_rows).sort_values('coef_incidence', key=lambda s: s.abs(), ascending=False).to_csv(os.path.join(OUT_DIR, f'global_equation_run{run_idx}.csv'), index=False)
        np.savez(os.path.join(OUT_DIR, f'global_g_matrix_run{run_idx}.npz'), g=model_final['g'], term_names=np.array(model_final['term_names'], dtype=object))
        for _, prow in test_df_cell.iterrows():
            province_r2_rows.append({'run': run_idx, 'province': prow['province'], 'cluster_id': prow['cluster_id'], 'r2': prow['r2'], 'mae': prow['mae'], 'mse': prow['mse'], 'included_in_headline': prow['province'] in included_provinces})
        else:
            pass
        llm_ablation_df = run_llm_ablation(model_final, train_pd, val_pd, trainval_pd, test_pd, included_provinces)
        hierarchy_ablation_df = run_hierarchy_ablation(model_final, train_pd, val_pd, trainval_pd, test_pd, included_provinces)
        no_llm_row = llm_ablation_df.loc[llm_ablation_df['variant'] == 'ablation_no_llm'].iloc[0]
        rand_row = llm_ablation_df.loc[llm_ablation_df['variant'] == 'ablation_random_terms'].iloc[0]
        manual_row = llm_ablation_df.loc[llm_ablation_df['variant'] == 'ablation_manual_terms'].iloc[0]
        with_row = llm_ablation_df.loc[llm_ablation_df['variant'] == 'with_llm'].iloc[0]
        global_row = hierarchy_ablation_df.loc[hierarchy_ablation_df['variant'] == 'global_only'].iloc[0]
        full_row = hierarchy_ablation_df.loc[hierarchy_ablation_df['variant'] == 'global_plus_cluster_plus_province (full)'].iloc[0]
        stability_rows.append({'run': run_idx, 'n_llm_terms_surviving': len(model_final['llm_formulas_used']), 'test_r2': test_r2, 'test_r2_filt': test_r2_filt, 'llm_gap_vs_no_llm': with_row['test_r2_filt'] - no_llm_row['test_r2_filt'], 'llm_gap_vs_random': with_row['test_r2_filt'] - rand_row['test_r2_filt'], 'llm_gap_vs_manual': with_row['test_r2_filt'] - manual_row['test_r2_filt'], 'hierarchy_global_only': global_row['test_r2_filt'], 'hierarchy_full': full_row['test_r2_filt'], 'hierarchy_gain': full_row['test_r2_filt'] - global_row['test_r2_filt']})
    else:
        pass
    stability_df = pd.DataFrame(stability_rows)
    stability_df.to_csv(os.path.join(OUT_DIR, 'stability_runs_raw.csv'), index=False)
    _metric_cols = [c for c in stability_df.columns if c != 'run']
    stability_summary = stability_df[_metric_cols].agg(['mean', 'std', 'min', 'max']).T
    stability_summary.index.name = 'metric'
    stability_summary = stability_summary.reset_index()
    stability_summary.to_csv(os.path.join(OUT_DIR, 'stability_summary.csv'), index=False)
    for _, r in stability_summary.iterrows():
        pass
    else:
        pass
    province_r2_df = pd.DataFrame(province_r2_rows)
    province_r2_df.to_csv(os.path.join(OUT_DIR, 'province_r2_stability_raw.csv'), index=False)
    province_r2_summary = province_r2_df.groupby('province')['r2'].agg(['mean', 'std', 'min', 'max', 'count']).reset_index().rename(columns={'count': 'n_runs'})
    incl_lookup = province_r2_df.drop_duplicates('province').set_index('province')['included_in_headline']
    province_r2_summary['included_in_headline'] = province_r2_summary['province'].map(incl_lookup)
    province_r2_summary = province_r2_summary.sort_values('mean', ascending=False).reset_index(drop=True)
    province_r2_summary.to_csv(os.path.join(OUT_DIR, 'province_r2_stability_summary.csv'), index=False)
    for _, r in province_r2_summary.iterrows():
        flag = 'yes' if r['included_in_headline'] else 'no, low quality'
        std_str = f"{r['std']:.4f}" if pd.notna(r['std']) else 'n/a'
    else:
        pass
    test_df_cell.to_csv(os.path.join(OUT_DIR, 'model_test_evaluation.csv'), index=False)
    llm_ablation_df.to_csv(os.path.join(OUT_DIR, 'llm_ablation_last_run.csv'), index=False)
    hierarchy_ablation_df.to_csv(os.path.join(OUT_DIR, 'hierarchy_ablation_last_run.csv'), index=False)
    full_history = build_full_history(latent)
    naive_per_prov_df, naive_summary_df = evaluate_naive_baselines(test_pd, full_history, model_final['n_lags'], included_provinces)
    naive_per_prov_df.to_csv(os.path.join(OUT_DIR, 'naive_baselines_per_province.csv'), index=False)
    stability_test_r2_filt = stability_summary.loc[stability_summary['metric'] == 'test_r2_filt'].iloc[0]
    main_benchmark_df = pd.concat([naive_summary_df, pd.DataFrame([{'variant': f'{WINNING_LABEL} (last run)', 'test_r2_pooled': float('nan'), 'test_r2_mean_per_province': test_r2_filt}, {'variant': f'{WINNING_LABEL} (mean of {N_STABILITY_RUNS} runs, +/- std)', 'test_r2_pooled': float('nan'), 'test_r2_mean_per_province': stability_test_r2_filt['mean']}])], ignore_index=True)
    main_benchmark_df.to_csv(os.path.join(OUT_DIR, 'main_benchmark_table.csv'), index=False)
    for _, r in main_benchmark_df.iterrows():
        pass
    else:
        pass
    all_forecast_rows = []
    driver_rows = []
    rng_master = np.random.default_rng(FS_RANDOM_SEED)
    for prov, pd_t in test_pd.items():
        s_obs, weeks, cid = (pd_t['s_obs'], pd_t['weeks'], pd_t['cluster'])
        if len(s_obs) <= model_final['n_lags'] + 1:
            continue
        else:
            pass
        meets_quality_gate = prov in included_provinces
        beta_p = _beta_for(model_final, prov, cid)
        q_diag_p = model_final['q_diag'].copy()
        if prov in model_final.get('q_inc_by_prov', {}):
            q_diag_p[INC_COL] = model_final['q_inc_by_prov'][prov]
        else:
            pass
        threshold_raw = float(np.percentile(_inc_scaled_to_raw(trainval_pd[prov]['s_obs'][:, INC_COL]), OUTBREAK_PERCENTILE)) if prov in trainval_pd else float('nan')
        paths_raw, fc_weeks = simulate_forecast_paths(model_final, s_obs, weeks, beta_p, q_diag_p, horizon=FORECAST_HORIZON, n_sims=N_SIM_PATHS, rng=rng_master)
        fc_df = summarize_forecast(paths_raw, fc_weeks, threshold_raw)
        fc_df.insert(0, 'province', prov)
        fc_df.insert(1, 'cluster_id', cid)
        fc_df['outbreak_threshold_incidence'] = threshold_raw
        fc_df['meets_quality_gate'] = meets_quality_gate
        last_year = TEST_YEARS[-1]
        if HAS_POPULATION:
            cases = _incidence_to_cases(fc_df['point_estimate_incidence'].values, prov, last_year)
            fc_df['point_estimate_cases_approx'] = cases
        else:
            pass
        all_forecast_rows.append(fc_df)
        drv = attribute_drivers(model_final, s_obs, weeks, beta_p, top_k=TOP_DRIVER_TERMS)
        drv.insert(0, 'province', prov)
        drv['meets_quality_gate'] = meets_quality_gate
        driver_rows.append(drv)
    else:
        pass
    forecast_all = pd.concat(all_forecast_rows, ignore_index=True) if all_forecast_rows else pd.DataFrame()
    driver_all = pd.concat(driver_rows, ignore_index=True) if driver_rows else pd.DataFrame()
    forecast_all.to_csv(os.path.join(OUT_DIR, 'forecast_multistep.csv'), index=False)
    driver_all.to_csv(os.path.join(OUT_DIR, 'driver_attribution.csv'), index=False)
    risk_df = compute_risk_ranking(trainval_pd, test_pd)
    risk_df.to_csv(os.path.join(OUT_DIR, 'risk_ranking.csv'), index=False)
    reliability_df = compute_reliability_flags(quality_df, test_df_cell)
    reliability_df.to_csv(os.path.join(OUT_DIR, 'reliability_flags.csv'), index=False)
    backtest_df = backtest_lead_time(model_final, test_pd, trainval_pd, stride=BACKTEST_STRIDE, horizon=FORECAST_HORIZON, alert_thresholds=ALERT_THRESHOLD_SWEEP)
    backtest_summary = summarize_backtest(backtest_df)
    backtest_df.to_csv(os.path.join(OUT_DIR, 'backtest_lead_time_raw.csv'), index=False)
    backtest_summary.to_csv(os.path.join(OUT_DIR, 'backtest_lead_time_summary.csv'), index=False)
    pooled_by_thr = backtest_summary[backtest_summary['level'] == 'pooled_by_threshold']
    for _, r in pooled_by_thr.sort_values('alert_threshold').iterrows():
        pass
    else:
        pass
    if DATA_CUTOFF is not None:
        pass
    else:
        pass
    default_row = pooled_by_thr[pooled_by_thr['alert_threshold'] == ALERT_PROB_THRESHOLD]
    if len(default_row):
        pooled = default_row.iloc[0]
    else:
        pass
    top_risk = risk_df.head(TOP_RISK_PROVINCES)
    for _, row in top_risk.iterrows():
        prov = row['province']
        rel_row = reliability_df[reliability_df['province'] == prov]
        rel = rel_row['reliability'].values[0] if len(rel_row) else 'UNKNOWN'
        fc_prov = forecast_all[forecast_all['province'] == prov]
        next_wk = fc_prov[fc_prov['week_ahead'] == 1]
        p_out = float(fc_prov['p_exceed_threshold_by_this_week'].max()) if len(fc_prov) else float('nan')
        drv_prov = driver_all[driver_all['province'] == prov].head(2)
        top_drivers = ', '.join(drv_prov['label'].tolist()) if len(drv_prov) else 'n/a'
        if len(next_wk):
            pass
        else:
            pass
    else:
        pass
    llm_gain = llm_ablation_df.loc[llm_ablation_df['variant'] == 'with_llm', 'test_r2_filt'].values[0] - llm_ablation_df.loc[llm_ablation_df['variant'] == 'ablation_no_llm', 'test_r2_filt'].values[0]
    llm_vs_random = llm_ablation_df.loc[llm_ablation_df['variant'] == 'with_llm', 'test_r2_filt'].values[0] - llm_ablation_df.loc[llm_ablation_df['variant'] == 'ablation_random_terms', 'test_r2_filt'].values[0]
    llm_vs_manual = llm_ablation_df.loc[llm_ablation_df['variant'] == 'with_llm', 'test_r2_filt'].values[0] - llm_ablation_df.loc[llm_ablation_df['variant'] == 'ablation_manual_terms', 'test_r2_filt'].values[0]
    hier_full = hierarchy_ablation_df.loc[hierarchy_ablation_df['variant'] == 'global_plus_cluster_plus_province (full)', 'test_r2_filt'].values[0]
    hier_global = hierarchy_ablation_df.loc[hierarchy_ablation_df['variant'] == 'global_only', 'test_r2_filt'].values[0]
    _gap_row = stability_summary.loc[stability_summary['metric'] == 'llm_gap_vs_no_llm'].iloc[0]
    _hier_row = stability_summary.loc[stability_summary['metric'] == 'hierarchy_gain'].iloc[0]
else:
    pass