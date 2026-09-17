import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
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
N_SEEDS = 3
MAX_ROUNDS = F.MAX_LLM_ROUNDS
try:
    import psutil
    _PROC = psutil.Process(os.getpid())

    def get_rss_mb():
        return _PROC.memory_info().rss / (1024 * 1024)
    MEM_SOURCE = 'psutil (live RSS)'
except ImportError:
    import resource

    def get_rss_mb():
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / (1024 * 1024) if sys.platform == 'darwin' else peak / 1024
    MEM_SOURCE = 'resource.getrusage (cumulative peak since process start, not a live snapshot -- install psutil for per-round precision)'
train_pd = F._build_province_dict(F.latent, F.TRAIN_YEARS)
val_pd = F._build_province_dict(F.latent, F.VAL_YEARS)
with open(F.SAVED_FS_JSON) as fh:
    fs_best = F.Program.from_dict(json.load(fh))
(N_LAGS, FS_TERMS) = (max(fs_best.n_lags, 3), fs_best.extra_terms)
sample_prov = list(train_pd.keys())[0]
SAMPLE_AUG = F._build_aug(train_pd[sample_prov]['s_obs'], N_LAGS)
SAMPLE_WEEKS = train_pd[sample_prov]['weeks'][N_LAGS:]
LATENT_MEANING = F.compute_latent_covariate_meaning()
rows = []
for seed in range(1, N_SEEDS + 1):
    formulas = []
    val_r2_hist = []
    spectral_hist = []
    prev_suggestions = []
    for r in range(MAX_ROUNDS):
        mem_before = get_rss_mb()
        t0 = time.perf_counter()
        model_train = F.run_hierarchical_em(train_pd, N_LAGS, FS_TERMS, formulas, label=f'diag_seed{seed}_round{r + 1}')
        (val_r2, val_df, val_preds) = F.evaluate_on_split(val_pd, model_train)
        t_fit = time.perf_counter()
        mem_after_fit = get_rss_mb()
        spectral_sc = F.compute_spectral_score(val_preds)
        spectral_delta = spectral_sc - spectral_hist[-1] if spectral_hist else 0.0
        val_r2_hist.append(val_r2)
        spectral_hist.append(spectral_sc)
        (active_terms, pruned_terms) = F.get_active_and_pruned(model_train)
        cluster_r2 = {int(cid): float(val_df[val_df['cluster_id'] == cid]['r2'].mean()) for cid in val_df['cluster_id'].unique()}
        prompt = F.build_llm_prompt(fs_best, val_r2, spectral_sc, spectral_delta, active_terms, pruned_terms, cluster_r2, r + 1, prev_suggestions, val_r2_hist, LATENT_MEANING)
        t_prompt = time.perf_counter()
        raw_resp = F.query_llm(prompt)
        t_llm = time.perf_counter()
        mem_after_llm = get_rss_mb()
        suggestions = F.parse_llm_suggestions(raw_resp)
        prev_suggestions = suggestions
        new_formulas = F.suggestions_to_formulas(suggestions, SAMPLE_AUG, N_LAGS, SAMPLE_WEEKS)
        kept = F.score_llm_candidates(new_formulas, train_pd, N_LAGS, FS_TERMS, formulas, model_train) if new_formulas else []
        t_screen = time.perf_counter()
        mem_after_screen = get_rss_mb()
        existing = {nm for (nm, _) in formulas}
        for (nm, f_) in kept:
            if nm not in existing:
                formulas.append((nm, f_))
        row = {'seed': seed, 'round': r + 1, 'em_fit_eval_seconds': t_fit - t0, 'prompt_build_seconds': t_prompt - t_fit, 'ollama_call_seconds': t_llm - t_prompt, 'screening_seconds': t_screen - t_llm, 'total_round_seconds': t_screen - t0, 'mem_mb_before_round': mem_before, 'mem_mb_after_em_fit': mem_after_fit, 'mem_mb_after_ollama_call': mem_after_llm, 'mem_mb_after_screening': mem_after_screen, 'prompt_chars': len(prompt), 'response_chars': len(raw_resp) if raw_resp else 0, 'n_suggestions': len(suggestions), 'n_valid': len(new_formulas), 'n_retained': len(kept)}
        rows.append(row)
per_round_df = pd.DataFrame(rows)
per_round_df.to_csv(os.path.join(OUT_DIR, 'llm_diagnostics_per_round.csv'), index=False)
numeric_cols = [c for c in per_round_df.columns if c not in ('seed', 'round')]
summary_df = per_round_df[numeric_cols].agg(['mean', 'std', 'min', 'max']).T
summary_df.index.name = 'metric'
summary_df = summary_df.reset_index()
summary_df.to_csv(os.path.join(OUT_DIR, 'llm_diagnostics_summary.csv'), index=False)
