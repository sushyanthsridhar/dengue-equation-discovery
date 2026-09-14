"""
REFEREE REPORT addendum: "strongly recommended if inexpensive" item under
Section 5 -- "LLM parse-failure, acceptance, runtime, and memory
diagnostics."

Parse-failure and acceptance (valid proposals, parse failures, retained
proposals, evaluations-to-best-validation) are already reported by
llm_vs_random_matched_experiment.py. This script adds the two pieces still
missing: per-round wall-clock runtime and process memory usage for the
LLM-guided term-proposal loop, using the identical round logic
forecast.py's fit_winning_model() runs (fit on TRAIN, evaluate on VAL,
build the prompt, query the LLM, screen proposals), so the timings
measured here reflect the actual cost of the LLM step as it already runs
in your pipeline -- this is not a separate synthetic benchmark.

WHAT IT MEASURES, PER ROUND, PER SEED:
  - EM fit + evaluation wall time (the run_hierarchical_em +
    evaluate_on_split cost -- what the round would cost even with no LLM
    step at all, useful for seeing what fraction of round time the LLM
    actually adds)
  - Prompt-build wall time (build_llm_prompt alone)
  - Ollama round-trip wall time (query_llm alone -- the network + model
    inference cost)
  - Screening wall time (suggestions_to_formulas + score_llm_candidates)
  - Total round wall time
  - Process RSS memory after each of the four steps above
  - Prompt length and raw response length, in characters
  - Suggestions returned / valid / retained counts, for cross-reference
    with llm_vs_random_matched_experiment.py's numbers

N_SEEDS repeats of the full MAX_LLM_ROUNDS-round loop are run. This is a
timing/memory measurement, not an accuracy comparison, so N_SEEDS defaults
lower than the matched experiment; raise or lower it below as needed.

MEMORY SOURCE: uses psutil for live RSS if it is installed
(pip install psutil). If psutil is not available, falls back to
resource.getrusage's peak RSS, which is a cumulative peak since process
start (not a live per-round snapshot) -- less precise round to round, but
needs no extra dependency. The script prints which source it is using;
installing psutil first is recommended if you want per-round precision.

Which model actually answered each Ollama call (primary vs fallback) is
already printed to the console by query_llm itself -- check the console
output if that matters for a given run; it is not re-parsed into the CSV
here to keep this script simple.

Usage:
    python3 llm_diagnostics.py

Writes to outputs/:
    llm_diagnostics_per_round.csv   -- one row per (seed, round)
    llm_diagnostics_summary.csv     -- mean/std/min/max across all rounds
"""
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

_stdout_before, _stderr_before = sys.stdout, sys.stderr
try:
    import forecast as F
except ModuleNotFoundError:
    import forecast_temporal_fix as F
# forecast.py / forecast_temporal_fix.py redirects stdout/stderr to its own
# log file as a side effect of import; put ours back for this script's own
# console output.
sys.stdout, sys.stderr = _stdout_before, _stderr_before

print('[llm-diagnostics] imported', F.__name__)

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
        # macOS reports ru_maxrss in bytes, Linux reports it in KB
        return peak / (1024 * 1024) if sys.platform == 'darwin' else peak / 1024

    MEM_SOURCE = "resource.getrusage (cumulative peak since process start, not a live snapshot -- install psutil for per-round precision)"

print(f'[llm-diagnostics] memory source: {MEM_SOURCE}')
print(f'[llm-diagnostics] N_SEEDS={N_SEEDS}  MAX_ROUNDS={MAX_ROUNDS}')

train_pd = F._build_province_dict(F.latent, F.TRAIN_YEARS)
val_pd = F._build_province_dict(F.latent, F.VAL_YEARS)

with open(F.SAVED_FS_JSON) as fh:
    fs_best = F.Program.from_dict(json.load(fh))
N_LAGS, FS_TERMS = max(fs_best.n_lags, 3), fs_best.extra_terms
print(f'[llm-diagnostics] winning structure: n_lags={N_LAGS}, {len(FS_TERMS)} fixed FunSearch terms')

sample_prov = list(train_pd.keys())[0]
SAMPLE_AUG = F._build_aug(train_pd[sample_prov]['s_obs'], N_LAGS)
SAMPLE_WEEKS = train_pd[sample_prov]['weeks'][N_LAGS:]
LATENT_MEANING = F.compute_latent_covariate_meaning()

rows = []

for seed in range(1, N_SEEDS + 1):
    print(f'\n[SEED {seed}/{N_SEEDS}]')
    formulas = []
    val_r2_hist = []
    spectral_hist = []
    prev_suggestions = []

    for r in range(MAX_ROUNDS):
        mem_before = get_rss_mb()
        t0 = time.perf_counter()

        model_train = F.run_hierarchical_em(train_pd, N_LAGS, FS_TERMS, formulas, label=f'diag_seed{seed}_round{r + 1}')
        val_r2, val_df, val_preds = F.evaluate_on_split(val_pd, model_train)
        t_fit = time.perf_counter()
        mem_after_fit = get_rss_mb()

        spectral_sc = F.compute_spectral_score(val_preds)
        spectral_delta = spectral_sc - spectral_hist[-1] if spectral_hist else 0.0
        val_r2_hist.append(val_r2)
        spectral_hist.append(spectral_sc)
        active_terms, pruned_terms = F.get_active_and_pruned(model_train)
        cluster_r2 = {int(cid): float(val_df[val_df['cluster_id'] == cid]['r2'].mean()) for cid in val_df['cluster_id'].unique()}

        prompt = F.build_llm_prompt(fs_best, val_r2, spectral_sc, spectral_delta, active_terms, pruned_terms,
                                     cluster_r2, r + 1, prev_suggestions, val_r2_hist, LATENT_MEANING)
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

        existing = {nm for nm, _ in formulas}
        for nm, f_ in kept:
            if nm not in existing:
                formulas.append((nm, f_))

        row = {
            'seed': seed,
            'round': r + 1,
            'em_fit_eval_seconds': t_fit - t0,
            'prompt_build_seconds': t_prompt - t_fit,
            'ollama_call_seconds': t_llm - t_prompt,
            'screening_seconds': t_screen - t_llm,
            'total_round_seconds': t_screen - t0,
            'mem_mb_before_round': mem_before,
            'mem_mb_after_em_fit': mem_after_fit,
            'mem_mb_after_ollama_call': mem_after_llm,
            'mem_mb_after_screening': mem_after_screen,
            'prompt_chars': len(prompt),
            'response_chars': len(raw_resp) if raw_resp else 0,
            'n_suggestions': len(suggestions),
            'n_valid': len(new_formulas),
            'n_retained': len(kept),
        }
        rows.append(row)
        print(f"  round {r + 1}/{MAX_ROUNDS}: EM+eval={row['em_fit_eval_seconds']:.1f}s  ollama={row['ollama_call_seconds']:.1f}s  "
              f"screening={row['screening_seconds']:.1f}s  total={row['total_round_seconds']:.1f}s  "
              f"mem_after_call={mem_after_llm:.0f}MB  suggestions={len(suggestions)}")

per_round_df = pd.DataFrame(rows)
per_round_df.to_csv(os.path.join(OUT_DIR, 'llm_diagnostics_per_round.csv'), index=False)

numeric_cols = [c for c in per_round_df.columns if c not in ('seed', 'round')]
summary_df = per_round_df[numeric_cols].agg(['mean', 'std', 'min', 'max']).T
summary_df.index.name = 'metric'
summary_df = summary_df.reset_index()
summary_df.to_csv(os.path.join(OUT_DIR, 'llm_diagnostics_summary.csv'), index=False)

print('\n[DONE] LLM runtime + memory diagnostics')
print(summary_df.to_string(index=False))
print('\nWrote llm_diagnostics_per_round.csv and llm_diagnostics_summary.csv to', OUT_DIR)
