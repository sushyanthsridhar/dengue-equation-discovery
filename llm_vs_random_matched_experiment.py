"""
REFEREE REPORT: Section 2.3 / 3.1 / 4.4 (Table 9) -- the decisive, matched
LLM-vs-random-proposer experiment. This is the third of the referee's three
"true submission blockers," and the only one not yet addressed
computationally (the validation-split fix is done, the state-space write-up
is a documentation task, not a computation).

WHAT THIS DOES
--------------
For 5 seeds, runs the same round-based term-proposal loop forecast.py
already uses (fit on TRAIN, propose candidate terms, screen them against
the pooled training residual, evaluate on VAL, repeat for MAX_LLM_ROUNDS
rounds, then refit on TRAIN+VAL and evaluate once on TEST) under two
conditions that differ ONLY in where the candidate terms come from:

  LLM proposer    -- exactly forecast.py's existing path: build_llm_prompt
                      + query_llm (Ollama) + parse_llm_suggestions, once
                      per round.
  Random proposer -- the SAME downstream machinery (suggestions_to_formulas
                      for validity checking, score_llm_candidates for the
                      residual-correlation screen) fed a randomly
                      generated batch of candidate formulas each round,
                      built from the same primitives (lagged incidence,
                      lagged latents, seasonal sin/cos terms, products,
                      powers) the LLM is itself prompted to use.

The two arms are advanced in lockstep, round by round, within each seed:
the LLM arm proposes first, and the random arm is then given exactly as
many candidates that same round as the LLM arm produced, so the "total
proposal/evaluation budget" the referee asks to match is equal round by
round, not just in aggregate over the whole run.

Both arms start from the same winning FunSearch structure (fs_terms,
n_lags, loaded from fs_programs/<label>_best.json, exactly as forecast.py
loads it), see the same TRAIN / VAL / TRAIN+VAL / TEST data, and are
screened by the same score_llm_candidates() acceptance rule. The
comparison is driven by validation R2 only; test R2 is computed once,
after that protocol is fixed, exactly as the referee's report asks.

GRAMMAR NOTE (the referee explicitly asks this be stated if it differs):
the random proposer is NOT restricted to FunSearch's own fixed ExtraTerm
grammar (the 8 hard-coded term types in TERM_TYPES). It generates free-form
eval()-able formula strings from the same primitives the LLM is prompted
with, validated by the same suggestions_to_formulas() function the LLM's
own output goes through. So both arms draw from the same expressive
grammar; the comparison is about which proposals get chosen and survive
screening, not about the LLM having access to a larger hypothesis space.

METRICS TRACKED (matching the referee's requested table):
  - Validation R2 (after the final round)
  - Final test R2 (once, after refit on TRAIN+VAL)
  - Valid proposals (candidates that parsed and produced finite,
    correctly shaped values)
  - Parse/validation failures (candidates proposed but rejected before
    ever reaching the residual-correlation screen)
  - Retained proposals (candidates that passed the residual-correlation
    screen and were folded into the model)
  - Evaluations to best validation score (cumulative candidate count,
    across rounds, at the round where validation R2 peaked)

SIMPLIFICATION relative to forecast.py's fit_winning_model(): the final
TRAIN+VAL refit here uses every term accumulated across all rounds,
without forecast.py's extra post-hoc "keep only terms that individually
survived thresholding in the last round" filter. This is applied
identically to both arms, so it does not favor either one; it just keeps
this script's logic easier to audit.

CAVEATS TO CARRY INTO THE PAPER:
  - The LLM's own sampling is not seedable through Ollama's chat API the
    way the random arm's RNG is. "Same five seeds" here means the same
    fixed seed governs the random proposer's RNG and any Python-level
    randomness in both arms, not literal determinism of the LLM's text
    output. Say this plainly if asked.
  - This uses whichever model OLLAMA_MODEL_PRIMARY / OLLAMA_MODEL_FALLBACK
    resolve to in forecast.py at run time -- check the console output
    (query_llm prints which model actually answered each call).
  - Runtime: 5 seeds x MAX_LLM_ROUNDS rounds x 2 arms, each round does one
    EM fit on TRAIN plus one screening pass, plus (LLM arm only) a network
    call to Ollama, plus one final TRAIN+VAL refit per arm per seed. This
    is roughly double the cost of one forecast.py stability run and will
    take a while on a laptop. N_SEEDS and MAX_ROUNDS are constants right
    below the imports if you need to shorten it for a first pass.

Usage:
    python3 llm_vs_random_matched_experiment.py

Writes to outputs/:
    llm_vs_random_matched_per_seed.csv   -- one row per (seed, arm)
    llm_vs_random_matched_summary.csv    -- mean/std per arm across seeds,
                                             in the referee's requested
                                             table layout
"""
import os
import sys
import json
import random
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

print('[matched-experiment] imported', F.__name__)

OUT_DIR = F.OUT_DIR
os.makedirs(OUT_DIR, exist_ok=True)

N_SEEDS = 5
MAX_ROUNDS = F.MAX_LLM_ROUNDS  # same round budget forecast.py itself uses

train_pd = F._build_province_dict(F.latent, F.TRAIN_YEARS)
val_pd = F._build_province_dict(F.latent, F.VAL_YEARS)
trainval_pd = F._build_province_dict(F.latent, F.TRAINVAL_YEARS)
test_pd = F._build_province_dict(F.latent, F.TEST_YEARS)

with open(F.SAVED_FS_JSON) as fh:
    fs_best = F.Program.from_dict(json.load(fh))
N_LAGS, FS_TERMS = max(fs_best.n_lags, 3), fs_best.extra_terms
print(f'[matched-experiment] winning structure: n_lags={N_LAGS}, {len(FS_TERMS)} fixed FunSearch terms (identical for both arms)')
print(f'[matched-experiment] N_SEEDS={N_SEEDS}  MAX_ROUNDS={MAX_ROUNDS}')

sample_prov = list(train_pd.keys())[0]
SAMPLE_AUG = F._build_aug(train_pd[sample_prov]['s_obs'], N_LAGS)
SAMPLE_WEEKS = train_pd[sample_prov]['weeks'][N_LAGS:]
LATENT_MEANING = F.compute_latent_covariate_meaning()

# Random-proposer templates: same primitives (lagged incidence, lagged
# latents, seasonal terms, products, powers) the LLM prompt itself uses.
RANDOM_TEMPLATES = [
    lambda a, b, c, d, e: (f'rand_pow2_l{a}', f'y_lag{a} ** 2'),
    lambda a, b, c, d, e: (f'rand_pow3_l{a}', f'y_lag{a} ** 3'),
    lambda a, b, c, d, e: (f'rand_yz_l{a}_z{b}l{c}', f'y_lag{a} * z{b}_lag{c}'),
    lambda a, b, c, d, e: (f'rand_zz_z{b}l{c}_z{d}l{e}', f'z{b}_lag{c} * z{d}_lag{e}'),
    lambda a, b, c, d, e: (f'rand_zsq_z{b}l{c}', f'z{b}_lag{c} ** 2'),
    lambda a, b, c, d, e: (f'rand_sin52_y_l{a}', f'np.sin(2*np.pi*week_arr/52) * y_lag{a}'),
    lambda a, b, c, d, e: (f'rand_sin26_y_l{a}', f'np.sin(2*np.pi*week_arr/26) * y_lag{a}'),
    lambda a, b, c, d, e: (f'rand_sin52_z{b}l{c}', f'np.sin(2*np.pi*week_arr/52) * z{b}_lag{c}'),
    lambda a, b, c, d, e: (f'rand_depletion_l{a}_l{b}', f'y_lag{a} * (1.0 - y_lag{b})'),
    lambda a, b, c, d, e: (f'rand_diff_l{a}_l{b}', f'y_lag{a} - y_lag{b}'),
]


def random_suggestions(n, rng, round_idx, n_lags):
    """n dicts shaped exactly like parse_llm_suggestions() output, built
    from the same primitives the LLM is prompted with (see the module
    docstring's GRAMMAR NOTE)."""
    out = []
    for i in range(n):
        template = rng.choice(RANDOM_TEMPLATES)
        a = rng.randint(0, n_lags)
        b = rng.randint(1, F.LATENT_DIM)
        c = rng.randint(0, n_lags)
        d = rng.randint(1, F.LATENT_DIM)
        e = rng.randint(0, n_lags)
        name, formula = template(a, b, c, d, e)
        out.append({'name': f'{name}_r{round_idx}_{i}', 'formula': formula, 'reason': 'random proposer'})
    return out


def new_arm_state():
    return dict(formulas=[], val_r2_hist=[], spectral_hist=[], prev_suggestions=[],
                total_valid=0, total_parse_fail=0, total_retained=0,
                evals_cumulative=0, evals_at_round=[])


def run_seed(seed):
    rng = random.Random(seed)
    state = {'llm_proposer': new_arm_state(), 'random_proposer': new_arm_state()}

    for r in range(MAX_ROUNDS):
        # --- LLM arm: fit, evaluate, propose via Ollama, screen ---
        st = state['llm_proposer']
        model_train = F.run_hierarchical_em(train_pd, N_LAGS, FS_TERMS, st['formulas'], label=f'llm_seed{seed}_round{r + 1}')
        val_r2, val_df, val_preds = F.evaluate_on_split(val_pd, model_train)
        spectral_sc = F.compute_spectral_score(val_preds)
        spectral_delta = spectral_sc - st['spectral_hist'][-1] if st['spectral_hist'] else 0.0
        st['val_r2_hist'].append(val_r2)
        st['spectral_hist'].append(spectral_sc)
        active_terms, pruned_terms = F.get_active_and_pruned(model_train)
        cluster_r2 = {int(cid): float(val_df[val_df['cluster_id'] == cid]['r2'].mean()) for cid in val_df['cluster_id'].unique()}
        prompt = F.build_llm_prompt(fs_best, val_r2, spectral_sc, spectral_delta, active_terms, pruned_terms,
                                     cluster_r2, r + 1, st['prev_suggestions'], st['val_r2_hist'], LATENT_MEANING)
        raw_resp = F.query_llm(prompt)
        suggestions = F.parse_llm_suggestions(raw_resp)
        st['prev_suggestions'] = suggestions
        n_target = len(suggestions)
        new_formulas = F.suggestions_to_formulas(suggestions, SAMPLE_AUG, N_LAGS, SAMPLE_WEEKS)
        st['total_valid'] += len(new_formulas)
        st['total_parse_fail'] += n_target - len(new_formulas)
        st['evals_cumulative'] += n_target
        kept = F.score_llm_candidates(new_formulas, train_pd, N_LAGS, FS_TERMS, st['formulas'], model_train) if new_formulas else []
        st['total_retained'] += len(kept)
        existing = {nm for nm, _ in st['formulas']}
        for nm, f_ in kept:
            if nm not in existing:
                st['formulas'].append((nm, f_))
        st['evals_at_round'].append((r, val_r2, st['evals_cumulative']))
        print(f'  [llm_proposer    seed={seed} round {r + 1}/{MAX_ROUNDS}] val_R2={val_r2:+.4f}  proposed={n_target}  valid={len(new_formulas)}  retained={len(kept)}')

        # --- Random arm: SAME round, matched to the LLM arm's proposal count this round ---
        st2 = state['random_proposer']
        model_train2 = F.run_hierarchical_em(train_pd, N_LAGS, FS_TERMS, st2['formulas'], label=f'random_seed{seed}_round{r + 1}')
        val_r2_2, val_df2, val_preds2 = F.evaluate_on_split(val_pd, model_train2)
        st2['val_r2_hist'].append(val_r2_2)
        suggestions2 = random_suggestions(max(n_target, 1), rng, r, N_LAGS)
        st2['prev_suggestions'] = suggestions2
        new_formulas2 = F.suggestions_to_formulas(suggestions2, SAMPLE_AUG, N_LAGS, SAMPLE_WEEKS)
        st2['total_valid'] += len(new_formulas2)
        st2['total_parse_fail'] += len(suggestions2) - len(new_formulas2)
        st2['evals_cumulative'] += len(suggestions2)
        kept2 = F.score_llm_candidates(new_formulas2, train_pd, N_LAGS, FS_TERMS, st2['formulas'], model_train2) if new_formulas2 else []
        st2['total_retained'] += len(kept2)
        existing2 = {nm for nm, _ in st2['formulas']}
        for nm, f_ in kept2:
            if nm not in existing2:
                st2['formulas'].append((nm, f_))
        st2['evals_at_round'].append((r, val_r2_2, st2['evals_cumulative']))
        print(f'  [random_proposer seed={seed} round {r + 1}/{MAX_ROUNDS}] val_R2={val_r2_2:+.4f}  proposed={len(suggestions2)}  valid={len(new_formulas2)}  retained={len(kept2)}')

    results = []
    for arm_name in ('llm_proposer', 'random_proposer'):
        st = state[arm_name]
        best_round = max(st['evals_at_round'], key=lambda t: t[1]) if st['evals_at_round'] else (0, float('nan'), 0)
        model_final = F.run_hierarchical_em(trainval_pd, N_LAGS, FS_TERMS, st['formulas'], label=f'{arm_name}_seed{seed}_FINAL')
        model_final['q_inc_by_prov'] = F.estimate_province_q_inc(model_final, trainval_pd)
        test_r2, _, _ = F.evaluate_on_split(test_pd, model_final)
        results.append({
            'arm': arm_name,
            'seed': seed,
            'val_r2': st['val_r2_hist'][-1] if st['val_r2_hist'] else float('nan'),
            'test_r2': test_r2,
            'valid_proposals': st['total_valid'],
            'parse_failures': st['total_parse_fail'],
            'retained_proposals': st['total_retained'],
            'evaluations_to_best_val': best_round[2],
            'n_terms_final': len(st['formulas']),
        })
    return results


all_rows = []
for seed in range(1, N_SEEDS + 1):
    print(f'\n[SEED {seed}/{N_SEEDS}]')
    all_rows.extend(run_seed(seed))

per_seed_df = pd.DataFrame(all_rows)
per_seed_df.to_csv(os.path.join(OUT_DIR, 'llm_vs_random_matched_per_seed.csv'), index=False)

summary_rows = []
for metric in ('val_r2', 'test_r2', 'valid_proposals', 'parse_failures', 'retained_proposals', 'evaluations_to_best_val'):
    llm_vals = per_seed_df.loc[per_seed_df['arm'] == 'llm_proposer', metric]
    rand_vals = per_seed_df.loc[per_seed_df['arm'] == 'random_proposer', metric]
    summary_rows.append({
        'metric': metric,
        'llm_proposer_mean': llm_vals.mean(),
        'llm_proposer_std': llm_vals.std(),
        'random_proposer_mean': rand_vals.mean(),
        'random_proposer_std': rand_vals.std(),
    })
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(OUT_DIR, 'llm_vs_random_matched_summary.csv'), index=False)

print('\n[DONE] matched LLM-vs-random-proposer experiment, referee report items 2.3 / 3.1 / 4.4 (Table 9)')
print(summary_df.to_string(index=False))
print('\nWrote llm_vs_random_matched_per_seed.csv and llm_vs_random_matched_summary.csv to', OUT_DIR)
