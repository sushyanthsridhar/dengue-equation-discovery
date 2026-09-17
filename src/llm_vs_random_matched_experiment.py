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
(_stdout_before, _stderr_before) = (sys.stdout, sys.stderr)
try:
    import forecast as F
except ModuleNotFoundError:
    import forecast_temporal_fix as F
(sys.stdout, sys.stderr) = (_stdout_before, _stderr_before)
OUT_DIR = F.OUT_DIR
os.makedirs(OUT_DIR, exist_ok=True)
N_SEEDS = 5
MAX_ROUNDS = F.MAX_LLM_ROUNDS
train_pd = F._build_province_dict(F.latent, F.TRAIN_YEARS)
val_pd = F._build_province_dict(F.latent, F.VAL_YEARS)
trainval_pd = F._build_province_dict(F.latent, F.TRAINVAL_YEARS)
test_pd = F._build_province_dict(F.latent, F.TEST_YEARS)
with open(F.SAVED_FS_JSON) as fh:
    fs_best = F.Program.from_dict(json.load(fh))
(N_LAGS, FS_TERMS) = (max(fs_best.n_lags, 3), fs_best.extra_terms)
sample_prov = list(train_pd.keys())[0]
SAMPLE_AUG = F._build_aug(train_pd[sample_prov]['s_obs'], N_LAGS)
SAMPLE_WEEKS = train_pd[sample_prov]['weeks'][N_LAGS:]
LATENT_MEANING = F.compute_latent_covariate_meaning()
RANDOM_TEMPLATES = [lambda a, b, c, d, e: (f'rand_pow2_l{a}', f'y_lag{a} ** 2'), lambda a, b, c, d, e: (f'rand_pow3_l{a}', f'y_lag{a} ** 3'), lambda a, b, c, d, e: (f'rand_yz_l{a}_z{b}l{c}', f'y_lag{a} * z{b}_lag{c}'), lambda a, b, c, d, e: (f'rand_zz_z{b}l{c}_z{d}l{e}', f'z{b}_lag{c} * z{d}_lag{e}'), lambda a, b, c, d, e: (f'rand_zsq_z{b}l{c}', f'z{b}_lag{c} ** 2'), lambda a, b, c, d, e: (f'rand_sin52_y_l{a}', f'np.sin(2*np.pi*week_arr/52) * y_lag{a}'), lambda a, b, c, d, e: (f'rand_sin26_y_l{a}', f'np.sin(2*np.pi*week_arr/26) * y_lag{a}'), lambda a, b, c, d, e: (f'rand_sin52_z{b}l{c}', f'np.sin(2*np.pi*week_arr/52) * z{b}_lag{c}'), lambda a, b, c, d, e: (f'rand_depletion_l{a}_l{b}', f'y_lag{a} * (1.0 - y_lag{b})'), lambda a, b, c, d, e: (f'rand_diff_l{a}_l{b}', f'y_lag{a} - y_lag{b}')]

def random_suggestions(n, rng, round_idx, n_lags):
    out = []
    for i in range(n):
        template = rng.choice(RANDOM_TEMPLATES)
        a = rng.randint(0, n_lags)
        b = rng.randint(1, F.LATENT_DIM)
        c = rng.randint(0, n_lags)
        d = rng.randint(1, F.LATENT_DIM)
        e = rng.randint(0, n_lags)
        (name, formula) = template(a, b, c, d, e)
        out.append({'name': f'{name}_r{round_idx}_{i}', 'formula': formula, 'reason': 'random proposer'})
    return out

def new_arm_state():
    return dict(formulas=[], val_r2_hist=[], spectral_hist=[], prev_suggestions=[], total_valid=0, total_parse_fail=0, total_retained=0, evals_cumulative=0, evals_at_round=[])

def run_seed(seed):
    rng = random.Random(seed)
    state = {'llm_proposer': new_arm_state(), 'random_proposer': new_arm_state()}
    for r in range(MAX_ROUNDS):
        st = state['llm_proposer']
        model_train = F.run_hierarchical_em(train_pd, N_LAGS, FS_TERMS, st['formulas'], label=f'llm_seed{seed}_round{r + 1}')
        (val_r2, val_df, val_preds) = F.evaluate_on_split(val_pd, model_train)
        spectral_sc = F.compute_spectral_score(val_preds)
        spectral_delta = spectral_sc - st['spectral_hist'][-1] if st['spectral_hist'] else 0.0
        st['val_r2_hist'].append(val_r2)
        st['spectral_hist'].append(spectral_sc)
        (active_terms, pruned_terms) = F.get_active_and_pruned(model_train)
        cluster_r2 = {int(cid): float(val_df[val_df['cluster_id'] == cid]['r2'].mean()) for cid in val_df['cluster_id'].unique()}
        prompt = F.build_llm_prompt(fs_best, val_r2, spectral_sc, spectral_delta, active_terms, pruned_terms, cluster_r2, r + 1, st['prev_suggestions'], st['val_r2_hist'], LATENT_MEANING)
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
        existing = {nm for (nm, _) in st['formulas']}
        for (nm, f_) in kept:
            if nm not in existing:
                st['formulas'].append((nm, f_))
        st['evals_at_round'].append((r, val_r2, st['evals_cumulative']))
        st2 = state['random_proposer']
        model_train2 = F.run_hierarchical_em(train_pd, N_LAGS, FS_TERMS, st2['formulas'], label=f'random_seed{seed}_round{r + 1}')
        (val_r2_2, val_df2, val_preds2) = F.evaluate_on_split(val_pd, model_train2)
        st2['val_r2_hist'].append(val_r2_2)
        suggestions2 = random_suggestions(max(n_target, 1), rng, r, N_LAGS)
        st2['prev_suggestions'] = suggestions2
        new_formulas2 = F.suggestions_to_formulas(suggestions2, SAMPLE_AUG, N_LAGS, SAMPLE_WEEKS)
        st2['total_valid'] += len(new_formulas2)
        st2['total_parse_fail'] += len(suggestions2) - len(new_formulas2)
        st2['evals_cumulative'] += len(suggestions2)
        kept2 = F.score_llm_candidates(new_formulas2, train_pd, N_LAGS, FS_TERMS, st2['formulas'], model_train2) if new_formulas2 else []
        st2['total_retained'] += len(kept2)
        existing2 = {nm for (nm, _) in st2['formulas']}
        for (nm, f_) in kept2:
            if nm not in existing2:
                st2['formulas'].append((nm, f_))
        st2['evals_at_round'].append((r, val_r2_2, st2['evals_cumulative']))
    results = []
    for arm_name in ('llm_proposer', 'random_proposer'):
        st = state[arm_name]
        best_round = max(st['evals_at_round'], key=lambda t: t[1]) if st['evals_at_round'] else (0, float('nan'), 0)
        model_final = F.run_hierarchical_em(trainval_pd, N_LAGS, FS_TERMS, st['formulas'], label=f'{arm_name}_seed{seed}_FINAL')
        model_final['q_inc_by_prov'] = F.estimate_province_q_inc(model_final, trainval_pd)
        (test_r2, _, _) = F.evaluate_on_split(test_pd, model_final)
        results.append({'arm': arm_name, 'seed': seed, 'val_r2': st['val_r2_hist'][-1] if st['val_r2_hist'] else float('nan'), 'test_r2': test_r2, 'valid_proposals': st['total_valid'], 'parse_failures': st['total_parse_fail'], 'retained_proposals': st['total_retained'], 'evaluations_to_best_val': best_round[2], 'n_terms_final': len(st['formulas'])})
    return results
all_rows = []
for seed in range(1, N_SEEDS + 1):
    all_rows.extend(run_seed(seed))
per_seed_df = pd.DataFrame(all_rows)
per_seed_df.to_csv(os.path.join(OUT_DIR, 'llm_vs_random_matched_per_seed.csv'), index=False)
summary_rows = []
for metric in ('val_r2', 'test_r2', 'valid_proposals', 'parse_failures', 'retained_proposals', 'evaluations_to_best_val'):
    llm_vals = per_seed_df.loc[per_seed_df['arm'] == 'llm_proposer', metric]
    rand_vals = per_seed_df.loc[per_seed_df['arm'] == 'random_proposer', metric]
    summary_rows.append({'metric': metric, 'llm_proposer_mean': llm_vals.mean(), 'llm_proposer_std': llm_vals.std(), 'random_proposer_mean': rand_vals.mean(), 'random_proposer_std': rand_vals.std()})
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(OUT_DIR, 'llm_vs_random_matched_summary.csv'), index=False)
