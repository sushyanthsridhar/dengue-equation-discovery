import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import forecast as F
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, 'results')
os.makedirs(OUT_DIR, exist_ok=True)
N_SIMS = 150

def collect_anchor_scores_and_labels(model, test_pd, trainval_pd, stride=F.BACKTEST_STRIDE, horizon=F.FORECAST_HORIZON, outbreak_pct=F.OUTBREAK_PERCENTILE, n_sims=N_SIMS):
    n_lags = model['n_lags']
    threshold_by_prov = {}
    for (prov, pd_t) in trainval_pd.items():
        raw_inc = F._inc_scaled_to_raw(pd_t['s_obs'][:, F.INC_COL])
        threshold_by_prov[prov] = float(np.percentile(raw_inc, outbreak_pct))
    rows = []
    for (prov, pd_t) in test_pd.items():
        (s_obs, weeks, cid) = (pd_t['s_obs'], pd_t['weeks'], pd_t['cluster'])
        T = len(s_obs)
        if T <= n_lags + horizon + 2 or prov not in threshold_by_prov:
            continue
        raw_inc_full = F._inc_scaled_to_raw(s_obs[:, F.INC_COL])
        threshold_raw = threshold_by_prov[prov]
        beta_p = F._beta_for(model, prov, cid)
        q_diag_p = model['q_diag'].copy()
        if prov in model.get('q_inc_by_prov', {}):
            q_diag_p[F.INC_COL] = model['q_inc_by_prov'][prov]
        rng = np.random.default_rng(abs(hash(prov)) % 2 ** 31)
        for t0 in range(n_lags, T - horizon - 1, stride):
            (s_hist, w_hist) = (s_obs[:t0 + 1], weeks[:t0 + 1])
            (paths_raw, fc_weeks) = F.simulate_forecast_paths(model, s_hist, w_hist, beta_p, q_diag_p, horizon=horizon, n_sims=n_sims, rng=rng)
            fc_df = F.summarize_forecast(paths_raw, fc_weeks, threshold_raw)
            score = float(fc_df['p_exceed_threshold_by_this_week'].iloc[-1])
            actual_future = raw_inc_full[t0 + 1:t0 + 1 + horizon]
            label = int(np.any(actual_future > threshold_raw))
            rows.append({'province': prov, 'anchor_idx': t0, 'score': score, 'label': label})
    return pd.DataFrame(rows)

def _safe_auc(labels, scores):
    labels = np.asarray(labels)
    (n_pos, n_neg) = (int((labels == 1).sum()), int((labels == 0).sum()))
    if n_pos == 0 or n_neg == 0:
        return (float('nan'), n_pos, n_neg)
    return (float(roc_auc_score(labels, scores)), n_pos, n_neg)
if __name__ == '__main__':
    train_pd = F._build_province_dict(F.latent, F.TRAIN_YEARS)
    val_pd = F._build_province_dict(F.latent, F.VAL_YEARS)
    trainval_pd = F._build_province_dict(F.latent, F.TRAINVAL_YEARS)
    test_pd = F._build_province_dict(F.latent, F.TEST_YEARS)
    quality_df = F.compute_province_quality_scores(train_pd)
    included_provinces = set(quality_df.loc[quality_df['quality_score'] >= F.QUALITY_THRESHOLD, 'province'])
    model_final = F.fit_winning_model(train_pd, val_pd, trainval_pd)
    anchors_df = collect_anchor_scores_and_labels(model_final, test_pd, trainval_pd)
    anchors_df['meets_quality_gate'] = anchors_df['province'].isin(included_provinces)
    anchors_df.to_csv(os.path.join(OUT_DIR, 'roc_auc_anchors_raw.csv'), index=False)
    summary_rows = []
    (auc_all28, n_pos_all28, n_neg_all28) = _safe_auc(anchors_df['label'], anchors_df['score'])
    summary_rows.append({'level': 'pooled_all28', 'province': None, 'roc_auc': auc_all28, 'n_pos': n_pos_all28, 'n_neg': n_neg_all28, 'n_anchors': len(anchors_df)})
    filt_df = anchors_df[anchors_df['meets_quality_gate']]
    (auc_filt26, n_pos_filt26, n_neg_filt26) = _safe_auc(filt_df['label'], filt_df['score'])
    summary_rows.append({'level': 'pooled_filtered26', 'province': None, 'roc_auc': auc_filt26, 'n_pos': n_pos_filt26, 'n_neg': n_neg_filt26, 'n_anchors': len(filt_df)})
    for (prov, g) in anchors_df.groupby('province'):
        (auc_p, n_pos_p, n_neg_p) = _safe_auc(g['label'], g['score'])
        summary_rows.append({'level': 'per_province', 'province': prov, 'roc_auc': auc_p, 'n_pos': n_pos_p, 'n_neg': n_neg_p, 'n_anchors': len(g)})
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUT_DIR, 'roc_auc_summary.csv'), index=False)
    n_undefined = summary_df[(summary_df['level'] == 'per_province') & summary_df['roc_auc'].isna()].shape[0]
    if n_undefined:
        pass
