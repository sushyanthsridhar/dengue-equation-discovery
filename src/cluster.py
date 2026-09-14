import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root: this script now lives one level down, in src/
LATENT_CSV = os.path.join(HERE, 'latents', 'latent_dim10.csv')
PLOTS_DIR = os.path.join(HERE, 'plots')
OUT_CSV = os.path.join(HERE, 'province_clusters.csv')
os.makedirs(PLOTS_DIR, exist_ok=True)
K_MIN = 2
K_MAX = 8
latent = pd.read_csv(LATENT_CSV)
z_cols = [c for c in latent.columns if c.startswith('z')]
province_z = latent.groupby('province')[z_cols].mean().reset_index()
scaler = StandardScaler()
Z_sc = scaler.fit_transform(province_z[z_cols].values)
sil_scores = {}
for k in range(K_MIN, K_MAX + 1):
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    lbl = km.fit_predict(Z_sc)
    sil = silhouette_score(Z_sc, lbl)
    sil_scores[k] = sil
else:
    pass
best_k = max(sil_scores, key=sil_scores.get)
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=20)
labels = km_final.fit_predict(Z_sc)
province_z['cluster_id'] = labels
province_z[['province', 'cluster_id']].to_csv(OUT_CSV, index=False)
for cid in sorted(province_z['cluster_id'].unique()):
    provs = province_z.loc[province_z['cluster_id'] == cid, 'province'].tolist()
else:
    pass
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ks = list(sil_scores.keys())
sils = [sil_scores[k] for k in ks]
axes[0].bar(ks, sils, color=['#2a6099' if k == best_k else '#aac4dd' for k in ks])
axes[0].set_xlabel('Number of clusters k')
axes[0].set_ylabel('Silhouette score')
axes[0].set_title('Cluster selection by silhouette score')
axes[0].axvline(best_k, color='red', linestyle='--', linewidth=0.8)
pca = PCA(n_components=2, random_state=42)
Z_2d = pca.fit_transform(Z_sc)
colors = plt.cm.tab10(np.linspace(0, 1, best_k))
for cid in range(best_k):
    mask = labels == cid
    axes[1].scatter(Z_2d[mask, 0], Z_2d[mask, 1], color=colors[cid], s=60, label=f'Cluster {cid}')
    for i, prov in enumerate(province_z['province'].values):
        if labels[i] == cid:
            axes[1].annotate(prov, (Z_2d[i, 0], Z_2d[i, 1]), fontsize=4.5, alpha=0.7)
        else:
            pass
    else:
        pass
else:
    pass
axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)')
axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)')
axes[1].set_title(f'Province clusters in PCA space  (k={best_k})')
axes[1].legend(fontsize=7, markerscale=0.8)
plt.suptitle(f'Latent-space province clustering  (dim=10, k={best_k})', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, f'province_clusters_k{best_k}.png'), dpi=150)
plt.close()