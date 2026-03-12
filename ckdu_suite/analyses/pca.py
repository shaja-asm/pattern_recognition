from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..shared import (
    LABEL_MAP,
    RANDOM_STATE,
    REVERSE_F_ALIAS_MAP,
    Dataset,
    _ensure_outdir,
)


def analysis_pca(ds: Dataset, outdir: str, n_components: int = 10):
    outdir = _ensure_outdir(outdir)
    X, y = ds.X, ds.y
    labels_sorted = sorted(np.unique(y))

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=min(n_components, X.shape[1]), random_state=RANDOM_STATE)),
        ]
    )
    X_pca = pipe.fit_transform(X)
    pca: PCA = pipe.named_steps["pca"]

    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(np.arange(1, len(evr) + 1), evr, marker="o", linewidth=1.5, label="Explained variance")
    ax.plot(np.arange(1, len(evr) + 1), cum, marker="s", linewidth=1.5, label="Cumulative")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Variance ratio")
    ax.set_title("PCA explained variance")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "pca_explained_variance.png"), dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 6.2))
    for lab in labels_sorted:
        idx = y == lab
        ax.scatter(X_pca[idx, 0], X_pca[idx, 1], s=28, alpha=0.8, label=LABEL_MAP.get(int(lab), str(lab)))
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA projection: PC1 vs PC2")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "pca_scatter_pc1_pc2.png"), dpi=180)
    plt.close(fig)

    loadings = pca.components_.T
    load_df = pd.DataFrame(
        loadings,
        index=ds.feature_names,
        columns=[f"PC{i}" for i in range(1, loadings.shape[1] + 1)],
    )
    load_df.to_csv(os.path.join(outdir, "pca_loadings.csv"))

    ckdu_label = 1
    if ckdu_label in labels_sorted:
        centroids = {lab: X_pca[y == lab].mean(axis=0) for lab in labels_sorted}
        overall = X_pca.mean(axis=0)
        d = centroids[ckdu_label] - overall
        m = min(n_components, len(d))
        feat_scores = loadings[:, :m] @ d[:m]
        sep_df = pd.DataFrame(
            {
                "feature": ds.feature_names,
                "pca_ckdu_separation_score": feat_scores,
                "abs_score": np.abs(feat_scores),
                "legacy_alias": [REVERSE_F_ALIAS_MAP.get(f, "") for f in ds.feature_names],
            }
        ).sort_values("abs_score", ascending=False)
        sep_df.to_csv(os.path.join(outdir, "pca_ckdu_separation_feature_scores.csv"), index=False)
        print("\n[PCA] Top features driving CKDu centroid separation (|score|):")
        print(sep_df.head(12).to_string(index=False))

    print(f"\n[PCA] Saved plots/tables to: {outdir}")
