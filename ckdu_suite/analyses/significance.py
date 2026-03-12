from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from ..shared import Dataset, _ensure_outdir, _neglog10p


def analysis_significance(ds: Dataset, outdir: str):
    outdir = _ensure_outdir(outdir)
    X, y = ds.X, ds.y
    feats = ds.feature_names

    labels_sorted = sorted(np.unique(y))
    groups = [X[y == lab] for lab in labels_sorted]

    anova_p, kruskal_p = [], []
    for j in range(X.shape[1]):
        cols = [g[:, j] for g in groups]
        anova_p.append(stats.f_oneway(*cols).pvalue)
        kruskal_p.append(stats.kruskal(*cols).pvalue)

    df_mc = pd.DataFrame(
        {
            "feature": feats,
            "anova_p": anova_p,
            "anova_neglog10p": _neglog10p(anova_p),
            "kruskal_p": kruskal_p,
            "kruskal_neglog10p": _neglog10p(kruskal_p),
        }
    ).sort_values("anova_p")
    df_mc.to_csv(os.path.join(outdir, "significance_multiclass_anova_kruskal.csv"), index=False)

    if 1 in np.unique(y):
        x1, x0 = X[y == 1], X[y != 1]
        t_p, mw_p = [], []
        for j in range(X.shape[1]):
            t_p.append(stats.ttest_ind(x1[:, j], x0[:, j], equal_var=False).pvalue)
            mw_p.append(stats.mannwhitneyu(x1[:, j], x0[:, j], alternative="two-sided").pvalue)

        df_bin = pd.DataFrame(
            {
                "feature": feats,
                "t_p": t_p,
                "t_neglog10p": _neglog10p(t_p),
                "mw_p": mw_p,
                "mw_neglog10p": _neglog10p(mw_p),
            }
        ).sort_values("t_p")
        df_bin.to_csv(os.path.join(outdir, "significance_ckdu_vs_rest_ttest_mwu.csv"), index=False)

    plot_df = df_mc.sort_values("anova_neglog10p", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.bar(np.arange(len(plot_df)), plot_df["anova_neglog10p"].to_numpy())
    ax.set_xlabel("Features (sorted)")
    ax.set_ylabel(r"$-\log_{10}(p)$")
    ax.set_title("Feature significance across 5 classes (ANOVA)")
    ax.grid(True, axis="y", alpha=0.25)
    for i in range(min(12, len(plot_df))):
        ax.text(
            i,
            plot_df.loc[i, "anova_neglog10p"] + 0.05,
            plot_df.loc[i, "feature"],
            rotation=90,
            fontsize=8,
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "significance_anova_neglog10p.png"), dpi=180)
    plt.close(fig)

    print(f"\n[Significance] Saved plots/tables to: {outdir}")
