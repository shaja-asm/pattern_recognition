from __future__ import annotations

import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

from ..shared import Dataset, _ensure_outdir


def analysis_skew_kurt(ds: Dataset, outdir: str):
    outdir = _ensure_outdir(outdir)
    X, y = ds.X, ds.y
    xs = StandardScaler().fit_transform(X)

    def skew_kurt(mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        sk = stats.skew(mat, axis=0, bias=False, nan_policy="omit")
        ku = stats.kurtosis(mat, axis=0, fisher=True, bias=False, nan_policy="omit")
        return sk, ku

    sk_all, ku_all = skew_kurt(xs)
    df_all = pd.DataFrame({"feature": ds.feature_names, "skew": sk_all, "excess_kurtosis": ku_all})
    df_all.to_csv(os.path.join(outdir, "skew_kurtosis_all.csv"), index=False)

    if 1 in np.unique(y):
        sk_ck, ku_ck = skew_kurt(xs[y == 1])
        sk_rest, ku_rest = skew_kurt(xs[y != 1])

        delta = pd.DataFrame(
            {
                "feature": ds.feature_names,
                "delta_skew_ckdu_minus_rest": sk_ck - sk_rest,
                "delta_kurtosis_ckdu_minus_rest": ku_ck - ku_rest,
            }
        )
        delta["delta_norm"] = np.sqrt(delta["delta_skew_ckdu_minus_rest"] ** 2 + delta["delta_kurtosis_ckdu_minus_rest"] ** 2)
        delta = delta.sort_values("delta_norm", ascending=False)
        delta.to_csv(os.path.join(outdir, "skew_kurtosis_delta_ckdu_vs_rest.csv"), index=False)

    def scatter(sk: np.ndarray, ku: np.ndarray, title: str, filename: str):
        fig, ax = plt.subplots(figsize=(7.0, 6.0))
        ax.scatter(sk, ku, s=38, alpha=0.85)
        ax.axvline(0, linewidth=1)
        ax.axhline(0, linewidth=1)
        ax.set_xlabel("Skewness")
        ax.set_ylabel("Excess kurtosis")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        idx = np.argsort(np.abs(sk) + np.abs(ku))[-8:]
        for i in idx:
            ax.annotate(ds.feature_names[i], (sk[i], ku[i]), fontsize=8, xytext=(4, 2), textcoords="offset points")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, filename), dpi=180)
        plt.close(fig)

    scatter(sk_all, ku_all, "Skewness vs kurtosis (all samples)", "skew_kurtosis_scatter_all.png")
    print(f"\n[Skew/Kurt] Saved plots/tables to: {outdir}")
