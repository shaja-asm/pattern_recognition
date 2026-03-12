from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ..shared import LABEL_MAP, RANDOM_STATE, Dataset, _basic_metrics, _ensure_outdir, _save_confusion, _save_json


def analysis_gmm(ds: Dataset, outdir: str, n_components_max: int = 3, test_size: float = 0.25):
    outdir = _ensure_outdir(outdir)
    X, y = ds.X, ds.y
    labels_sorted = sorted(np.unique(y))
    label_names = [LABEL_MAP[int(l)] for l in labels_sorted]

    xtr, xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y)

    scaler = StandardScaler()
    xtr_s = scaler.fit_transform(xtr)
    xte_s = scaler.transform(xte)

    gmm_by_class: Dict[int, GaussianMixture] = {}
    bic_rows = []
    for lab in labels_sorted:
        xc = xtr_s[ytr == lab]
        best_gmm = None
        best_bic = np.inf
        best_k = 1
        for k in range(1, n_components_max + 1):
            gmm = GaussianMixture(
                n_components=k,
                covariance_type="diag",
                random_state=RANDOM_STATE,
                n_init=3,
                reg_covar=3e-5,
            )
            gmm.fit(xc)
            bic = gmm.bic(xc)
            bic_rows.append((lab, k, bic))
            if bic < best_bic:
                best_bic = bic
                best_gmm = gmm
                best_k = k
        gmm_by_class[int(lab)] = best_gmm
        print(f"[GMM] Class {LABEL_MAP[int(lab)]}: best #components by BIC = {best_k}")

    bic_df = pd.DataFrame(bic_rows, columns=["class_label", "n_components", "bic"])
    bic_df["class_name"] = bic_df["class_label"].map(LABEL_MAP)
    bic_df.to_csv(os.path.join(outdir, "gmm_bic_grid.csv"), index=False)

    priors = {int(l): float(np.mean(ytr == l)) for l in labels_sorted}
    log_prior = np.array([np.log(priors[int(l)]) for l in labels_sorted])

    loglik = np.zeros((len(xte_s), len(labels_sorted)), dtype=float)
    for j, lab in enumerate(labels_sorted):
        loglik[:, j] = gmm_by_class[int(lab)].score_samples(xte_s)
    logpost = loglik + log_prior
    yhat = np.array([labels_sorted[i] for i in np.argmax(logpost, axis=1)], dtype=int)

    metrics = _basic_metrics(yte, yhat)
    _save_json(metrics, os.path.join(outdir, "gmm_metrics.json"))

    _save_confusion(
        y_true=yte,
        y_pred=yhat,
        labels=labels_sorted,
        label_names=label_names,
        outpath=os.path.join(outdir, "gmm_confusion_matrix.png"),
        title="GMM classifier (one diag-GMM per class, test set)",
    )

    if 1 in np.unique(y):

        def gmm_moments_diag(g: GaussianMixture) -> Tuple[np.ndarray, np.ndarray]:
            w = g.weights_
            mu = g.means_
            var = g.covariances_
            mean = (w[:, None] * mu).sum(axis=0)
            second = (w[:, None] * (var + mu**2)).sum(axis=0)
            variance = np.maximum(second - mean**2, 1e-12)
            return mean, variance

        mu_ck, var_ck = gmm_moments_diag(gmm_by_class[1])
        other_labels = [int(l) for l in labels_sorted if int(l) != 1]
        w_other = np.array([priors[l] for l in other_labels], dtype=float)
        w_other /= w_other.sum()

        mu_rest = np.zeros_like(mu_ck)
        sec_rest = np.zeros_like(mu_ck)
        for w, lab in zip(w_other, other_labels):
            mu_c, var_c = gmm_moments_diag(gmm_by_class[lab])
            mu_rest += w * mu_c
            sec_rest += w * (var_c + mu_c**2)
        var_rest = np.maximum(sec_rest - mu_rest**2, 1e-12)

        d = (mu_ck - mu_rest) / np.sqrt(0.5 * (var_ck + var_rest))
        eff_df = pd.DataFrame(
            {
                "feature": ds.feature_names,
                "gmm_effect_size_ckdu_vs_rest": d,
                "abs_effect_size": np.abs(d),
            }
        ).sort_values("abs_effect_size", ascending=False)
        eff_df.to_csv(os.path.join(outdir, "gmm_feature_effect_size_ckdu_vs_rest.csv"), index=False)

    print(f"\n[GMM] Saved plots/tables to: {outdir}")
