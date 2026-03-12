from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..shared import (
    LABEL_MAP,
    RANDOM_STATE,
    Dataset,
    _basic_metrics,
    _ensure_outdir,
    _safe_multiclass_auc,
    _save_confusion,
    _save_json,
)


def _log_gauss_pdf(x: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    eps = 1e-9
    var = np.maximum(var, eps)
    return -0.5 * np.log(2 * np.pi * var) - 0.5 * ((x - mean) ** 2) / var


def analysis_bayes(ds: Dataset, outdir: str, n_splits: int = 5):
    outdir = _ensure_outdir(outdir)
    X, y = ds.X, ds.y
    labels_sorted = sorted(np.unique(y))
    label_names = [LABEL_MAP[int(l)] for l in labels_sorted]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    y_pred_oof = np.zeros_like(y)
    y_proba_oof = np.zeros((len(y), len(labels_sorted)), dtype=float)

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("nb", GaussianNB(var_smoothing=1e-3)),
        ]
    )

    for tr, te in skf.split(X, y):
        pipe.fit(X[tr], y[tr])
        y_pred_oof[te] = pipe.predict(X[te])
        proba = pipe.predict_proba(X[te])
        classes = pipe.named_steps["nb"].classes_
        col_idx = [np.where(classes == l)[0][0] for l in labels_sorted]
        y_proba_oof[te, :] = proba[:, col_idx]

    metrics = _basic_metrics(y, y_pred_oof)
    metrics["auc_macro_ovr"] = _safe_multiclass_auc(y, y_proba_oof)
    _save_json(metrics, os.path.join(outdir, "bayes_cv_metrics.json"))

    _save_confusion(
        y_true=y,
        y_pred=y_pred_oof,
        labels=labels_sorted,
        label_names=label_names,
        outpath=os.path.join(outdir, "bayes_confusion_matrix.png"),
        title=f"Gaussian Naive Bayes (OOF, {n_splits}-fold CV)",
    )

    print("\n[Bayes] CV metrics:")
    print(json.dumps(metrics, indent=2))
    print("\n[Bayes] Classification report (OOF):")
    print(classification_report(y, y_pred_oof, target_names=label_names, digits=3))

    single_rows = []
    for j, feat in enumerate(ds.feature_names):
        y_pred = np.zeros_like(y)
        for tr, te in skf.split(X[:, [j]], y):
            pipe.fit(X[tr][:, [j]], y[tr])
            y_pred[te] = pipe.predict(X[te][:, [j]])
        single_rows.append((feat, f1_score(y, y_pred, average="macro"), accuracy_score(y, y_pred)))
    single_df = pd.DataFrame(single_rows, columns=["feature", "cv_f1_macro", "cv_accuracy"]).sort_values(
        "cv_f1_macro", ascending=False
    )
    single_df.to_csv(os.path.join(outdir, "bayes_single_feature_ranking.csv"), index=False)

    base_f1 = metrics["f1_macro"]
    loo_rows = []
    for j, feat in enumerate(ds.feature_names):
        cols = [i for i in range(X.shape[1]) if i != j]
        y_pred = np.zeros_like(y)
        for tr, te in skf.split(X[:, cols], y):
            pipe.fit(X[tr][:, cols], y[tr])
            y_pred[te] = pipe.predict(X[te][:, cols])
        f1m = f1_score(y, y_pred, average="macro")
        loo_rows.append((feat, base_f1 - f1m))
    loo_df = pd.DataFrame(loo_rows, columns=["feature", "f1_drop_when_removed"]).sort_values(
        "f1_drop_when_removed", ascending=False
    )
    loo_df.to_csv(os.path.join(outdir, "bayes_leave_one_out_importance.csv"), index=False)

    pipe.fit(X, y)
    nb: GaussianNB = pipe.named_steps["nb"]
    scaler: StandardScaler = pipe.named_steps["scaler"]
    xs = scaler.transform(X)
    mu = nb.theta_
    var = nb.var_
    classes = nb.classes_

    if 1 in classes:
        idx_ckdu = np.where(classes == 1)[0][0]
        other_idx = [i for i, c in enumerate(classes) if c != 1]
        llr_abs_accum = np.zeros(X.shape[1])
        llr_signed_ckdu_accum = np.zeros(X.shape[1])
        for k in other_idx:
            llr = _log_gauss_pdf(xs, mu[idx_ckdu], var[idx_ckdu]) - _log_gauss_pdf(xs, mu[k], var[k])
            llr_abs_accum += np.mean(np.abs(llr), axis=0)
            llr_signed_ckdu_accum += np.mean(llr[y == 1], axis=0)

        llr_df = pd.DataFrame(
            {
                "feature": ds.feature_names,
                "avg_abs_llr_vs_each_class": llr_abs_accum / max(len(other_idx), 1),
                "avg_signed_llr_on_ckdu_samples": llr_signed_ckdu_accum / max(len(other_idx), 1),
            }
        ).sort_values("avg_abs_llr_vs_each_class", ascending=False)
        llr_df.to_csv(os.path.join(outdir, "bayes_llr_feature_contributions.csv"), index=False)

    print(f"\n[Bayes] Saved plots/tables to: {outdir}")
