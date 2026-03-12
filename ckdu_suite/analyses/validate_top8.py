from __future__ import annotations

import os
from typing import List

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from ..shared import (
    LABEL_MAP,
    RANDOM_STATE,
    REVERSE_F_ALIAS_MAP,
    Dataset,
    _align_proba,
    _basic_metrics,
    _ensure_outdir,
    _make_multinomial_logreg,
    _make_regularized_mlp,
    _safe_multiclass_auc,
    _save_confusion,
    _save_json,
    make_preprocessor,
    resolve_feature_names,
)


def analysis_validate_top_features(
    ds: Dataset,
    outdir: str,
    top_features: List[str],
    preproc_kind: str = "robust",
    n_splits: int = 5,
    perm_repeats: int = 10,
):
    outdir = _ensure_outdir(outdir)

    top_features = resolve_feature_names(top_features, ds.feature_names)
    name_to_idx = {n: i for i, n in enumerate(ds.feature_names)}
    col_idx = [name_to_idx[f] for f in top_features]
    X = ds.X[:, col_idx]
    y = ds.y

    _save_json(
        {
            "top_features": top_features,
            "top_features_alias": [REVERSE_F_ALIAS_MAP.get(f, "") for f in top_features],
            "preproc_kind": preproc_kind,
            "n_splits": n_splits,
            "perm_repeats": perm_repeats,
        },
        os.path.join(outdir, "consensus_features_used.json"),
    )

    labels_sorted = sorted(np.unique(y))
    label_names = [LABEL_MAP[int(l)] for l in labels_sorted]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    preproc = make_preprocessor(preproc_kind)

    logreg_pipe = Pipeline(
        [
            ("pre", preproc),
            ("clf", _make_multinomial_logreg(class_weight="balanced")),
        ]
    )

    oof_pred_lr = np.zeros_like(y)
    oof_proba_lr = np.zeros((len(y), len(labels_sorted)), dtype=float)
    fold_metrics_lr = []
    coef_by_fold = []

    for fold, (tr, te) in enumerate(skf.split(X, y), start=1):
        model = clone(logreg_pipe)
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])
        proba = model.predict_proba(X[te])

        classes = model.named_steps["clf"].classes_
        proba_aligned = _align_proba(proba, classes, np.asarray(labels_sorted))

        oof_pred_lr[te] = pred
        oof_proba_lr[te, :] = proba_aligned

        m = _basic_metrics(y[te], pred)
        m["auc_macro_ovr"] = _safe_multiclass_auc(y[te], proba_aligned)
        m["fold"] = fold
        fold_metrics_lr.append(m)

        clf: LogisticRegression = model.named_steps["clf"]
        if 1 in clf.classes_:
            ckdu_idx = int(np.where(clf.classes_ == 1)[0][0])
            coef_by_fold.append(clf.coef_[ckdu_idx].copy())

    lr_metrics_df = pd.DataFrame(fold_metrics_lr)
    lr_metrics_df.to_csv(os.path.join(outdir, "logreg_cv_fold_metrics.csv"), index=False)

    lr_summary = {
        k: {
            "mean": float(np.nanmean(lr_metrics_df[k].to_numpy())),
            "std": float(np.nanstd(lr_metrics_df[k].to_numpy(), ddof=1)),
        }
        for k in ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "auc_macro_ovr"]
        if k in lr_metrics_df.columns
    }
    _save_json(lr_summary, os.path.join(outdir, "logreg_cv_summary.json"))

    _save_confusion(
        y_true=y,
        y_pred=oof_pred_lr,
        labels=labels_sorted,
        label_names=label_names,
        outpath=os.path.join(outdir, "logreg_oof_confusion_matrix.png"),
        title=f"LogReg + {preproc_kind} preproc (OOF, {n_splits}-fold CV)",
    )

    with open(os.path.join(outdir, "logreg_oof_classification_report.txt"), "w") as f:
        f.write(classification_report(y, oof_pred_lr, target_names=label_names, digits=3))

    if coef_by_fold:
        coef_mat = np.vstack(coef_by_fold)
        coef_summary_df = pd.DataFrame(
            {
                "feature": top_features,
                "coef_mean": coef_mat.mean(axis=0),
                "coef_std": coef_mat.std(axis=0, ddof=1),
                "abs_coef_mean": np.abs(coef_mat).mean(axis=0),
                "abs_coef_std": np.abs(coef_mat).std(axis=0, ddof=1),
            }
        ).sort_values("abs_coef_mean", ascending=False)
        coef_summary_df.to_csv(os.path.join(outdir, "logreg_ckdu_coef_cv_summary.csv"), index=False)

    mlp_pipe = Pipeline(
        [
            ("pre", preproc),
            (
                "clf",
                _make_regularized_mlp(
                    hidden_layer_sizes=(16, 8),
                    random_state=RANDOM_STATE,
                    max_iter=2000,
                    n_iter_no_change=25,
                ),
            ),
        ]
    )

    oof_pred_mlp = np.zeros_like(y)
    oof_proba_mlp = np.zeros((len(y), len(labels_sorted)), dtype=float)
    fold_metrics_mlp = []
    imp_by_fold = []

    scorer = make_scorer(f1_score, average="macro")

    for fold, (tr, te) in enumerate(skf.split(X, y), start=1):
        model = clone(mlp_pipe)
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])
        proba = model.predict_proba(X[te])

        classes = model.named_steps["clf"].classes_
        proba_aligned = _align_proba(proba, classes, np.asarray(labels_sorted))

        oof_pred_mlp[te] = pred
        oof_proba_mlp[te, :] = proba_aligned

        m = _basic_metrics(y[te], pred)
        m["auc_macro_ovr"] = _safe_multiclass_auc(y[te], proba_aligned)
        m["fold"] = fold
        fold_metrics_mlp.append(m)

        imp = permutation_importance(
            model,
            X[te],
            y[te],
            scoring=scorer,
            n_repeats=perm_repeats,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        imp_by_fold.append(imp.importances_mean.copy())

    mlp_metrics_df = pd.DataFrame(fold_metrics_mlp)
    mlp_metrics_df.to_csv(os.path.join(outdir, "mlp_cv_fold_metrics.csv"), index=False)

    mlp_summary = {
        k: {
            "mean": float(np.nanmean(mlp_metrics_df[k].to_numpy())),
            "std": float(np.nanstd(mlp_metrics_df[k].to_numpy(), ddof=1)),
        }
        for k in ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "auc_macro_ovr"]
        if k in mlp_metrics_df.columns
    }
    _save_json(mlp_summary, os.path.join(outdir, "mlp_cv_summary.json"))

    _save_confusion(
        y_true=y,
        y_pred=oof_pred_mlp,
        labels=labels_sorted,
        label_names=label_names,
        outpath=os.path.join(outdir, "mlp_oof_confusion_matrix.png"),
        title=f"MLP + {preproc_kind} preproc (OOF, {n_splits}-fold CV)",
    )

    if imp_by_fold:
        imp_mat = np.vstack(imp_by_fold)
        imp_summary_df = pd.DataFrame(
            {
                "feature": top_features,
                "perm_importance_mean": imp_mat.mean(axis=0),
                "perm_importance_std": imp_mat.std(axis=0, ddof=1),
            }
        ).sort_values("perm_importance_mean", ascending=False)
        imp_summary_df.to_csv(os.path.join(outdir, "mlp_perm_importance_cv_summary.csv"), index=False)

    print("\n[Validate Top Features] Completed.")
    print(f"[Validate Top Features] Features used: {top_features}")
    print(f"[Validate Top Features] Saved outputs to: {outdir}")
