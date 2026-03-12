from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..shared import (
    LABEL_MAP,
    RANDOM_STATE,
    Dataset,
    _basic_metrics,
    _ensure_outdir,
    _make_multinomial_logreg,
    _safe_multiclass_auc,
    _save_confusion,
    _save_json,
)


def analysis_regression(ds: Dataset, outdir: str, test_size: float = 0.25):
    outdir = _ensure_outdir(outdir)
    X, y = ds.X, ds.y
    labels_sorted = sorted(np.unique(y))
    label_names = [LABEL_MAP[int(l)] for l in labels_sorted]

    xtr, xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y)

    logreg = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", _make_multinomial_logreg(class_weight="balanced")),
        ]
    )
    logreg.fit(xtr, ytr)
    yhat = logreg.predict(xte)
    proba = logreg.predict_proba(xte)

    metrics = _basic_metrics(yte, yhat)
    metrics["auc_macro_ovr"] = _safe_multiclass_auc(yte, proba)
    _save_json(metrics, os.path.join(outdir, "logreg_metrics.json"))

    _save_confusion(
        y_true=yte,
        y_pred=yhat,
        labels=labels_sorted,
        label_names=label_names,
        outpath=os.path.join(outdir, "logreg_confusion_matrix.png"),
        title="Multinomial Logistic Regression (test set)",
    )

    clf: LogisticRegression = logreg.named_steps["clf"]
    coefs = clf.coef_
    classes = clf.classes_

    coef_df = pd.DataFrame(coefs, index=[LABEL_MAP[int(c)] for c in classes], columns=ds.feature_names)
    coef_df.to_csv(os.path.join(outdir, "logreg_coefficients_all_classes.csv"))

    if 1 in classes:
        ckdu_idx = np.where(classes == 1)[0][0]
        ck = pd.DataFrame(
            {
                "feature": ds.feature_names,
                "coef_ckdu": coefs[ckdu_idx],
                "abs_coef": np.abs(coefs[ckdu_idx]),
            }
        ).sort_values("abs_coef", ascending=False)
        ck.to_csv(os.path.join(outdir, "logreg_coefficients_ckdu.csv"), index=False)

    if 1 in np.unique(y):
        y_bin = (y == 1).astype(int)
        xtr, xte, ytr, yte = train_test_split(
            X, y_bin, test_size=test_size, random_state=RANDOM_STATE, stratify=y_bin
        )
        ridge = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("reg", Ridge(alpha=1.2, random_state=RANDOM_STATE)),
            ]
        )
        ridge.fit(xtr, ytr)
        yscore = ridge.predict(xte)
        yhat = (yscore >= 0.5).astype(int)
        metrics_r = {
            "accuracy": float(accuracy_score(yte, yhat)),
            "f1": float(f1_score(yte, yhat)),
            "auc": float(roc_auc_score(yte, yscore)),
        }
        _save_json(metrics_r, os.path.join(outdir, "ridge_ckdu_vs_rest_metrics.json"))

        w = ridge.named_steps["reg"].coef_
        w_df = pd.DataFrame(
            {
                "feature": ds.feature_names,
                "ridge_w": w,
                "abs_w": np.abs(w),
            }
        ).sort_values("abs_w", ascending=False)
        w_df.to_csv(os.path.join(outdir, "ridge_ckdu_vs_rest_coefficients.csv"), index=False)

    print(f"\n[Regression] Saved plots/tables to: {outdir}")
