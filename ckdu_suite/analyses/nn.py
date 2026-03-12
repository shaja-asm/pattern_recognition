from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..shared import (
    LABEL_MAP,
    RANDOM_STATE,
    Dataset,
    _basic_metrics,
    _ensure_outdir,
    _make_regularized_mlp,
    _safe_multiclass_auc,
    _save_confusion,
    _save_json,
    _top_bar_plot,
)


def analysis_nn(ds: Dataset, outdir: str, test_size: float = 0.25):
    outdir = _ensure_outdir(outdir)
    X, y = ds.X, ds.y
    labels_sorted = sorted(np.unique(y))
    label_names = [LABEL_MAP[int(l)] for l in labels_sorted]

    xtr, xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y)

    mlp = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                _make_regularized_mlp(
                    hidden_layer_sizes=(32, 16),
                    random_state=RANDOM_STATE,
                    max_iter=1500,
                    n_iter_no_change=20,
                ),
            ),
        ]
    )
    mlp.fit(xtr, ytr)
    yhat = mlp.predict(xte)
    proba = mlp.predict_proba(xte)

    metrics = _basic_metrics(yte, yhat)
    metrics["auc_macro_ovr"] = _safe_multiclass_auc(yte, proba)
    _save_json(metrics, os.path.join(outdir, "mlp_metrics.json"))

    _save_confusion(
        y_true=yte,
        y_pred=yhat,
        labels=labels_sorted,
        label_names=label_names,
        outpath=os.path.join(outdir, "mlp_confusion_matrix.png"),
        title="MLP neural network (test set)",
    )

    scorer = make_scorer(f1_score, average="macro")
    imp = permutation_importance(
        mlp,
        xte,
        yte,
        scoring=scorer,
        n_repeats=8,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    imp_df = pd.DataFrame(
        {
            "feature": ds.feature_names,
            "perm_importance_mean": imp.importances_mean,
            "perm_importance_std": imp.importances_std,
        }
    ).sort_values("perm_importance_mean", ascending=False)
    imp_df.to_csv(os.path.join(outdir, "mlp_permutation_importance.csv"), index=False)
    _top_bar_plot(
        imp_df,
        value_col="perm_importance_mean",
        title="MLP permutation feature importance (top 15)",
        outpath=os.path.join(outdir, "mlp_permutation_importance_top15.png"),
        top_n=15,
    )

    print(f"\n[NN/MLP] Saved plots/tables to: {outdir}")
