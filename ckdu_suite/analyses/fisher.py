from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..shared import (
    LABEL_MAP,
    RANDOM_STATE,
    Dataset,
    _basic_metrics,
    _ensure_outdir,
    _save_confusion,
)


def analysis_fisher(ds: Dataset, outdir: str, n_splits: int = 5):
    outdir = _ensure_outdir(outdir)
    X, y = ds.X, ds.y
    labels_sorted = sorted(np.unique(y))
    label_names = [LABEL_MAP[int(l)] for l in labels_sorted]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    y_pred_oof = np.zeros_like(y)
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lda", LinearDiscriminantAnalysis(solver="svd")),
        ]
    )
    for tr, te in skf.split(X, y):
        pipe.fit(X[tr], y[tr])
        y_pred_oof[te] = pipe.predict(X[te])

    metrics = _basic_metrics(y, y_pred_oof)
    with open(os.path.join(outdir, "fisher_lda_cv_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    _save_confusion(
        y_true=y,
        y_pred=y_pred_oof,
        labels=labels_sorted,
        label_names=label_names,
        outpath=os.path.join(outdir, "fisher_lda_confusion_matrix.png"),
        title=f"Fisher LDA (OOF, {n_splits}-fold CV)",
    )
    print("\n[Fisher/LDA] CV metrics:")
    print(json.dumps(metrics, indent=2))

    pipe.fit(X, y)
    lda: LinearDiscriminantAnalysis = pipe.named_steps["lda"]
    coef = getattr(lda, "coef_", None)
    classes = lda.classes_
    if coef is not None and 1 in classes:
        ckdu_idx = np.where(classes == 1)[0][0]
        ckdu_coef = coef[ckdu_idx]
        coef_df = pd.DataFrame(
            {
                "feature": ds.feature_names,
                "coef_ckdu_ovr": ckdu_coef,
                "abs_coef": np.abs(ckdu_coef),
            }
        ).sort_values("abs_coef", ascending=False)
        coef_df.to_csv(os.path.join(outdir, "fisher_lda_ckdu_coefficients.csv"), index=False)
        print("\n[Fisher/LDA] Top features by |LDA coef| for CKDu (one-vs-rest):")
        print(coef_df.head(12).to_string(index=False))

    try:
        scaler: StandardScaler = pipe.named_steps["scaler"]
        xs = scaler.transform(X)
        x_lda = lda.transform(xs)
        if x_lda.shape[1] >= 2:
            fig, ax = plt.subplots(figsize=(7.0, 6.2))
            for lab in labels_sorted:
                idx = y == lab
                ax.scatter(
                    x_lda[idx, 0],
                    x_lda[idx, 1],
                    s=28,
                    alpha=0.8,
                    label=LABEL_MAP.get(int(lab), str(lab)),
                )
            ax.set_xlabel("LD1")
            ax.set_ylabel("LD2")
            ax.set_title("LDA projection: LD1 vs LD2")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best", frameon=True)
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, "fisher_lda_scatter_ld1_ld2.png"), dpi=180)
            plt.close(fig)
    except Exception:
        pass

    print(f"\n[Fisher/LDA] Saved plots/tables to: {outdir}")
