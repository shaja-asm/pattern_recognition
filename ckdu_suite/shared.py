from __future__ import annotations

import inspect
import json
import os
import re
import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, RobustScaler, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_STATE = 42

LABEL_MAP = {
    1: "CKDu",
    2: "EC",
    3: "NEC",
    4: "ECKD",
    5: "NECKD",
}

METAL_FEATURE_NAMES = [
    "Na",
    "Mg",
    "K",
    "Ca",
    "Li",
    "Be",
    "Al",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "As",
    "Se",
    "Rb",
    "Sr",
    "Ag",
    "Cd",
    "In",
    "Cs",
    "Ba",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "U",
]

F_ALIAS_MAP = {f"F{i + 1}": name for i, name in enumerate(METAL_FEATURE_NAMES)}
REVERSE_F_ALIAS_MAP = {v: k for k, v in F_ALIAS_MAP.items()}


@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    feature_names: List[str]
    class_names: List[str]


def _ensure_outdir(outdir: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    return outdir


def _norm_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def _make_feature_alias_lookup(feature_names: Iterable[str]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for feat in feature_names:
        lookup[_norm_name(feat)] = feat
        if feat in REVERSE_F_ALIAS_MAP:
            lookup[_norm_name(REVERSE_F_ALIAS_MAP[feat])] = feat
    return lookup


def resolve_feature_names(requested_features: List[str], feature_names: List[str]) -> List[str]:
    lookup = _make_feature_alias_lookup(feature_names)
    resolved: List[str] = []
    missing: List[str] = []
    for item in requested_features:
        key = _norm_name(item)
        if key in lookup:
            resolved.append(lookup[key])
        else:
            missing.append(item)
    if missing:
        raise ValueError(
            f"Could not resolve these feature names: {missing}. "
            f"Use actual names such as {feature_names[:8]} or aliases like F1..F30."
        )
    return resolved


def load_ckdu_processed(csv_path: str) -> Dataset:
    df = pd.read_csv(csv_path, header=None)
    if df.shape[1] < 1 + len(METAL_FEATURE_NAMES):
        raise ValueError(
            f"Expected at least {1 + len(METAL_FEATURE_NAMES)} columns "
            f"(label + {len(METAL_FEATURE_NAMES)} metals), got {df.shape[1]}"
        )

    y = df.iloc[:, 0].astype(int).to_numpy()
    mask = np.isin(y, list(LABEL_MAP.keys()))
    df = df.loc[mask].reset_index(drop=True)
    y = y[mask]

    feature_df = df.iloc[:, 1:].copy()
    feature_df = feature_df.apply(pd.to_numeric, errors="coerce")

    n_features = feature_df.shape[1]
    if n_features == len(METAL_FEATURE_NAMES):
        metal_df = feature_df.iloc[:, : len(METAL_FEATURE_NAMES)].copy()
    elif n_features >= len(METAL_FEATURE_NAMES) + 2:
        metal_df = feature_df.iloc[:, 2 : 2 + len(METAL_FEATURE_NAMES)].copy()
    else:
        raise ValueError(
            "Unsupported feature-column count after label: "
            f"{n_features}. Expected either 30 metals-only columns, "
            "or at least 32 columns including Age/S.cr + 30 metals."
        )

    if metal_df.shape[1] != len(METAL_FEATURE_NAMES):
        raise ValueError(
            f"Expected exactly {len(METAL_FEATURE_NAMES)} metal features after filtering, "
            f"got {metal_df.shape[1]}."
        )

    metal_df.columns = METAL_FEATURE_NAMES

    X = metal_df.to_numpy(dtype=float)
    final_feature_names = list(metal_df.columns)
    class_names = [LABEL_MAP[i] for i in sorted(np.unique(y))]
    return Dataset(X=X, y=y, feature_names=final_feature_names, class_names=class_names)


def _make_multinomial_logreg(class_weight=None) -> LogisticRegression:
    kwargs = dict(
        solver="lbfgs",
        max_iter=5000,
        C=0.8,
        tol=1e-4,
        random_state=RANDOM_STATE,
        class_weight=class_weight,
    )
    if "multi_class" in inspect.signature(LogisticRegression).parameters:
        kwargs["multi_class"] = "multinomial"
    return LogisticRegression(**kwargs)


def _make_regularized_lda(priors: Optional[np.ndarray] = None) -> LinearDiscriminantAnalysis:
    return LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto", priors=priors)


def _make_regularized_mlp(
    hidden_layer_sizes: Tuple[int, int],
    random_state: int,
    max_iter: int = 1600,
    n_iter_no_change: int = 25,
) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        alpha=3e-4,
        learning_rate_init=1e-3,
        learning_rate="adaptive",
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.10,
        n_iter_no_change=n_iter_no_change,
        random_state=random_state,
    )


class QuantileClipper(BaseEstimator, TransformerMixin):
    def __init__(self, low_q: float = 0.01, high_q: float = 0.99):
        self.low_q = low_q
        self.high_q = high_q

    def fit(self, X, y=None):
        X = np.asarray(X)
        self.lower_ = np.nanquantile(X, self.low_q, axis=0)
        self.upper_ = np.nanquantile(X, self.high_q, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X)
        return np.clip(X, self.lower_, self.upper_)


class PowerTransformerSafe(BaseEstimator, TransformerMixin):
    """Yeo-Johnson that gracefully handles near-constant columns."""

    def __init__(self):
        self.transformers_: List[Optional[PowerTransformer]] = []
        self.constant_values_: List[float] = []

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.transformers_ = []
        self.constant_values_ = []
        for j in range(X.shape[1]):
            col = X[:, [j]]
            finite_col = col[np.isfinite(col)]
            if finite_col.size == 0:
                self.transformers_.append(None)
                self.constant_values_.append(0.0)
                continue
            if np.nanstd(finite_col) < 1e-12:
                self.transformers_.append(None)
                self.constant_values_.append(float(np.nanmedian(finite_col)))
                continue
            pt = PowerTransformer(method="yeo-johnson", standardize=False)
            pt.fit(col)
            self.transformers_.append(pt)
            self.constant_values_.append(float(np.nanmedian(finite_col)))
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        Xt = np.zeros_like(X, dtype=float)
        for j in range(X.shape[1]):
            col = X[:, [j]]
            pt = self.transformers_[j]
            if pt is None:
                fill = self.constant_values_[j]
                col = np.where(np.isfinite(col), col, fill)
                Xt[:, j] = col.ravel()
            else:
                Xt[:, j] = pt.transform(col).ravel()
        return Xt


class IdentityTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray(X)


def make_preprocessor(kind: str = "robust") -> Pipeline:
    kind = kind.lower().strip()
    if kind == "robust":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("clip", QuantileClipper(low_q=0.01, high_q=0.99)),
                ("power", PowerTransformerSafe()),
                ("scaler", RobustScaler(quantile_range=(25.0, 75.0))),
            ]
        )
    if kind == "standard":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
    raise ValueError(f"Unknown preprocessor kind: {kind}")


def _basic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
    }


def _safe_multiclass_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
    except Exception:
        return float("nan")


def _save_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[int],
    label_names: List[str],
    outpath: str,
    title: str,
):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    disp.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def _save_json(obj: dict, outpath: str):
    with open(outpath, "w") as f:
        json.dump(obj, f, indent=2)


def _neglog10p(pvals: np.ndarray) -> np.ndarray:
    return -np.log10(np.maximum(np.asarray(pvals, dtype=float), 1e-300))


def random_oversample(
    X: np.ndarray, y: np.ndarray, random_state: int = RANDOM_STATE
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    classes, counts = np.unique(y, return_counts=True)
    max_count = int(np.max(counts))
    indices: List[np.ndarray] = []
    for c, cnt in zip(classes, counts):
        idx = np.flatnonzero(y == c)
        if cnt < max_count:
            extra = rng.choice(idx, size=max_count - cnt, replace=True)
            idx = np.concatenate([idx, extra])
        indices.append(idx)
    all_idx = np.concatenate(indices)
    rng.shuffle(all_idx)
    return X[all_idx], y[all_idx]


def _uniform_priors(classes: np.ndarray) -> np.ndarray:
    return np.ones(len(classes), dtype=float) / float(len(classes))


def _align_proba(proba: np.ndarray, model_classes: np.ndarray, target_classes: np.ndarray) -> np.ndarray:
    aligned = np.zeros((proba.shape[0], len(target_classes)), dtype=float)
    for j, c in enumerate(target_classes):
        k = np.where(model_classes == c)[0]
        if len(k):
            aligned[:, j] = proba[:, k[0]]
    row_sum = aligned.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return aligned / row_sum


def _top_bar_plot(df: pd.DataFrame, value_col: str, title: str, outpath: str, top_n: int = 15):
    plot_df = df.sort_values(value_col, ascending=False).head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.barh(plot_df["feature"], plot_df[value_col])
    ax.set_xlabel(value_col)
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def _derive_frequency_consensus_features_from_outputs(
    root_outdir: str,
    feature_names: List[str],
    top_k: int = 8,
    source_top_n: int = 8,
) -> Tuple[List[str], pd.DataFrame, pd.DataFrame]:
    source_specs = [
        ("pca_sep", "pca/pca_ckdu_separation_feature_scores.csv", "abs_score", False),
        ("bayes_single", "bayes/bayes_single_feature_ranking.csv", "cv_f1_macro", False),
        ("bayes_loo", "bayes/bayes_leave_one_out_importance.csv", "f1_drop_when_removed", False),
        ("bayes_llr", "bayes/bayes_llr_feature_contributions.csv", "avg_abs_llr_vs_each_class", False),
        ("fisher_coef", "fisher/fisher_lda_ckdu_coefficients.csv", "abs_coef", False),
        ("skew_kurt_delta", "skew_kurt/skew_kurtosis_delta_ckdu_vs_rest.csv", "delta_norm", False),
        ("signif_anova", "significance/significance_multiclass_anova_kruskal.csv", "anova_neglog10p", False),
        ("signif_ttest", "significance/significance_ckdu_vs_rest_ttest_mwu.csv", "t_neglog10p", False),
        ("logreg_ckdu", "regression/logreg_coefficients_ckdu.csv", "abs_coef", False),
        ("ridge_ckdu", "regression/ridge_ckdu_vs_rest_coefficients.csv", "abs_w", False),
        ("gmm_effect", "gmm/gmm_feature_effect_size_ckdu_vs_rest.csv", "abs_effect_size", False),
        ("nn_perm", "nn/mlp_permutation_importance.csv", "perm_importance_mean", False),
    ]

    name_set = set(feature_names)
    counts = {f: 0 for f in feature_names}
    rank_sums = {f: 0.0 for f in feature_names}
    hit_rows: List[Dict[str, object]] = []

    for source_name, rel_path, score_col, ascending in source_specs:
        path = os.path.join(root_outdir, rel_path)
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "feature" not in df.columns or score_col not in df.columns:
            continue

        source_df = df[["feature", score_col]].copy()
        source_df["feature"] = source_df["feature"].astype(str)
        source_df = source_df[source_df["feature"].isin(name_set)]
        source_df = source_df.dropna(subset=[score_col])
        if source_df.empty:
            continue

        source_df = source_df.sort_values(score_col, ascending=ascending).head(max(1, int(source_top_n)))
        for rank, row in enumerate(source_df.itertuples(index=False), start=1):
            feat = str(getattr(row, "feature"))
            score = float(getattr(row, score_col))
            counts[feat] += 1
            rank_sums[feat] += float(rank)
            hit_rows.append(
                {
                    "source": source_name,
                    "feature": feat,
                    "rank_in_source": rank,
                    "source_score": score,
                }
            )

    summary_df = pd.DataFrame(
        {
            "feature": feature_names,
            "appearance_count": [counts[f] for f in feature_names],
            "avg_rank_if_hit": [(rank_sums[f] / counts[f]) if counts[f] > 0 else np.nan for f in feature_names],
            "legacy_alias": [REVERSE_F_ALIAS_MAP.get(f, "") for f in feature_names],
        }
    ).sort_values(["appearance_count", "avg_rank_if_hit", "feature"], ascending=[False, True, True]).reset_index(
        drop=True
    )

    selected = summary_df.loc[summary_df["appearance_count"] > 0, "feature"].head(max(1, int(top_k))).tolist()
    hits_df = pd.DataFrame(hit_rows)
    return selected, summary_df, hits_df
