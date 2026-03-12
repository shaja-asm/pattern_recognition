from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold

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
    _neglog10p,
    _safe_multiclass_auc,
    _save_confusion,
    _save_json,
    _top_bar_plot,
    make_preprocessor,
    random_oversample,
    resolve_feature_names,
)


def _feature_selection_consensus(
    x_proc: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    raw_top_k: int,
) -> pd.DataFrame:
    labels_sorted = sorted(np.unique(y))
    groups = [x_proc[y == lab] for lab in labels_sorted]

    anova_p = []
    for j in range(x_proc.shape[1]):
        cols = [g[:, j] for g in groups]
        anova_p.append(stats.f_oneway(*cols).pvalue)
    anova_score = _neglog10p(anova_p)

    if 1 in np.unique(y):
        x_ck = x_proc[y == 1]
        x_rest = x_proc[y != 1]
        t_p = [stats.ttest_ind(x_ck[:, j], x_rest[:, j], equal_var=False).pvalue for j in range(x_proc.shape[1])]
        ckdu_score = _neglog10p(t_p)
    else:
        ckdu_score = np.zeros(x_proc.shape[1])

    lda = LinearDiscriminantAnalysis(solver="svd")
    lda.fit(x_proc, y)
    if 1 in lda.classes_ and hasattr(lda, "coef_"):
        lda_coef = np.abs(lda.coef_[np.where(lda.classes_ == 1)[0][0]])
    else:
        lda_coef = np.zeros(x_proc.shape[1])

    logreg = _make_multinomial_logreg(class_weight="balanced")
    logreg.fit(x_proc, y)
    if 1 in logreg.classes_:
        log_coef = np.abs(logreg.coef_[np.where(logreg.classes_ == 1)[0][0]])
    else:
        log_coef = np.zeros(x_proc.shape[1])

    df = pd.DataFrame(
        {
            "feature": feature_names,
            "anova_score": anova_score,
            "ckdu_t_score": ckdu_score,
            "lda_abs_coef": lda_coef,
            "logreg_abs_coef": log_coef,
        }
    )
    for col in ["anova_score", "ckdu_t_score", "lda_abs_coef", "logreg_abs_coef"]:
        df[f"rank_{col}"] = df[col].rank(ascending=False, method="average")
    df["avg_rank"] = df[
        [
            "rank_anova_score",
            "rank_ckdu_t_score",
            "rank_lda_abs_coef",
            "rank_logreg_abs_coef",
        ]
    ].mean(axis=1)
    df["selected_for_raw_branch"] = 0
    if raw_top_k > 0:
        selected_idx = df.nsmallest(raw_top_k, "avg_rank").index
        df.loc[selected_idx, "selected_for_raw_branch"] = 1
    return df.sort_values("avg_rank", ascending=True).reset_index(drop=True)


class PCAGuidedFusionClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        preproc_kind: str = "robust",
        pca_variance_threshold: float = 0.95,
        max_pca_components: int = 12,
        raw_top_k: int = 8,
        internal_cv: int = 3,
        random_state: int = RANDOM_STATE,
        base_weights: Optional[List[float]] = None,
        mlp_hidden: Tuple[int, int] = (24, 12),
        feature_names: Optional[List[str]] = None,
        forced_raw_features: Optional[List[str]] = None,
    ):
        self.preproc_kind = preproc_kind
        self.pca_variance_threshold = pca_variance_threshold
        self.max_pca_components = max_pca_components
        self.raw_top_k = raw_top_k
        self.internal_cv = internal_cv
        self.random_state = random_state
        self.base_weights = base_weights
        self.mlp_hidden = mlp_hidden
        self.feature_names = feature_names
        self.forced_raw_features = forced_raw_features

    def _build_representation(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_proc = self.pre_.transform(x)
        x_pca = self.pca_.transform(x_proc)
        if len(self.selected_raw_idx_):
            x_raw = x_proc[:, self.selected_raw_idx_]
            x_fused = np.hstack([x_pca, x_raw])
        else:
            x_fused = x_pca.copy()
        return x_proc, x_pca, x_fused

    def _auto_weights(self, x_pca: np.ndarray, x_fused: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.base_weights is not None:
            arr = np.asarray(self.base_weights, dtype=float)
            arr = np.maximum(arr, 1e-9)
            return arr / arr.sum()

        skf = StratifiedKFold(n_splits=max(2, self.internal_cv), shuffle=True, random_state=self.random_state)
        score_map = {"lda": [], "logreg": [], "mlp": []}

        for tr, te in skf.split(x_fused, y):
            lda = LinearDiscriminantAnalysis(solver="svd")
            lda.fit(x_pca[tr], y[tr])
            score_map["lda"].append(f1_score(y[te], lda.predict(x_pca[te]), average="macro"))

            logreg = _make_multinomial_logreg(class_weight="balanced")
            logreg.fit(x_fused[tr], y[tr])
            score_map["logreg"].append(f1_score(y[te], logreg.predict(x_fused[te]), average="macro"))

            xb, yb = random_oversample(x_fused[tr], y[tr], random_state=self.random_state)
            mlp = _make_regularized_mlp(
                hidden_layer_sizes=self.mlp_hidden,
                random_state=self.random_state,
                max_iter=1600,
                n_iter_no_change=25,
            )
            mlp.fit(xb, yb)
            score_map["mlp"].append(f1_score(y[te], mlp.predict(x_fused[te]), average="macro"))

        scores = np.array(
            [
                np.mean(score_map["lda"]),
                np.mean(score_map["logreg"]),
                np.mean(score_map["mlp"]),
            ],
            dtype=float,
        )
        scores = np.maximum(scores, 1e-9)
        return scores / scores.sum()

    def fit(self, x: np.ndarray, y: np.ndarray):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.sort(np.unique(y))
        self.feature_names_in_ = list(self.feature_names) if self.feature_names is not None else [f"x{i}" for i in range(x.shape[1])]

        self.pre_ = make_preprocessor(self.preproc_kind)
        x_proc = self.pre_.fit_transform(x)

        self.selector_table_ = _feature_selection_consensus(
            x_proc=x_proc,
            y=y,
            feature_names=self.feature_names_in_,
            raw_top_k=self.raw_top_k,
        )
        name_to_idx = {n: i for i, n in enumerate(self.feature_names_in_)}
        if self.forced_raw_features:
            forced = resolve_feature_names([s for s in self.forced_raw_features if str(s).strip()], self.feature_names_in_)
            selected = list(dict.fromkeys(forced))
            self.selector_table_["selected_for_raw_branch"] = 0
            self.selector_table_.loc[self.selector_table_["feature"].isin(selected), "selected_for_raw_branch"] = 1
        else:
            selected = self.selector_table_.loc[self.selector_table_["selected_for_raw_branch"] == 1, "feature"].tolist()
        self.selected_raw_features_ = selected
        self.selected_raw_idx_ = np.array([name_to_idx[f] for f in selected], dtype=int) if selected else np.array([], dtype=int)

        max_allowed = min(self.max_pca_components, x_proc.shape[0], x_proc.shape[1])
        probe = PCA(n_components=max_allowed, random_state=self.random_state)
        probe.fit(x_proc)
        cum = np.cumsum(probe.explained_variance_ratio_)
        n_by_var = int(np.searchsorted(cum, self.pca_variance_threshold) + 1)
        self.n_pca_components_ = max(2, min(max_allowed, n_by_var))
        self.pca_ = PCA(n_components=self.n_pca_components_, random_state=self.random_state)
        x_pca = self.pca_.fit_transform(x_proc)

        if len(self.selected_raw_idx_):
            x_fused = np.hstack([x_pca, x_proc[:, self.selected_raw_idx_]])
        else:
            x_fused = x_pca.copy()

        self.lda_ = LinearDiscriminantAnalysis(solver="svd")
        self.lda_.fit(x_pca, y)

        self.logreg_ = _make_multinomial_logreg(class_weight="balanced")
        self.logreg_.fit(x_fused, y)

        xb, yb = random_oversample(x_fused, y, random_state=self.random_state)
        self.mlp_ = _make_regularized_mlp(
            hidden_layer_sizes=self.mlp_hidden,
            random_state=self.random_state,
            max_iter=1600,
            n_iter_no_change=25,
        )
        self.mlp_.fit(xb, yb)

        self.model_weights_ = self._auto_weights(x_pca, x_fused, y)
        self.model_weight_dict_ = {
            "LDA_PCA": float(self.model_weights_[0]),
            "LogReg_PCAplusRaw": float(self.model_weights_[1]),
            "MLP_PCAplusRaw": float(self.model_weights_[2]),
        }
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        _, x_pca, x_fused = self._build_representation(x)
        p_lda = _align_proba(self.lda_.predict_proba(x_pca), self.lda_.classes_, self.classes_)
        p_log = _align_proba(self.logreg_.predict_proba(x_fused), self.logreg_.classes_, self.classes_)
        p_mlp = _align_proba(self.mlp_.predict_proba(x_fused), self.mlp_.classes_, self.classes_)
        probs = self.model_weights_[0] * p_lda + self.model_weights_[1] * p_log + self.model_weights_[2] * p_mlp
        row_sum = probs.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        return probs / row_sum

    def predict(self, x: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(x)
        return self.classes_[np.argmax(proba, axis=1)]


def _extract_shap_matrix_for_class(shap_values, class_pos: int) -> np.ndarray:
    if isinstance(shap_values, list):
        return np.asarray(shap_values[class_pos])
    arr = np.asarray(shap_values)
    if arr.ndim == 3 and arr.shape[2] > class_pos:
        return arr[:, :, class_pos]
    if arr.ndim == 3 and arr.shape[0] > class_pos:
        return arr[class_pos]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Unsupported SHAP value shape: {arr.shape}")


def _global_rank_fusion(
    feature_names: List[str],
    selector_df: pd.DataFrame,
    perm_df: pd.DataFrame,
    shap_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    out = selector_df[["feature", "avg_rank", "selected_for_raw_branch"]].copy()
    out = out.merge(perm_df[["feature", "perm_importance_mean"]], on="feature", how="left")
    if shap_df is not None:
        out = out.merge(shap_df[["feature", "mean_abs_shap_ckdu"]], on="feature", how="left")
    else:
        out["mean_abs_shap_ckdu"] = np.nan

    out["perm_importance_mean"] = out["perm_importance_mean"].fillna(0.0)
    out["mean_abs_shap_ckdu"] = out["mean_abs_shap_ckdu"].fillna(0.0)

    out["selector_rank_norm"] = 1.0 - (out["avg_rank"] - out["avg_rank"].min()) / max(
        out["avg_rank"].max() - out["avg_rank"].min(), 1e-12
    )
    out["perm_rank_norm"] = out["perm_importance_mean"].rank(ascending=False, method="average")
    out["perm_rank_norm"] = 1.0 - (out["perm_rank_norm"] - out["perm_rank_norm"].min()) / max(
        out["perm_rank_norm"].max() - out["perm_rank_norm"].min(), 1e-12
    )
    out["shap_rank_norm"] = out["mean_abs_shap_ckdu"].rank(ascending=False, method="average")
    out["shap_rank_norm"] = 1.0 - (out["shap_rank_norm"] - out["shap_rank_norm"].min()) / max(
        out["shap_rank_norm"].max() - out["shap_rank_norm"].min(), 1e-12
    )

    out["fusion_contribution_score"] = (
        0.25 * out["selector_rank_norm"]
        + 0.35 * out["perm_rank_norm"]
        + 0.35 * out["shap_rank_norm"]
        + 0.05 * out["selected_for_raw_branch"]
    )
    out["legacy_alias"] = [REVERSE_F_ALIAS_MAP.get(f, "") for f in out["feature"]]
    return out.sort_values("fusion_contribution_score", ascending=False).reset_index(drop=True)


def analysis_fusion(
    ds: Dataset,
    outdir: str,
    n_splits: int = 5,
    preproc_kind: str = "robust",
    pca_variance_threshold: float = 0.95,
    max_pca_components: int = 12,
    raw_top_k: int = 8,
    forced_raw_features: Optional[List[str]] = None,
    shap_background: int = 40,
    shap_samples: int = 20,
    lime_samples: int = 5,
    perm_repeats: int = 10,
):
    outdir = _ensure_outdir(outdir)
    X, y = ds.X, ds.y
    labels_sorted = sorted(np.unique(y))
    label_names = [LABEL_MAP[int(l)] for l in labels_sorted]
    class_array = np.asarray(labels_sorted)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    y_pred_oof = np.zeros_like(y)
    y_proba_oof = np.zeros((len(y), len(labels_sorted)), dtype=float)
    fold_metrics: List[Dict[str, float]] = []
    perm_rows: List[np.ndarray] = []
    selected_counts = {feat: 0 for feat in ds.feature_names}
    model_weight_rows = []

    scorer = make_scorer(f1_score, average="macro")

    for fold, (tr, te) in enumerate(skf.split(X, y), start=1):
        model = PCAGuidedFusionClassifier(
            preproc_kind=preproc_kind,
            pca_variance_threshold=pca_variance_threshold,
            max_pca_components=max_pca_components,
            raw_top_k=raw_top_k,
            forced_raw_features=forced_raw_features,
            internal_cv=3,
            random_state=RANDOM_STATE + fold,
            feature_names=ds.feature_names,
        )
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])
        proba = model.predict_proba(X[te])

        y_pred_oof[te] = pred
        y_proba_oof[te, :] = _align_proba(proba, model.classes_, class_array)

        m = _basic_metrics(y[te], pred)
        m["auc_macro_ovr"] = _safe_multiclass_auc(y[te], y_proba_oof[te, :])
        m["fold"] = fold
        fold_metrics.append(m)

        model_weight_rows.append({"fold": fold, **model.model_weight_dict_})

        for feat in model.selected_raw_features_:
            selected_counts[feat] += 1

        imp = permutation_importance(
            model,
            X[te],
            y[te],
            scoring=scorer,
            n_repeats=perm_repeats,
            random_state=RANDOM_STATE + fold,
            n_jobs=1,
        )
        perm_rows.append(imp.importances_mean.copy())

    metrics = _basic_metrics(y, y_pred_oof)
    metrics["auc_macro_ovr"] = _safe_multiclass_auc(y, y_proba_oof)
    _save_json(metrics, os.path.join(outdir, "fusion_cv_metrics.json"))

    fold_df = pd.DataFrame(fold_metrics)
    fold_df.to_csv(os.path.join(outdir, "fusion_cv_fold_metrics.csv"), index=False)

    weight_df = pd.DataFrame(model_weight_rows)
    weight_df.to_csv(os.path.join(outdir, "fusion_model_weights_by_fold.csv"), index=False)

    _save_confusion(
        y_true=y,
        y_pred=y_pred_oof,
        labels=labels_sorted,
        label_names=label_names,
        outpath=os.path.join(outdir, "fusion_oof_confusion_matrix.png"),
        title=f"PCA-guided fusion model (OOF, {n_splits}-fold CV)",
    )

    with open(os.path.join(outdir, "fusion_oof_classification_report.txt"), "w") as f:
        f.write(classification_report(y, y_pred_oof, target_names=label_names, digits=3))

    perm_mat = np.vstack(perm_rows)
    perm_df = pd.DataFrame(
        {
            "feature": ds.feature_names,
            "perm_importance_mean": perm_mat.mean(axis=0),
            "perm_importance_std": perm_mat.std(axis=0, ddof=1),
            "selected_raw_branch_count": [selected_counts[f] for f in ds.feature_names],
            "legacy_alias": [REVERSE_F_ALIAS_MAP.get(f, "") for f in ds.feature_names],
        }
    ).sort_values("perm_importance_mean", ascending=False)
    perm_df.to_csv(os.path.join(outdir, "fusion_permutation_importance.csv"), index=False)
    _top_bar_plot(
        perm_df,
        value_col="perm_importance_mean",
        title="Fusion model permutation importance (top 15)",
        outpath=os.path.join(outdir, "fusion_permutation_importance_top15.png"),
        top_n=15,
    )

    final_model = PCAGuidedFusionClassifier(
        preproc_kind=preproc_kind,
        pca_variance_threshold=pca_variance_threshold,
        max_pca_components=max_pca_components,
        raw_top_k=raw_top_k,
        forced_raw_features=forced_raw_features,
        internal_cv=3,
        random_state=RANDOM_STATE,
        feature_names=ds.feature_names,
    )
    final_model.fit(X, y)

    selector_df = final_model.selector_table_.copy()
    selector_df["legacy_alias"] = [REVERSE_F_ALIAS_MAP.get(f, "") for f in selector_df["feature"]]
    selector_df.to_csv(os.path.join(outdir, "fusion_raw_feature_selector_consensus.csv"), index=False)
    _save_json(
        {
            "selected_raw_features": final_model.selected_raw_features_,
            "selected_raw_features_alias": [REVERSE_F_ALIAS_MAP.get(f, "") for f in final_model.selected_raw_features_],
            "forced_raw_features": forced_raw_features if forced_raw_features else [],
            "n_pca_components": int(final_model.n_pca_components_),
            "pca_variance_threshold": pca_variance_threshold,
            "model_weights": final_model.model_weight_dict_,
        },
        os.path.join(outdir, "fusion_final_model_summary.json"),
    )

    shap_df: Optional[pd.DataFrame] = None
    try:
        import shap  # type: ignore

        if 1 in final_model.classes_:
            ckdu_class_pos = int(np.where(final_model.classes_ == 1)[0][0])
        else:
            ckdu_class_pos = 0

        rng = np.random.default_rng(RANDOM_STATE)
        idx_ckdu = np.flatnonzero(y == 1) if 1 in np.unique(y) else np.arange(len(y))
        idx_all = np.arange(len(y))

        bg_size = min(shap_background, len(idx_all))
        ex_size = min(shap_samples, len(idx_ckdu) if len(idx_ckdu) else len(idx_all))
        bg_idx = rng.choice(idx_all, size=bg_size, replace=False)
        if len(idx_ckdu):
            explain_idx = rng.choice(idx_ckdu, size=ex_size, replace=False)
        else:
            explain_idx = rng.choice(idx_all, size=ex_size, replace=False)

        background = X[bg_idx]
        explain_x = X[explain_idx]

        explainer = shap.KernelExplainer(final_model.predict_proba, background)
        shap_values = explainer.shap_values(explain_x, nsamples="auto")
        ckdu_shap = _extract_shap_matrix_for_class(shap_values, ckdu_class_pos)

        shap_df = pd.DataFrame(
            {
                "feature": ds.feature_names,
                "mean_abs_shap_ckdu": np.mean(np.abs(ckdu_shap), axis=0),
                "mean_signed_shap_ckdu": np.mean(ckdu_shap, axis=0),
                "legacy_alias": [REVERSE_F_ALIAS_MAP.get(f, "") for f in ds.feature_names],
            }
        ).sort_values("mean_abs_shap_ckdu", ascending=False)
        shap_df.to_csv(os.path.join(outdir, "fusion_shap_global_ckdu.csv"), index=False)
        _top_bar_plot(
            shap_df,
            value_col="mean_abs_shap_ckdu",
            title="Fusion model SHAP importance for CKDu (top 15)",
            outpath=os.path.join(outdir, "fusion_shap_global_ckdu_top15.png"),
            top_n=15,
        )

        per_sample_df = pd.DataFrame(ckdu_shap, columns=ds.feature_names)
        per_sample_df.insert(0, "sample_index", explain_idx)
        per_sample_df.to_csv(os.path.join(outdir, "fusion_shap_ckdu_per_sample.csv"), index=False)
    except Exception as e:
        with open(os.path.join(outdir, "fusion_shap_skipped.txt"), "w") as f:
            f.write(f"SHAP could not be generated: {e}\n")

    try:
        from lime.lime_tabular import LimeTabularExplainer  # type: ignore

        class_names = [LABEL_MAP[int(c)] for c in final_model.classes_]
        lime_outdir = _ensure_outdir(os.path.join(outdir, "lime_explanations"))
        explainer = LimeTabularExplainer(
            training_data=X,
            feature_names=ds.feature_names,
            class_names=class_names,
            mode="classification",
            discretize_continuous=True,
            random_state=RANDOM_STATE,
        )

        if 1 in np.unique(y):
            idx_pool = np.flatnonzero(y == 1)
        else:
            idx_pool = np.arange(len(y))
        rng = np.random.default_rng(RANDOM_STATE)
        chosen = rng.choice(idx_pool, size=min(lime_samples, len(idx_pool)), replace=False)
        ckdu_pos = int(np.where(final_model.classes_ == 1)[0][0]) if 1 in final_model.classes_ else 0
        lime_rows = []
        for idx in chosen:
            exp = explainer.explain_instance(
                data_row=X[idx],
                predict_fn=final_model.predict_proba,
                labels=[ckdu_pos],
                num_features=min(12, X.shape[1]),
            )
            html_path = os.path.join(lime_outdir, f"lime_sample_{idx}_ckdu.html")
            txt_path = os.path.join(lime_outdir, f"lime_sample_{idx}_ckdu.txt")
            exp.save_to_file(html_path)
            items = exp.as_list(label=ckdu_pos)
            with open(txt_path, "w") as f:
                for cond, weight in items:
                    f.write(f"{cond}\t{weight}\n")
            for cond, weight in items:
                lime_rows.append({"sample_index": int(idx), "rule": cond, "weight": float(weight)})
        pd.DataFrame(lime_rows).to_csv(os.path.join(outdir, "fusion_lime_summary_ckdu.csv"), index=False)
    except Exception as e:
        with open(os.path.join(outdir, "fusion_lime_skipped.txt"), "w") as f:
            f.write(
                "LIME explanations were skipped. Install the package with: pip install lime\n"
                f"Underlying error: {e}\n"
            )

    fused_rank_df = _global_rank_fusion(
        feature_names=ds.feature_names,
        selector_df=selector_df,
        perm_df=perm_df,
        shap_df=shap_df,
    )
    fused_rank_df.to_csv(os.path.join(outdir, "fusion_final_element_ranking.csv"), index=False)
    _top_bar_plot(
        fused_rank_df,
        value_col="fusion_contribution_score",
        title="Final fusion-based element ranking (top 15)",
        outpath=os.path.join(outdir, "fusion_final_element_ranking_top15.png"),
        top_n=15,
    )

    print("\n[Fusion] CV metrics:")
    print(json.dumps(metrics, indent=2))
    print("\n[Fusion] Selected raw-feature branch:")
    print(final_model.selected_raw_features_)
    print("\n[Fusion] Final top-ranked elements:")
    print(fused_rank_df.head(12).to_string(index=False))
    print(f"\n[Fusion] Saved outputs to: {outdir}")
