from __future__ import annotations

import os

from ..shared import (
    REVERSE_F_ALIAS_MAP,
    Dataset,
    _derive_frequency_consensus_features_from_outputs,
    _ensure_outdir,
    _save_json,
    resolve_feature_names,
)
from .bayes import analysis_bayes
from .fisher import analysis_fisher
from .fusion import analysis_fusion
from .gmm import analysis_gmm
from .nn import analysis_nn
from .pca import analysis_pca
from .regression import analysis_regression
from .significance import analysis_significance
from .skew_kurt import analysis_skew_kurt
from .validate_top8 import analysis_validate_top_features


def analysis_all(ds: Dataset, outdir: str, args):
    root = _ensure_outdir(outdir)
    analysis_pca(ds, os.path.join(root, "pca"))
    analysis_skew_kurt(ds, os.path.join(root, "skew_kurt"))
    analysis_significance(ds, os.path.join(root, "significance"))
    analysis_bayes(ds, os.path.join(root, "bayes"), n_splits=args.cv_splits)
    analysis_fisher(ds, os.path.join(root, "fisher"), n_splits=args.cv_splits)
    analysis_regression(ds, os.path.join(root, "regression"))
    analysis_gmm(ds, os.path.join(root, "gmm"))
    analysis_nn(ds, os.path.join(root, "nn"))

    consensus_top, consensus_summary_df, consensus_hits_df = _derive_frequency_consensus_features_from_outputs(
        root_outdir=root,
        feature_names=ds.feature_names,
        top_k=args.consensus_top_k,
        source_top_n=args.consensus_source_top_n,
    )
    if consensus_top:
        selected_top_features = consensus_top
        selection_mode = "frequency_consensus_from_previous_analyses"
    else:
        selected_top_features = [s.strip() for s in args.top_features.split(",") if s.strip()]
        selected_top_features = resolve_feature_names(selected_top_features, ds.feature_names)
        selection_mode = "fallback_manual_top_features"

    consensus_summary_df.to_csv(os.path.join(root, "consensus_feature_frequency_summary.csv"), index=False)
    if not consensus_hits_df.empty:
        consensus_hits_df.to_csv(os.path.join(root, "consensus_feature_frequency_hits.csv"), index=False)
    _save_json(
        {
            "selection_mode": selection_mode,
            "consensus_top_features": selected_top_features,
            "consensus_top_features_alias": [REVERSE_F_ALIAS_MAP.get(f, "") for f in selected_top_features],
            "consensus_top_k": int(args.consensus_top_k),
            "consensus_source_top_n": int(args.consensus_source_top_n),
        },
        os.path.join(root, "consensus_feature_selection.json"),
    )
    print("\n[All] Consensus top features from previous analyses:")
    print(selected_top_features)

    analysis_validate_top_features(
        ds,
        outdir=os.path.join(root, "validate_top8"),
        top_features=selected_top_features,
        preproc_kind=args.preproc,
        n_splits=args.cv_splits,
        perm_repeats=args.perm_repeats,
    )
    analysis_fusion(
        ds,
        outdir=os.path.join(root, "fusion"),
        n_splits=args.cv_splits,
        preproc_kind=args.preproc,
        pca_variance_threshold=args.fusion_pca_var,
        max_pca_components=args.fusion_max_pca,
        raw_top_k=max(args.fusion_raw_top_k, len(selected_top_features)),
        forced_raw_features=selected_top_features,
        shap_background=args.shap_background,
        shap_samples=args.shap_samples,
        lime_samples=args.lime_samples,
        perm_repeats=args.perm_repeats,
    )
