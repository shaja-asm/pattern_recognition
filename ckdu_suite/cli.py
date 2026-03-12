from __future__ import annotations

import argparse
from typing import Optional

from .config import ANALYSIS_CHOICES, CKDUConfig
from .runner import CKDUAnalysisRunner


def build_parser(include_analysis: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="CKDu_processed.csv", help="Path to CKDu_processed.csv")
    if include_analysis:
        parser.add_argument(
            "--analysis",
            required=True,
            choices=ANALYSIS_CHOICES,
            help="Which analysis to run",
        )
    parser.add_argument("--outdir", default="ckdu_results", help="Output directory")

    parser.add_argument(
        "--top_features",
        default="Ag,Ca,Na,K,Al,V,Sr,In",
        help=(
            "Comma-separated consensus features to validate. "
            "You may use real names (e.g. Ag,Ca,Na) or legacy aliases (e.g. F21,F4,F1)."
        ),
    )
    parser.add_argument(
        "--preproc",
        default="robust",
        choices=["robust", "standard"],
        help="Preprocessing pipeline for validate_top8/fusion",
    )
    parser.add_argument("--cv_splits", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--perm_repeats", type=int, default=10, help="Permutation repeats")
    parser.add_argument(
        "--consensus_top_k",
        type=int,
        default=8,
        help="When running all analyses: number of frequency-consensus features for validate_top8 and fusion",
    )
    parser.add_argument(
        "--consensus_source_top_n",
        type=int,
        default=8,
        help="When running all analyses: top-N features taken from each prior analysis output for consensus counting",
    )
    parser.add_argument("--fusion_raw_top_k", type=int, default=8, help="Number of raw features in the fusion branch")
    parser.add_argument("--fusion_max_pca", type=int, default=12, help="Maximum PCA components for the fusion branch")
    parser.add_argument("--fusion_pca_var", type=float, default=0.95, help="Cumulative variance target for fusion PCA")
    parser.add_argument("--shap_background", type=int, default=40, help="Background samples for SHAP")
    parser.add_argument("--shap_samples", type=int, default=20, help="Number of samples to explain with SHAP")
    parser.add_argument("--lime_samples", type=int, default=5, help="Number of local CKDu samples for LIME")

    return parser


def run_cli(default_analysis: Optional[str] = None) -> None:
    include_analysis = default_analysis is None
    parser = build_parser(include_analysis=include_analysis)
    args = parser.parse_args()

    if default_analysis is not None:
        if default_analysis not in ANALYSIS_CHOICES:
            raise ValueError(f"Unsupported analysis script target: {default_analysis}")
        args.analysis = default_analysis

    config = CKDUConfig.from_namespace(args)
    CKDUAnalysisRunner(config).run()
