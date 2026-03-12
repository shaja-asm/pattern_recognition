from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass

ANALYSIS_CHOICES = (
    "pca",
    "bayes",
    "fisher",
    "skew_kurt",
    "significance",
    "regression",
    "gmm",
    "nn",
    "validate_top8",
    "fusion",
    "all",
)


@dataclass(frozen=True)
class CKDUConfig:
    csv: str
    analysis: str
    outdir: str
    top_features: str
    preproc: str
    cv_splits: int
    perm_repeats: int
    consensus_top_k: int
    consensus_source_top_n: int
    fusion_raw_top_k: int
    fusion_max_pca: int
    fusion_pca_var: float
    shap_background: int
    shap_samples: int
    lime_samples: int

    @classmethod
    def from_namespace(cls, args: Namespace) -> "CKDUConfig":
        return cls(
            csv=args.csv,
            analysis=args.analysis,
            outdir=args.outdir,
            top_features=args.top_features,
            preproc=args.preproc,
            cv_splits=args.cv_splits,
            perm_repeats=args.perm_repeats,
            consensus_top_k=args.consensus_top_k,
            consensus_source_top_n=args.consensus_source_top_n,
            fusion_raw_top_k=args.fusion_raw_top_k,
            fusion_max_pca=args.fusion_max_pca,
            fusion_pca_var=args.fusion_pca_var,
            shap_background=args.shap_background,
            shap_samples=args.shap_samples,
            lime_samples=args.lime_samples,
        )

    def to_namespace(self) -> Namespace:
        return Namespace(
            csv=self.csv,
            analysis=self.analysis,
            outdir=self.outdir,
            top_features=self.top_features,
            preproc=self.preproc,
            cv_splits=self.cv_splits,
            perm_repeats=self.perm_repeats,
            consensus_top_k=self.consensus_top_k,
            consensus_source_top_n=self.consensus_source_top_n,
            fusion_raw_top_k=self.fusion_raw_top_k,
            fusion_max_pca=self.fusion_max_pca,
            fusion_pca_var=self.fusion_pca_var,
            shap_background=self.shap_background,
            shap_samples=self.shap_samples,
            lime_samples=self.lime_samples,
        )

    def parsed_top_features(self) -> list[str]:
        return [item.strip() for item in self.top_features.split(",") if item.strip()]
