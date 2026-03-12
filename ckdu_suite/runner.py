from __future__ import annotations

import os
from typing import Callable

from .analyses import (
    analysis_all,
    analysis_bayes,
    analysis_fisher,
    analysis_fusion,
    analysis_gmm,
    analysis_nn,
    analysis_pca,
    analysis_regression,
    analysis_significance,
    analysis_skew_kurt,
    analysis_validate_top_features,
)
from .config import CKDUConfig
from .shared import load_ckdu_processed


class CKDUAnalysisRunner:
    """Orchestrates CKDu analyses from modular analysis scripts."""

    def __init__(self, config: CKDUConfig):
        self.config = config
        self.dataset = load_ckdu_processed(config.csv)

    def run(self) -> None:
        if self.config.analysis == "all":
            analysis_all(self.dataset, outdir=self.config.outdir, args=self.config.to_namespace())
            return

        outdir = os.path.join(self.config.outdir, self.config.analysis)
        run_map: dict[str, Callable[[str], None]] = {
            "pca": self._run_pca,
            "bayes": self._run_bayes,
            "fisher": self._run_fisher,
            "skew_kurt": self._run_skew_kurt,
            "significance": self._run_significance,
            "regression": self._run_regression,
            "gmm": self._run_gmm,
            "nn": self._run_nn,
            "validate_top8": self._run_validate_top8,
            "fusion": self._run_fusion,
        }
        try:
            run_map[self.config.analysis](outdir)
        except KeyError as exc:
            raise ValueError(f"Unsupported analysis: {self.config.analysis}") from exc

    def _run_pca(self, outdir: str) -> None:
        analysis_pca(self.dataset, outdir=outdir)

    def _run_bayes(self, outdir: str) -> None:
        analysis_bayes(self.dataset, outdir=outdir, n_splits=self.config.cv_splits)

    def _run_fisher(self, outdir: str) -> None:
        analysis_fisher(self.dataset, outdir=outdir, n_splits=self.config.cv_splits)

    def _run_skew_kurt(self, outdir: str) -> None:
        analysis_skew_kurt(self.dataset, outdir=outdir)

    def _run_significance(self, outdir: str) -> None:
        analysis_significance(self.dataset, outdir=outdir)

    def _run_regression(self, outdir: str) -> None:
        analysis_regression(self.dataset, outdir=outdir)

    def _run_gmm(self, outdir: str) -> None:
        analysis_gmm(self.dataset, outdir=outdir)

    def _run_nn(self, outdir: str) -> None:
        analysis_nn(self.dataset, outdir=outdir)

    def _run_validate_top8(self, outdir: str) -> None:
        analysis_validate_top_features(
            self.dataset,
            outdir=outdir,
            top_features=self.config.parsed_top_features(),
            preproc_kind=self.config.preproc,
            n_splits=self.config.cv_splits,
            perm_repeats=self.config.perm_repeats,
        )

    def _run_fusion(self, outdir: str) -> None:
        analysis_fusion(
            self.dataset,
            outdir=outdir,
            n_splits=self.config.cv_splits,
            preproc_kind=self.config.preproc,
            pca_variance_threshold=self.config.fusion_pca_var,
            max_pca_components=self.config.fusion_max_pca,
            raw_top_k=self.config.fusion_raw_top_k,
            shap_background=self.config.shap_background,
            shap_samples=self.config.shap_samples,
            lime_samples=self.config.lime_samples,
            perm_repeats=self.config.perm_repeats,
        )
