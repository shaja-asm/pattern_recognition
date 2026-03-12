# CKDu Analysis Suite

This repository contains a modular CKDu multiclass analysis pipeline.

## Current Script Structure

- `main.py`: unified CLI entrypoint (`--analysis` required)
- `ckdu_suite/cli.py`: shared argument parser and CLI wiring
- `ckdu_suite/config.py`: config dataclass and analysis choices
- `ckdu_suite/runner.py`: orchestration and dispatch
- `ckdu_suite/analyses/*.py`: analysis implementations

## Setup

Install dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

## Run From Repository Root

Unified entrypoint:

```bash
python main.py --analysis all --csv CKDu_processed.csv --outdir ckdu_results
python main.py --analysis pca --csv CKDu_processed.csv --outdir ckdu_results
```

Valid `--analysis` values:

- `pca`
- `bayes`
- `fisher`
- `skew_kurt`
- `significance`
- `regression`
- `gmm`
- `nn`
- `validate_top8`
- `fusion`
- `all`

Run each analysis individually:

```bash
python main.py --analysis pca --csv CKDu_processed.csv --outdir ckdu_results
python main.py --analysis bayes --csv CKDu_processed.csv --outdir ckdu_results
python main.py --analysis fisher --csv CKDu_processed.csv --outdir ckdu_results
python main.py --analysis skew_kurt --csv CKDu_processed.csv --outdir ckdu_results
python main.py --analysis significance --csv CKDu_processed.csv --outdir ckdu_results
python main.py --analysis regression --csv CKDu_processed.csv --outdir ckdu_results
python main.py --analysis gmm --csv CKDu_processed.csv --outdir ckdu_results
python main.py --analysis nn --csv CKDu_processed.csv --outdir ckdu_results
python main.py --analysis validate_top8 --csv CKDu_processed.csv --outdir ckdu_results
python main.py --analysis fusion --csv CKDu_processed.csv --outdir ckdu_results
```

## Common CLI Options

All options are supported through `main.py`:

- `--csv CKDu_processed.csv`
- `--outdir ckdu_results`
- `--cv_splits 5`
- `--perm_repeats 10`
- `--preproc robust` (`robust` or `standard`)
- `--top_features Ag,Ca,Na,K,Al,V,Sr,In`
- `--consensus_top_k 8`
- `--consensus_source_top_n 8`
- `--fusion_raw_top_k 8`
- `--fusion_max_pca 12`
- `--fusion_pca_var 0.95`
- `--shap_background 40`
- `--shap_samples 20`
- `--lime_samples 5`

## Data Expectations

- Input CSV has no header row.
- Column 0 is class label (`1..5`).
- Features must contain either exactly 30 metal columns, or at least 32 columns where the first 2 feature columns are non-metal values (e.g. Age/S.cr) followed by 30 metals.

## Output Layout

- Single analysis runs write to `ckdu_results/<analysis>/`.
- `--analysis all` writes all analysis subfolders under `ckdu_results/` and also generates consensus files at that root.
