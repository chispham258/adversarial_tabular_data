# Adversarial Attack Detection for Tabular Data

This repository implements an end-to-end experimental workflow for detecting
adversarial attacks against tabular classifiers. It trains monitored
classifiers on clean tabular datasets, generates or reuses adversarial examples,
summarizes clean and attacked batches with diagnostic neighborhood features,
and trains secondary classifiers that predict whether a batch is adversarial and
which attack produced it.

The project is oriented toward scientific experimentation rather than a single
production model. Most intermediate artifacts are saved as CSV, JSON, Joblib, or
PyTorch files so that every stage can be inspected independently.

## Research Objective

Given a monitored classifier \(f\), a clean tabular instance \(x \in
\mathbb{R}^d\), and an adversarial perturbation \(\delta\), an adversarial
instance is represented as:

```text
x' = x + delta, subject to ||delta||_p <= epsilon and f(x') != y
```

The central question is whether a detector can identify adversarial manipulation
from diagnostic behavior rather than from the original raw feature space. The
detector therefore operates on batch-level meta-features derived from local
neighborhoods, predictive uncertainty, class-consistency statistics, diversity
statistics, and monitored-model balanced accuracy.

The implemented tasks are:

- Binary detection: classify a batch as clean (`org`) or attacked.
- Attack-family classification: classify a batch as clean, gradient-based, or
  black-box/query-based.
- Exact attack-type classification: predict the exact attack label, such as
  `bim`, `fgm`, `pgd`, `hsj`, or `zoo`.

## Roles

Project responsibilities:

- Pham Ngoc Tho: attack generation and testing.
- Doan Quoc Kien: attack classification and demo.

## Datasets

The workflow currently supports three binary tabular datasets:

| Dataset | Target column | Source handled by code | Test samples in saved run |
| --- | --- | --- | ---: |
| Banknote authentication | `Class` | Hugging Face CSV mirror | 275 |
| Pima Indians diabetes | `Outcome` | public CSV mirror | 154 |
| Wilt | `target` | original adversarial-tabular data mirror | 968 |

Raw files are stored under `data/raw/`. Processed train/test splits and scalers
are stored under `data/processed/<dataset>/`.

## Monitored Models and Attacks

The upstream attack stage trains monitored classifiers and attacks them with
methods from the Adversarial Robustness Toolbox (ART).

Monitored model labels:

- `lin`: logistic regression.
- `svm`: support-vector classifier with probability outputs.
- `xgb`: upstream gradient-boosting classifier saved with an XGB-style label.
- `nn`: small PyTorch feed-forward network.

Attack labels:

- `fgm`: Fast Gradient Method.
- `bim`: Basic Iterative Method.
- `pgd`: Projected Gradient Descent.
- `hsj`: HopSkipJump.
- `zoo`: Zeroth Order Optimization.
- `lpf`: LowProFool, used in upstream evaluation where compatible.

Saved adversarial batches are located at
`data/adversarial_upstream/<dataset>/<model>_<attack>*.csv`.

## Method Summary

The detector does not directly classify individual raw rows. Instead, each clean
or adversarial batch is converted to diagnostic attributes:

1. A clean reference set is built for each dataset and monitored model.
2. A nearest-neighbor diagnostic context is fitted in the processed feature
   space.
3. For each row in a clean or adversarial batch, the code computes local
   statistics such as neighborhood size, uncertainty, target/prediction
   consistency, and class diversity.
4. Row-level diagnostics are aggregated into fixed-width batch vectors using
   mean, minimum, quartiles, maximum, and range.
5. Random Forest, XGBoost, and optimized tree ensembles are evaluated on
   binary, attack-family, and exact attack-type targets.

Balanced accuracy is used throughout because clean and attacked labels can be
imbalanced:

```text
BACC = (1 / C) * sum_c TP_c / (TP_c + FN_c)
```

Generalization is evaluated with:

- 10-fold stratified cross-validation.
- One-dataset-out evaluation.
- One-monitored-model-out evaluation.
- One-attack-out evaluation for binary detection.

## Repository Layout

```text
data/
  raw/                         Raw tabular datasets.
  processed/                   Train/test splits and scalers.
  adversarial_upstream/        Generated adversarial batches and meta-tables.
guidance/                      Original notebooks/scripts used as references.
models/
  <dataset>/                   Saved upstream monitored models.
  binary_attack/               Saved binary detector artifacts.
  attack_type/                 Saved baseline exact attack-type classifiers.
  attack_type_optimized/       Saved optimized exact and family classifiers.
reports/
  figures/                     Generated report figures.
  adversarial_attack_detection_report.md
  adversarial_attack_detection_report.pdf
results/                       Metrics, predictions, feature importance, and
                               aggregated diagnostic tables.
src/
  preprocess.py                Dataset download, split, encoding, scaling.
  train_baselines.py           Optional clean-data baseline classifiers.
  upstream_attacks.py          Upstream monitored-model training and attacks.
  rf_xgb_detection.py          Diagnostic generation and RF/XGB detectors.
  generate_report_assets.py    Figure and Markdown report generation.
  demo_app.py                  Streamlit demo for attack generation/detection.
```

## Important Artifacts

| Path | Description |
| --- | --- |
| `results/upstream_attack_results.csv` | Clean/attacked accuracy, balanced accuracy, ASR, and perturbation norms. |
| `results/attacks_diagnoses.csv` | Row-level diagnostic features for non-neural monitored models. |
| `results/attacks_diagnoses_nn.csv` | Row-level diagnostic features including neural monitored models. |
| `results/attr_attacks_binary_agr_nn_bacc.csv` | Aggregated binary-detector training table. |
| `results/attr_attacks_type_agr_nn.csv` | Aggregated exact attack-type training table. |
| `results/attr_attacks_type_optimized_windowed.csv` | Windowed diagnostic table for optimized classifiers. |
| `results/detection_bacc_with_bacc.csv` | Binary detector evaluation metrics. |
| `results/isolation_bacc_nn.csv` | Baseline exact attack-type evaluation metrics. |
| `results/attack_type_optimized_metrics.csv` | Optimized exact attack-type evaluation metrics. |
| `results/attack_type_optimized_grouped_metrics.csv` | Attack-family evaluation metrics. |
| `reports/adversarial_attack_detection_report.md` | Full generated scientific report. |

## Installation

Create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The dependency list includes PyTorch, scikit-learn, XGBoost, LightGBM,
CatBoost, Optuna, Streamlit, Plotly, and ART. A GPU is not required; the
included neural networks are small enough to run on CPU.

## Reproducing the Pipeline

### 1. Preprocess a dataset

```powershell
python -m src.preprocess --dataset banknote
python -m src.preprocess --dataset diabetes
python -m src.preprocess --dataset wilt
```

### 2. Generate upstream adversarial data

A full run over all datasets, monitored models, and attacks can be expensive,
especially for query-based attacks:

```powershell
python -m src.upstream_attacks --datasets banknote diabetes wilt --models nn lin svm xgb
```

For a quick smoke run:

```powershell
python -m src.upstream_attacks --datasets banknote --models nn --nn-attacks fgm --eps 0.1 --steps 10 --nn-epochs 20
```

This stage writes:

- adversarial CSVs to `data/adversarial_upstream/`;
- monitored-model checkpoints to `models/<dataset>/`;
- attack-effectiveness metrics to `results/upstream_attack_results.csv`.

### 3. Generate diagnostic tables and train detectors

Use the existing saved adversarial batches and diagnostics:

```powershell
python -m src.rf_xgb_detection --task both --no-recompute-diagnostics
```

For a faster detector smoke run without hyperparameter search:

```powershell
python -m src.rf_xgb_detection --task both --no-recompute-diagnostics --no-tune
```

Available tasks are `diagnostics`, `binary`, `type`, and `both`.

### 4. Regenerate report figures and Markdown

```powershell
python -m src.generate_report_assets
```

This reads the saved result CSVs and writes updated figures under
`reports/figures/` plus `reports/adversarial_attack_detection_report.md`.

### 5. Launch the Streamlit demo

```powershell
streamlit run src/demo_app.py
```

The demo can load precomputed attack batches, compute diagnostic vectors, and
run the saved binary, exact-type, and attack-family detectors.

## Current Experimental Findings

The saved report and metrics show a consistent pattern:

- Binary attack detection is the most stable task because it only separates
  clean from suspicious batch behavior.
- Exact attack-type classification is harder because several attacks produce
  overlapping diagnostic signatures.
- Attack-family classification provides an intermediate target that is easier
  than exact attack identification while still being more informative than
  binary detection.
- Strong diagnostic predictors include monitored-model balanced accuracy,
  neighborhood-size summaries, uncertainty, prediction/target consistency, and
  neighborhood diversity.

See `reports/adversarial_attack_detection_report.md` for the full narrative,
tables, confusion matrices, and feature-importance figures.

## Notes and Caveats

- The upstream `xgb` label in `upstream_attacks.py` is implemented with
  scikit-learn `GradientBoostingClassifier`; downstream detector experiments
  use the external XGBoost package when available.
- Query-based attacks such as ZOO and HSJ are computationally expensive. The
  default script limits black-box attacks to a sample subset for tractability.
- Neural-network epsilon `0.00` attack rows are sanity checks rather than
  meaningful adversarial perturbations.
- The repository contains generated models and result files. Re-running the
  full pipeline may overwrite metrics and artifacts in `results/` and `models/`.
