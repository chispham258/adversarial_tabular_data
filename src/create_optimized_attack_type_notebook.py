import json
from pathlib import Path

import pandas as pd


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip().splitlines(True),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


INTRO = """
# Optimized Attack-Type Classification with Optuna

Standalone Kaggle-friendly notebook for optimizing the attack-type classifier on the aggregated diagnostic table.

The goal is honest generalization, not just a high cross-validation number. The primary model-selection score is leave-one-dataset-out balanced accuracy with Cohen kappa as a tie-breaker.
"""


IMPORTS = """
from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

try:
    import optuna
except ImportError as exc:
    raise ImportError("Install optuna before running this notebook: pip install optuna") from exc

try:
    import xgboost as xgb
except ImportError as exc:
    raise ImportError("Install xgboost before running this notebook: pip install xgboost") from exc

try:
    from catboost import CatBoostClassifier
except ImportError as exc:
    raise ImportError("Install catboost before running this notebook: pip install catboost") from exc

try:
    from imblearn.over_sampling import RandomOverSampler, SMOTE
    from imblearn.combine import SMOTEENN
except ImportError as exc:
    raise ImportError("Install imbalanced-learn before running this notebook: pip install imbalanced-learn") from exc

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)
"""


SETUP = """
KAGGLE_INPUT_DIR = Path("/kaggle/input/datasets/hongthnguyn/attack-data/data")

RANDOM_STATE = 123
N_TRIALS_PER_FAMILY = 40
MAX_CV_SPLITS = 10
PRIMARY_SCENARIO = "one-data-set-out"
SELECTION_MODE = "combined"
DATASET_SCORE_WEIGHT = 0.65
MODEL_SCORE_WEIGHT = 0.35
USE_CONSTRAINED_PREDICTIONS = True
WINDOWED_AGGREGATION_ENABLED = True
WINDOWS_PER_GROUP = 5
MIN_WINDOW_ROWS = 20
TARGET_FILE = "attr_attacks_type_agr_nn.csv"
EMBEDDED_ATTACK_TYPE_TABLE_JSON = __EMBEDDED_ATTACK_TYPE_TABLE_JSON__
SUPPORTED_ATTACKS = ["bim", "fgm", "hsj", "org", "pgd", "zoo"]
GRADIENT_ATTACKS = {"bim", "fgm", "pgd"}
BLACKBOX_ATTACKS = {"hsj", "zoo"}
ALLOWED_ATTACKS_BY_MONITORED_MODEL = {
    "nn": ["org", "bim", "fgm", "pgd"],
    "lin": ["org", "hsj", "zoo"],
    "svm": ["org", "hsj", "zoo"],
    "xgb": ["org", "hsj", "zoo"],
}
DIAGNOSTIC_DROP_COLUMNS = [
    "approx",
    "target",
    "pred",
    "error",
    "name",
    "overall_mean_target",
    "scores",
    "mean_target_in_neighborhood",
    "mean_approx_in_neighborhood",
    "neighborhood_size_div_model_avg",
    "neighborhood_size_pct",
    "r_centered_entropy",
    "entropy",
    "logk_r_centered_entropy",
]


def candidate_roots() -> List[Path]:
    roots = [
        KAGGLE_INPUT_DIR,
        KAGGLE_INPUT_DIR.parent,
        Path.cwd(),
        *Path.cwd().parents,
    ]
    kaggle_working = Path("/kaggle/working")
    if kaggle_working.exists():
        roots.insert(0, kaggle_working)
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        roots.extend(p for p in kaggle_input.glob("**") if p.is_dir())
    return roots


def find_results_table() -> Optional[Path]:
    checked = []
    for root in candidate_roots():
        for path in [
            root / "adversarial_upstream" / TARGET_FILE,
            root / "results" / TARGET_FILE,
            root / "data" / "adversarial_upstream" / TARGET_FILE,
            root / "data" / "results" / TARGET_FILE,
            root / TARGET_FILE,
        ]:
            checked.append(path)
            if path.exists():
                return path.resolve()
    print(
        "Could not locate attr_attacks_type_agr_nn.csv. "
        "Using embedded fallback table from the repository snapshot. Checked examples: "
        + ", ".join(str(p) for p in checked[:12])
    )
    return None


def find_optional_file(file_name: str) -> Optional[Path]:
    for root in candidate_roots():
        for path in [
            root / "adversarial_upstream" / file_name,
            root / "results" / file_name,
            root / "data" / "adversarial_upstream" / file_name,
            root / "data" / "results" / file_name,
            root / file_name,
        ]:
            if path.exists():
                return path.resolve()
    return None


TABLE_PATH = find_results_table()
REPO_ROOT = TABLE_PATH.parents[1] if TABLE_PATH is not None and TABLE_PATH.parent.name == "results" else Path.cwd()
RESULTS_DIR = Path("/kaggle/working/results") if Path("/kaggle/working").exists() else REPO_ROOT / "results"
MODEL_DIR = Path("/kaggle/working/models/attack_type_optimized") if Path("/kaggle/working").exists() else REPO_ROOT / "models" / "attack_type_optimized"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TABLE_SOURCE = "file"
if TABLE_PATH is None:
    TABLE_SOURCE = "embedded_fallback"
    TABLE_PATH = RESULTS_DIR / TARGET_FILE
    fallback_df = pd.DataFrame(json.loads(EMBEDDED_ATTACK_TYPE_TABLE_JSON))
    fallback_df.to_csv(TABLE_PATH, index=False)

print(f"TABLE_PATH={TABLE_PATH}")
print(f"TABLE_SOURCE={TABLE_SOURCE}")
print(f"KAGGLE_INPUT_DIR={KAGGLE_INPUT_DIR}")
print(f"RESULTS_DIR={RESULTS_DIR}")
print(f"MODEL_DIR={MODEL_DIR}")
print(f"N_TRIALS_PER_FAMILY={N_TRIALS_PER_FAMILY}")
print(f"SELECTION_MODE={SELECTION_MODE}")
print(f"USE_CONSTRAINED_PREDICTIONS={USE_CONSTRAINED_PREDICTIONS}")
print(f"WINDOWED_AGGREGATION_ENABLED={WINDOWED_AGGREGATION_ENABLED}")
"""


LOAD_DATA = """
def aggregate_numeric(frame: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    numeric_cols = [
        col for col in frame.columns
        if col not in group_cols
        and col not in DIAGNOSTIC_DROP_COLUMNS
        and pd.api.types.is_numeric_dtype(frame[col])
    ]
    rows = []
    for keys, group in frame.groupby(list(group_cols), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        for col in numeric_cols:
            values = pd.to_numeric(group[col], errors="coerce").dropna()
            if values.empty:
                continue
            row[f"{col}_mean"] = float(values.mean())
            row[f"{col}_q0"] = float(values.min())
            row[f"{col}_q25"] = float(values.quantile(0.25))
            row[f"{col}_q50"] = float(values.quantile(0.50))
            row[f"{col}_q75"] = float(values.quantile(0.75))
            row[f"{col}_q1"] = float(values.max())
            row[f"{col}_minmax"] = float(values.max() - values.min())
        rows.append(row)
    return pd.DataFrame(rows)


def add_windowed_aggregates(frame: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    full = aggregate_numeric(frame, group_cols)
    full["window_id"] = "full"
    full["window_source"] = "full_group"
    full["window_rows"] = np.nan
    if not WINDOWED_AGGREGATION_ENABLED:
        return full

    rng = np.random.default_rng(RANDOM_STATE)
    window_frames = [full]
    for keys, group in frame.groupby(list(group_cols), dropna=False):
        if len(group) < MIN_WINDOW_ROWS * 2:
            continue
        shuffled = group.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
        chunks = [chunk for chunk in np.array_split(shuffled, WINDOWS_PER_GROUP) if len(chunk) >= MIN_WINDOW_ROWS]
        for idx, chunk in enumerate(chunks):
            agg = aggregate_numeric(chunk, group_cols)
            agg["window_id"] = f"window_{idx}"
            agg["window_source"] = "diagnostic_window"
            agg["window_rows"] = len(chunk)
            window_frames.append(agg)
    return pd.concat(window_frames, ignore_index=True)


def build_training_table() -> Tuple[pd.DataFrame, str]:
    base_diag_path = find_optional_file("attacks_diagnoses.csv")
    nn_diag_path = find_optional_file("attacks_diagnoses_nn.csv")
    if base_diag_path is not None and nn_diag_path is not None:
        print(f"Using raw diagnostics for windowed aggregates: {base_diag_path}, {nn_diag_path}")
        base = pd.read_csv(base_diag_path)
        base = base[(base["dataset"] != "mfeat-morphological") & (base["attack"] != "lpf")].copy()
        base = base[base["attack"].isin(SUPPORTED_ATTACKS)].copy()
        nn_df = pd.read_csv(nn_diag_path)
        nn_df = nn_df[nn_df["attack"].isin(SUPPORTED_ATTACKS)].copy()
        raw = pd.concat([base, nn_df], ignore_index=True)
        table = add_windowed_aggregates(raw, ["dataset", "model", "attack", "bacc_test", "n_test", "n_classes"])
        table.to_csv(RESULTS_DIR / "attr_attacks_type_optimized_windowed.csv", index=False)
        return table, "windowed_raw_diagnostics"

    print("Raw diagnostic CSVs not found. Falling back to the pre-aggregated 39-row table.")
    table = pd.read_csv(TABLE_PATH)
    table["window_id"] = "full"
    table["window_source"] = "preaggregated"
    table["window_rows"] = np.nan
    return table, "preaggregated"


df, TRAINING_TABLE_SOURCE = build_training_table()
df = df[df["attack"].isin(SUPPORTED_ATTACKS)].reset_index(drop=True)

DROP_COLUMNS = ["dataset", "model", "attack", "window_id", "window_source", "window_rows"]
FEATURE_COLUMNS = [
    col for col in df.columns
    if col not in DROP_COLUMNS and pd.api.types.is_numeric_dtype(df[col])
]

encoder = LabelEncoder()
y_all = encoder.fit_transform(df["attack"])
class_labels = list(encoder.classes_)

print("Shape:", df.shape)
print("Training table source:", TRAINING_TABLE_SOURCE)
print("Feature count:", len(FEATURE_COLUMNS))
display(df.head())
display(df["attack"].value_counts().sort_index().rename_axis("attack").reset_index(name="rows"))
display(df.groupby(["dataset", "model", "attack"]).size().unstack(fill_value=0))
display(df["window_source"].value_counts(dropna=False).rename_axis("window_source").reset_index(name="rows"))
"""


HELPERS = """
@dataclass
class Candidate:
    family: str
    params: Dict[str, Any]
    variance_threshold: float
    corr_threshold: float
    top_k: Optional[int]
    resampler: str


def adaptive_stratified_folds(y: Sequence[int], max_splits: int = MAX_CV_SPLITS) -> int:
    counts = pd.Series(y).value_counts()
    if counts.empty:
        return 0
    return int(max(0, min(max_splits, counts.min())))


def select_features(
    train_df: pd.DataFrame,
    y_train: np.ndarray,
    feature_cols: Sequence[str],
    variance_threshold: float,
    corr_threshold: float,
    top_k: Optional[int],
) -> List[str]:
    X = train_df[list(feature_cols)].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    variances = X.var(axis=0)
    kept = [col for col in X.columns if float(variances[col]) > variance_threshold]
    if not kept:
        kept = list(feature_cols)

    X_kept = X[kept]
    if corr_threshold < 1.0 and len(kept) > 1:
        corr = X_kept.corr().abs().fillna(0.0)
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        drop = [col for col in upper.columns if any(upper[col] > corr_threshold)]
        kept = [col for col in kept if col not in drop]

    if top_k is not None and len(kept) > top_k:
        try:
            scores = mutual_info_classif(X[kept], y_train, random_state=RANDOM_STATE, discrete_features=False)
            ranked = pd.Series(scores, index=kept).sort_values(ascending=False)
            kept = ranked.head(top_k).index.tolist()
        except Exception:
            kept = kept[:top_k]

    return kept or list(feature_cols)


def clean_matrix(frame: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    return frame[list(cols)].replace([np.inf, -np.inf], np.nan).fillna(0.0)


def guarded_resample(X: pd.DataFrame, y: np.ndarray, strategy: str) -> Tuple[pd.DataFrame, np.ndarray, str]:
    if strategy == "none":
        return X, y, "none"

    counts = pd.Series(y).value_counts()
    if counts.empty or counts.min() < 2:
        if strategy in {"smote", "smoteenn"}:
            strategy = "ros"

    try:
        if strategy == "ros":
            sampler = RandomOverSampler(random_state=RANDOM_STATE)
        elif strategy == "smote":
            sampler = SMOTE(random_state=RANDOM_STATE, k_neighbors=1)
        elif strategy == "smoteenn":
            sampler = SMOTEENN(random_state=RANDOM_STATE, smote=SMOTE(random_state=RANDOM_STATE, k_neighbors=1))
        else:
            return X, y, "none"

        X_res, y_res = sampler.fit_resample(X, y)
        if len(np.unique(y_res)) < len(np.unique(y)):
            return X, y, "none"
        return pd.DataFrame(X_res, columns=X.columns), np.asarray(y_res), strategy
    except Exception:
        return X, y, "none"


def make_model(candidate: Candidate, n_classes: int):
    p = dict(candidate.params)
    if candidate.family == "xgb":
        return xgb.XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            num_class=n_classes,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
            **p,
        )
    if candidate.family == "catboost":
        return CatBoostClassifier(
            loss_function="MultiClass",
            random_seed=RANDOM_STATE,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
            **p,
        )
    if candidate.family == "extratrees":
        return ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced", **p)
    if candidate.family == "rf":
        return RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced", **p)
    if candidate.family == "mlp":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("mlp", MLPClassifier(random_state=RANDOM_STATE, max_iter=1000, early_stopping=True, **p)),
            ]
        )
    raise ValueError(f"Unknown family: {candidate.family}")


def fit_model(model: Any, X: pd.DataFrame, y: np.ndarray, use_sample_weight: bool = True) -> Any:
    if use_sample_weight and not isinstance(model, Pipeline):
        sample_weight = compute_sample_weight(class_weight="balanced", y=y)
        try:
            model.fit(X, y, sample_weight=sample_weight)
            return model
        except TypeError:
            pass
    model.fit(X, y)
    return model


def predict_labels(model: Any, X: pd.DataFrame) -> np.ndarray:
    pred = model.predict(X)
    pred = np.asarray(pred)
    if pred.ndim == 2 and pred.shape[1] > 1:
        pred = pred.argmax(axis=1)
    return pred.reshape(-1).astype(int)


def allowed_global_labels(monitored_model: str) -> np.ndarray:
    allowed = ALLOWED_ATTACKS_BY_MONITORED_MODEL.get(str(monitored_model), SUPPORTED_ATTACKS)
    labels = [label for label in allowed if label in set(encoder.classes_)]
    return encoder.transform(labels) if labels else np.arange(len(encoder.classes_))


def predict_global_labels(
    model: Any,
    X: pd.DataFrame,
    train_classes: np.ndarray,
    monitored_models: Sequence[str],
) -> np.ndarray:
    pred_local = predict_labels(model, X)
    pred_global = train_classes[np.clip(pred_local, 0, len(train_classes) - 1)]
    if not USE_CONSTRAINED_PREDICTIONS or not hasattr(model, "predict_proba"):
        return pred_global

    try:
        probs = np.asarray(model.predict_proba(X))
    except Exception:
        return pred_global
    if probs.ndim != 2 or probs.shape[1] != len(train_classes):
        return pred_global

    constrained = []
    for row_idx, monitored_model in enumerate(monitored_models):
        allowed = set(allowed_global_labels(str(monitored_model)).tolist())
        local_positions = [pos for pos, global_label in enumerate(train_classes) if int(global_label) in allowed]
        if not local_positions:
            constrained.append(pred_global[row_idx])
            continue
        best_pos = max(local_positions, key=lambda pos: probs[row_idx, pos])
        constrained.append(train_classes[best_pos])
    return np.asarray(constrained, dtype=int)


def metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {
            "bacc": float(balanced_accuracy_score(y_true, y_pred)),
            "kappa": float(cohen_kappa_score(y_true, y_pred)),
            "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
            "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        }
"""


EVALUATION = """
def fit_predict_split(train_df: pd.DataFrame, test_df: pd.DataFrame, candidate: Candidate) -> Tuple[np.ndarray, np.ndarray, List[str], Any, str]:
    y_train_global = encoder.transform(train_df["attack"])
    y_test = encoder.transform(test_df["attack"])
    train_classes = np.asarray(sorted(np.unique(y_train_global)))
    local_map = {label: idx for idx, label in enumerate(train_classes)}
    y_train = np.asarray([local_map[label] for label in y_train_global], dtype=int)
    selected = select_features(
        train_df,
        y_train_global,
        FEATURE_COLUMNS,
        candidate.variance_threshold,
        candidate.corr_threshold,
        candidate.top_k,
    )
    X_train = clean_matrix(train_df, selected)
    X_test = clean_matrix(test_df, selected)
    X_fit, y_fit, actual_resampler = guarded_resample(X_train, y_train, candidate.resampler)
    model = make_model(candidate, n_classes=len(train_classes))
    fit_model(model, X_fit, y_fit, use_sample_weight=(actual_resampler == "none"))
    y_pred = predict_global_labels(model, X_test, train_classes, test_df["model"].tolist())
    return y_test, y_pred, selected, model, actual_resampler


def evaluate_leave_group(candidate: Candidate, group_col: str, scenario: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    pred_rows = []
    for value in sorted(df[group_col].unique()):
        train_df = df[df[group_col] != value].copy()
        test_df = df[df[group_col] == value].copy()
        y_true, y_pred, selected, _model, actual_resampler = fit_predict_split(train_df, test_df, candidate)
        metrics = metric_row(y_true, y_pred)
        metric_rows.append(
            {
                "scenario": scenario,
                "split": value,
                "family": candidate.family,
                "feature_count": len(selected),
                "resampler": actual_resampler,
                **metrics,
            }
        )
        keep_cols = [c for c in ["dataset", "model", "attack", "window_id", "window_source"] if c in test_df.columns]
        out = test_df[keep_cols].copy()
        out["true"] = encoder.inverse_transform(y_true)
        out["pred"] = encoder.inverse_transform(np.clip(y_pred, 0, len(class_labels) - 1))
        out["scenario"] = scenario
        out["split"] = value
        out["family"] = candidate.family
        pred_rows.append(out)
    return pd.DataFrame(metric_rows), pd.concat(pred_rows, ignore_index=True)


def evaluate_cv(candidate: Candidate) -> Tuple[pd.DataFrame, pd.DataFrame]:
    y = encoder.transform(df["attack"])
    n_splits = adaptive_stratified_folds(y)
    if n_splits < 2:
        return pd.DataFrame(), pd.DataFrame()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    metric_rows = []
    pred_rows = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(df[FEATURE_COLUMNS], y)):
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()
        y_true, y_pred, selected, _model, actual_resampler = fit_predict_split(train_df, test_df, candidate)
        metric_rows.append(
            {
                "scenario": "10-fold cross-validation",
                "split": fold,
                "family": candidate.family,
                "feature_count": len(selected),
                "resampler": actual_resampler,
                **metric_row(y_true, y_pred),
            }
        )
        keep_cols = [c for c in ["dataset", "model", "attack", "window_id", "window_source"] if c in test_df.columns]
        out = test_df[keep_cols].copy()
        out["true"] = encoder.inverse_transform(y_true)
        out["pred"] = encoder.inverse_transform(np.clip(y_pred, 0, len(class_labels) - 1))
        out["scenario"] = "10-fold cross-validation"
        out["split"] = fold
        out["family"] = candidate.family
        pred_rows.append(out)
    return pd.DataFrame(metric_rows), pd.concat(pred_rows, ignore_index=True)


def evaluate_candidate(candidate: Candidate, include_cv: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metrics = []
    predictions = []
    m, p = evaluate_leave_group(candidate, "dataset", "one-data-set-out")
    metrics.append(m)
    predictions.append(p)
    m, p = evaluate_leave_group(candidate, "model", "one-model-out")
    metrics.append(m)
    predictions.append(p)
    if include_cv:
        m, p = evaluate_cv(candidate)
        if not m.empty:
            metrics.append(m)
            predictions.append(p)
    return pd.concat(metrics, ignore_index=True), pd.concat(predictions, ignore_index=True)


def selection_scores(candidate: Candidate) -> Dict[str, float]:
    dataset_metrics, _ = evaluate_leave_group(candidate, "dataset", "one-data-set-out")
    model_metrics, _ = evaluate_leave_group(candidate, "model", "one-model-out")
    dataset_bacc = float(dataset_metrics["bacc"].mean())
    dataset_kappa = float(dataset_metrics["kappa"].mean())
    model_bacc = float(model_metrics["bacc"].mean())
    model_kappa = float(model_metrics["kappa"].mean())
    if SELECTION_MODE == "dataset":
        combined_bacc = dataset_bacc
        combined_kappa = dataset_kappa
    else:
        combined_bacc = DATASET_SCORE_WEIGHT * dataset_bacc + MODEL_SCORE_WEIGHT * model_bacc
        combined_kappa = DATASET_SCORE_WEIGHT * dataset_kappa + MODEL_SCORE_WEIGHT * model_kappa
    return {
        "dataset_bacc": dataset_bacc,
        "dataset_kappa": dataset_kappa,
        "model_bacc": model_bacc,
        "model_kappa": model_kappa,
        "selection_bacc": combined_bacc,
        "selection_kappa": combined_kappa,
    }
"""


OPTUNA_SEARCH = """
def sample_common(trial: optuna.Trial) -> Dict[str, Any]:
    top_k_choice = trial.suggest_categorical("top_k", ["none", 12, 20, 35, 50])
    return {
        "variance_threshold": trial.suggest_categorical("variance_threshold", [0.0, 1e-10, 1e-6, 1e-4]),
        "corr_threshold": trial.suggest_categorical("corr_threshold", [0.90, 0.95, 0.98, 1.0]),
        "top_k": None if top_k_choice == "none" else int(top_k_choice),
        "resampler": trial.suggest_categorical("resampler", ["none", "ros", "smote", "smoteenn"]),
    }


def sample_candidate(trial: optuna.Trial, family: str) -> Candidate:
    common = sample_common(trial)
    if family == "xgb":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 80, 800, step=40),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.35, log=True),
            "subsample": trial.suggest_float("subsample", 0.55, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 8.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 2.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 8.0, log=True),
        }
    elif family == "catboost":
        params = {
            "iterations": trial.suggest_int("iterations", 80, 700, step=40),
            "depth": trial.suggest_int("depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.35, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.0, 4.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 3.0),
        }
    elif family == "extratrees":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 900, step=50),
            "max_depth": trial.suggest_categorical("max_depth", [None, 3, 5, 8, 12, 20]),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 8),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }
    elif family == "rf":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 900, step=50),
            "max_depth": trial.suggest_categorical("max_depth", [None, 3, 5, 8, 12, 20, 50]),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 8),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }
    elif family == "mlp":
        params = {
            "hidden_layer_sizes": trial.suggest_categorical("hidden_layer_sizes", [(16,), (32,), (32, 16), (64, 32)]),
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 5e-2, log=True),
            "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
        }
    else:
        raise ValueError(family)
    return Candidate(family=family, params=params, **common)


def optimize_family(family: str, n_trials: int = N_TRIALS_PER_FAMILY) -> Tuple[Candidate, pd.DataFrame]:
    def objective(trial: optuna.Trial) -> float:
        candidate = sample_candidate(trial, family)
        trial.set_user_attr("candidate", {
            "family": candidate.family,
            "params": candidate.params,
            "variance_threshold": candidate.variance_threshold,
            "corr_threshold": candidate.corr_threshold,
            "top_k": candidate.top_k,
            "resampler": candidate.resampler,
        })
        try:
            scores = selection_scores(candidate)
        except Exception as exc:
            trial.set_user_attr("error", repr(exc))
            return -999.0
        for key, value in scores.items():
            trial.set_user_attr(key, value)
        return scores["selection_bacc"] + 0.001 * scores["selection_kappa"]

    study = optuna.create_study(direction="maximize", study_name=f"attack_type_{family}")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best_attrs = study.best_trial.user_attrs["candidate"]
    best = Candidate(
        family=best_attrs["family"],
        params=best_attrs["params"],
        variance_threshold=best_attrs["variance_threshold"],
        corr_threshold=best_attrs["corr_threshold"],
        top_k=best_attrs["top_k"],
        resampler=best_attrs["resampler"],
    )
    rows = []
    for t in study.trials:
        row = {
            "family": family,
            "trial": t.number,
            "objective": t.value,
            "selection_bacc": t.user_attrs.get("selection_bacc", np.nan),
            "selection_kappa": t.user_attrs.get("selection_kappa", np.nan),
            "dataset_bacc": t.user_attrs.get("dataset_bacc", np.nan),
            "dataset_kappa": t.user_attrs.get("dataset_kappa", np.nan),
            "model_bacc": t.user_attrs.get("model_bacc", np.nan),
            "model_kappa": t.user_attrs.get("model_kappa", np.nan),
            "state": str(t.state),
        }
        row.update(t.params)
        rows.append(row)
    return best, pd.DataFrame(rows)


families = ["xgb", "catboost", "extratrees", "rf", "mlp"]
best_candidates: Dict[str, Candidate] = {}
trial_frames = []

for family in families:
    print(f"Optimizing {family}...")
    best, trials = optimize_family(family)
    best_candidates[family] = best
    trial_frames.append(trials)
    print(best)
    display(trials.sort_values(["selection_bacc", "selection_kappa"], ascending=False).head(5))

trials_all = pd.concat(trial_frames, ignore_index=True)
trials_path = RESULTS_DIR / "attack_type_optimized_trials.csv"
trials_all.to_csv(trials_path, index=False)
print(trials_path)
"""


VOTING = """
def fit_final_candidate(candidate: Candidate, train_df: pd.DataFrame) -> Tuple[Any, List[str], str]:
    y_train = encoder.transform(train_df["attack"])
    selected = select_features(
        train_df,
        y_train,
        FEATURE_COLUMNS,
        candidate.variance_threshold,
        candidate.corr_threshold,
        candidate.top_k,
    )
    X_train = clean_matrix(train_df, selected)
    X_fit, y_fit, actual_resampler = guarded_resample(X_train, y_train, candidate.resampler)
    model = make_model(candidate, n_classes=len(class_labels))
    fit_model(model, X_fit, y_fit, use_sample_weight=(actual_resampler == "none"))
    return model, selected, actual_resampler


def fit_final_group_candidate(candidate: Candidate, train_df: pd.DataFrame) -> Tuple[Any, LabelEncoder, List[str], str]:
    group_encoder = LabelEncoder()
    y_group = group_encoder.fit_transform(train_df["attack"].map(attack_group))
    selected = select_features(
        train_df,
        y_group,
        FEATURE_COLUMNS,
        candidate.variance_threshold,
        candidate.corr_threshold,
        candidate.top_k,
    )
    X_train = clean_matrix(train_df, selected)
    X_fit, y_fit, actual_resampler = guarded_resample(X_train, y_group, candidate.resampler)
    model = make_model(candidate, n_classes=len(group_encoder.classes_))
    fit_model(model, X_fit, y_fit, use_sample_weight=(actual_resampler == "none"))
    return model, group_encoder, selected, actual_resampler


def make_voting_candidate(base_candidates: Dict[str, Candidate]) -> Candidate:
    return Candidate(
        family="voting",
        params={"members": ["xgb", "catboost", "extratrees"]},
        variance_threshold=0.0,
        corr_threshold=0.98,
        top_k=None,
        resampler="none",
    )


def fit_predict_voting_split(train_df: pd.DataFrame, test_df: pd.DataFrame, members: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    y_train_global = encoder.transform(train_df["attack"])
    y_test = encoder.transform(test_df["attack"])
    train_classes = np.asarray(sorted(np.unique(y_train_global)))
    local_map = {label: idx for idx, label in enumerate(train_classes)}
    y_train = np.asarray([local_map[label] for label in y_train_global], dtype=int)
    selected_sets = []
    estimators = []
    for member in members:
        candidate = best_candidates[member]
        selected = select_features(
            train_df,
            y_train_global,
            FEATURE_COLUMNS,
            candidate.variance_threshold,
            candidate.corr_threshold,
            candidate.top_k,
        )
        selected_sets.append(set(selected))
    selected = sorted(set.intersection(*selected_sets)) if selected_sets else list(FEATURE_COLUMNS)
    if not selected:
        selected = list(FEATURE_COLUMNS)
    X_train = clean_matrix(train_df, selected)
    X_test = clean_matrix(test_df, selected)
    X_fit, y_fit, _actual_resampler = guarded_resample(X_train, y_train, "ros")
    for member in members:
        candidate = best_candidates[member]
        estimator = make_model(Candidate(member, candidate.params, 0.0, 1.0, None, "none"), len(train_classes))
        estimators.append((member, estimator))
    voter = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)
    voter.fit(X_fit, y_fit)
    y_pred = predict_global_labels(voter, X_test, train_classes, test_df["model"].tolist())
    return y_test, y_pred, selected


def evaluate_voting(members: Sequence[str] = ("xgb", "catboost", "extratrees")) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    pred_rows = []
    for group_col, scenario in [("dataset", "one-data-set-out"), ("model", "one-model-out")]:
        for value in sorted(df[group_col].unique()):
            train_df = df[df[group_col] != value].copy()
            test_df = df[df[group_col] == value].copy()
            y_true, y_pred, selected = fit_predict_voting_split(train_df, test_df, members)
            metric_rows.append(
                {
                    "scenario": scenario,
                    "split": value,
                    "family": "voting",
                    "feature_count": len(selected),
                    "resampler": "ros",
                    **metric_row(y_true, y_pred),
                }
            )
            keep_cols = [c for c in ["dataset", "model", "attack", "window_id", "window_source"] if c in test_df.columns]
            out = test_df[keep_cols].copy()
            out["true"] = encoder.inverse_transform(y_true)
            out["pred"] = encoder.inverse_transform(np.clip(y_pred, 0, len(class_labels) - 1))
            out["scenario"] = scenario
            out["split"] = value
            out["family"] = "voting"
            pred_rows.append(out)
    return pd.DataFrame(metric_rows), pd.concat(pred_rows, ignore_index=True)


voting_candidate = make_voting_candidate(best_candidates)
voting_metrics, voting_predictions = evaluate_voting()
display(voting_metrics)
"""


EVALUATE_SAVE = """
metric_frames = []
prediction_frames = []
candidate_records = {}

for family, candidate in best_candidates.items():
    print(f"Evaluating best {family}...")
    metrics, predictions = evaluate_candidate(candidate, include_cv=True)
    metric_frames.append(metrics)
    prediction_frames.append(predictions)
    dataset_primary = metrics[metrics["scenario"] == "one-data-set-out"]
    model_primary = metrics[metrics["scenario"] == "one-model-out"]
    dataset_bacc = float(dataset_primary["bacc"].mean())
    dataset_kappa = float(dataset_primary["kappa"].mean())
    model_bacc = float(model_primary["bacc"].mean())
    model_kappa = float(model_primary["kappa"].mean())
    candidate_records[family] = {
        "candidate": candidate,
        "dataset_bacc": dataset_bacc,
        "dataset_kappa": dataset_kappa,
        "model_bacc": model_bacc,
        "model_kappa": model_kappa,
        "selection_bacc": DATASET_SCORE_WEIGHT * dataset_bacc + MODEL_SCORE_WEIGHT * model_bacc,
        "selection_kappa": DATASET_SCORE_WEIGHT * dataset_kappa + MODEL_SCORE_WEIGHT * model_kappa,
    }

metric_frames.append(voting_metrics)
prediction_frames.append(voting_predictions)
dataset_voting = voting_metrics[voting_metrics["scenario"] == "one-data-set-out"]
model_voting = voting_metrics[voting_metrics["scenario"] == "one-model-out"]
dataset_bacc = float(dataset_voting["bacc"].mean())
dataset_kappa = float(dataset_voting["kappa"].mean())
model_bacc = float(model_voting["bacc"].mean())
model_kappa = float(model_voting["kappa"].mean())
candidate_records["voting"] = {
    "candidate": voting_candidate,
    "dataset_bacc": dataset_bacc,
    "dataset_kappa": dataset_kappa,
    "model_bacc": model_bacc,
    "model_kappa": model_kappa,
    "selection_bacc": DATASET_SCORE_WEIGHT * dataset_bacc + MODEL_SCORE_WEIGHT * model_bacc,
    "selection_kappa": DATASET_SCORE_WEIGHT * dataset_kappa + MODEL_SCORE_WEIGHT * model_kappa,
}

metrics_all = pd.concat(metric_frames, ignore_index=True)
predictions_all = pd.concat(prediction_frames, ignore_index=True)

leaderboard = (
    pd.DataFrame(
        [
            {
                "family": family,
                "selection_bacc": rec["selection_bacc"],
                "selection_kappa": rec["selection_kappa"],
                "dataset_bacc": rec["dataset_bacc"],
                "dataset_kappa": rec["dataset_kappa"],
                "model_bacc": rec["model_bacc"],
                "model_kappa": rec["model_kappa"],
            }
            for family, rec in candidate_records.items()
        ]
    )
    .sort_values(["selection_bacc", "selection_kappa"], ascending=False)
    .reset_index(drop=True)
)

display(leaderboard)
display(metrics_all.groupby(["scenario", "family"])[["bacc", "kappa", "precision_weighted", "recall_weighted", "f1_weighted"]].agg(["mean", "std"]).round(4))
"""


REPORTS = """
def save_confusion_matrices(predictions: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    rows = []
    labels = class_labels
    for (scenario, family), group in predictions.groupby(["scenario", "family"]):
        cm = confusion_matrix(group["true"], group["pred"], labels=labels)
        for i, true_label in enumerate(labels):
            for j, pred_label in enumerate(labels):
                rows.append(
                    {
                        "scenario": scenario,
                        "family": family,
                        "true": true_label,
                        "pred": pred_label,
                        "count": int(cm[i, j]),
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    return out


def per_class_report(predictions: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    rows = []
    for (scenario, family), group in predictions.groupby(["scenario", "family"]):
        report = classification_report(group["true"], group["pred"], labels=class_labels, output_dict=True, zero_division=0)
        for label in class_labels:
            rows.append(
                {
                    "scenario": scenario,
                    "family": family,
                    "label": label,
                    "precision": report[label]["precision"],
                    "recall": report[label]["recall"],
                    "f1": report[label]["f1-score"],
                    "support": report[label]["support"],
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    return out


def attack_group(label: str) -> str:
    if label in GRADIENT_ATTACKS:
        return "gradient_attack"
    if label in BLACKBOX_ATTACKS:
        return "blackbox_attack"
    return "org"


def grouped_attack_metrics(predictions: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    rows = []
    grouped = predictions.copy()
    grouped["true_group"] = grouped["true"].map(attack_group)
    grouped["pred_group"] = grouped["pred"].map(attack_group)
    for (scenario, family), group in grouped.groupby(["scenario", "family"]):
        y_true = group["true_group"]
        y_pred = group["pred_group"]
        rows.append(
            {
                "scenario": scenario,
                "family": family,
                "bacc": balanced_accuracy_score(y_true, y_pred),
                "kappa": cohen_kappa_score(y_true, y_pred),
                "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
                "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
                "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    return out


metrics_path = RESULTS_DIR / "attack_type_optimized_metrics.csv"
predictions_path = RESULTS_DIR / "attack_type_optimized_predictions.csv"
confusion_path = RESULTS_DIR / "attack_type_optimized_confusion_matrices.csv"
per_class_path = RESULTS_DIR / "attack_type_optimized_per_class_metrics.csv"
grouped_metrics_path = RESULTS_DIR / "attack_type_optimized_grouped_metrics.csv"
best_params_path = RESULTS_DIR / "attack_type_optimized_best_params.json"

metrics_all.to_csv(metrics_path, index=False)
predictions_all.to_csv(predictions_path, index=False)
confusion_df = save_confusion_matrices(predictions_all, confusion_path)
per_class_df = per_class_report(predictions_all, per_class_path)
grouped_metrics_df = grouped_attack_metrics(predictions_all, grouped_metrics_path)

best_family = leaderboard.iloc[0]["family"]
best_record = candidate_records[best_family]
best_candidate = best_record["candidate"]

best_params_payload = {
    "selected_family": best_family,
    "selection_mode": SELECTION_MODE,
    "dataset_score_weight": DATASET_SCORE_WEIGHT,
    "model_score_weight": MODEL_SCORE_WEIGHT,
    "selection_bacc": best_record["selection_bacc"],
    "selection_kappa": best_record["selection_kappa"],
    "dataset_bacc": best_record["dataset_bacc"],
    "dataset_kappa": best_record["dataset_kappa"],
    "model_bacc": best_record["model_bacc"],
    "model_kappa": best_record["model_kappa"],
    "leaderboard": leaderboard.to_dict(orient="records"),
    "candidates": {
        family: {
            "family": rec["candidate"].family,
            "params": rec["candidate"].params,
            "variance_threshold": rec["candidate"].variance_threshold,
            "corr_threshold": rec["candidate"].corr_threshold,
            "top_k": rec["candidate"].top_k,
            "resampler": rec["candidate"].resampler,
            "selection_bacc": rec["selection_bacc"],
            "selection_kappa": rec["selection_kappa"],
            "dataset_bacc": rec["dataset_bacc"],
            "dataset_kappa": rec["dataset_kappa"],
            "model_bacc": rec["model_bacc"],
            "model_kappa": rec["model_kappa"],
        }
        for family, rec in candidate_records.items()
    },
}
best_params_path.write_text(json.dumps(best_params_payload, indent=2), encoding="utf-8")

print("Saved:")
for path in [metrics_path, predictions_path, confusion_path, per_class_path, grouped_metrics_path, trials_path, best_params_path]:
    print(path)
"""


FINAL_MODEL = """
def final_feature_importance(model: Any, feature_cols: Sequence[str], family: str) -> pd.DataFrame:
    if family == "voting":
        rows = []
        for name, estimator in model.named_estimators_.items():
            importance = getattr(estimator, "feature_importances_", None)
            if importance is not None:
                rows.append(pd.DataFrame({"family": name, "var": list(feature_cols), "importance": importance}))
        if rows:
            out = pd.concat(rows, ignore_index=True)
            avg = out.groupby("var", as_index=False)["importance"].mean()
            avg["family"] = "voting_mean"
            avg["rank"] = avg["importance"].rank(ascending=False)
            return avg.sort_values("rank")
        return pd.DataFrame(columns=["family", "var", "importance", "rank"])
    raw_model = model
    if isinstance(model, Pipeline):
        raw_model = model.steps[-1][1]
    importance = getattr(raw_model, "feature_importances_", None)
    if importance is None:
        return pd.DataFrame(columns=["family", "var", "importance", "rank"])
    out = pd.DataFrame({"family": family, "var": list(feature_cols), "importance": importance})
    out["rank"] = out["importance"].rank(ascending=False)
    return out.sort_values("rank")


if best_family == "voting":
    members = best_candidate.params["members"]
    selected_sets = []
    y_full = encoder.transform(df["attack"])
    for member in members:
        candidate = best_candidates[member]
        selected_sets.append(
            set(
                select_features(
                    df,
                    y_full,
                    FEATURE_COLUMNS,
                    candidate.variance_threshold,
                    candidate.corr_threshold,
                    candidate.top_k,
                )
            )
        )
    final_features = sorted(set.intersection(*selected_sets)) if selected_sets else list(FEATURE_COLUMNS)
    if not final_features:
        final_features = list(FEATURE_COLUMNS)
    X_full = clean_matrix(df, final_features)
    X_fit, y_fit, final_resampler = guarded_resample(X_full, y_full, "ros")
    estimators = []
    for member in members:
        candidate = best_candidates[member]
        estimators.append((member, make_model(Candidate(member, candidate.params, 0.0, 1.0, None, "none"), len(class_labels))))
    final_model = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)
    final_model.fit(X_fit, y_fit)
else:
    final_model, final_features, final_resampler = fit_final_candidate(best_candidate, df)

model_path = MODEL_DIR / "attack_type_optimized_best.joblib"
encoder_path = MODEL_DIR / "attack_type_optimized_label_encoder.joblib"
features_path = MODEL_DIR / "attack_type_optimized_features.json"
config_path = MODEL_DIR / "attack_type_optimized_config.json"
family_model_path = MODEL_DIR / "attack_family_optimized_best.joblib"
family_encoder_path = MODEL_DIR / "attack_family_optimized_label_encoder.joblib"
family_features_path = MODEL_DIR / "attack_family_optimized_features.json"
family_config_path = MODEL_DIR / "attack_family_optimized_config.json"
fi_path = RESULTS_DIR / "attack_type_optimized_feature_importance.csv"

joblib.dump(final_model, model_path)
joblib.dump(encoder, encoder_path)
features_path.write_text(json.dumps(final_features, indent=2), encoding="utf-8")

family_model, family_encoder, family_features, family_resampler = fit_final_group_candidate(best_candidate, df)
joblib.dump(family_model, family_model_path)
joblib.dump(family_encoder, family_encoder_path)
family_features_path.write_text(json.dumps(family_features, indent=2), encoding="utf-8")

config = {
    "task": "attack_type",
    "selected_family": best_family,
    "params": best_candidate.params,
    "feature_columns": final_features,
    "preprocessing": {
        "variance_threshold": best_candidate.variance_threshold,
        "corr_threshold": best_candidate.corr_threshold,
        "top_k": best_candidate.top_k,
    },
    "resampling_strategy_requested": best_candidate.resampler,
    "resampling_strategy_final": final_resampler,
    "classes": class_labels,
    "attack_groups": {
        "gradient_attack": sorted(GRADIENT_ATTACKS),
        "blackbox_attack": sorted(BLACKBOX_ATTACKS),
        "org": ["org"],
    },
    "prediction_constraints_enabled": USE_CONSTRAINED_PREDICTIONS,
    "allowed_attacks_by_monitored_model": ALLOWED_ATTACKS_BY_MONITORED_MODEL,
    "selection_mode": SELECTION_MODE,
    "dataset_score_weight": DATASET_SCORE_WEIGHT,
    "model_score_weight": MODEL_SCORE_WEIGHT,
    "selection_bacc": best_record["selection_bacc"],
    "selection_kappa": best_record["selection_kappa"],
    "dataset_bacc": best_record["dataset_bacc"],
    "dataset_kappa": best_record["dataset_kappa"],
    "model_bacc": best_record["model_bacc"],
    "model_kappa": best_record["model_kappa"],
    "training_table_source": TRAINING_TABLE_SOURCE,
    "windowed_aggregation_enabled": WINDOWED_AGGREGATION_ENABLED,
    "notes": "Experimental optimized attack-type classifier. Current table is very small; validate on new diagnostic rows before relying on it.",
}
config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

family_config = {
    "task": "attack_family",
    "selected_family": best_family,
    "params": best_candidate.params,
    "feature_columns": family_features,
    "preprocessing": {
        "variance_threshold": best_candidate.variance_threshold,
        "corr_threshold": best_candidate.corr_threshold,
        "top_k": best_candidate.top_k,
    },
    "resampling_strategy_requested": best_candidate.resampler,
    "resampling_strategy_final": family_resampler,
    "classes": list(family_encoder.classes_),
    "attack_groups": {
        "gradient_attack": sorted(GRADIENT_ATTACKS),
        "blackbox_attack": sorted(BLACKBOX_ATTACKS),
        "org": ["org"],
    },
    "selection_source": "same candidate family/params selected by exact attack-type combined score",
    "training_table_source": TRAINING_TABLE_SOURCE,
    "windowed_aggregation_enabled": WINDOWED_AGGREGATION_ENABLED,
    "notes": "Direct grouped attack-family classifier. Intended for demo/report use when exact bim/fgm/pgd separation is unstable.",
}
family_config_path.write_text(json.dumps(family_config, indent=2), encoding="utf-8")

fi = final_feature_importance(final_model, final_features, best_family)
fi.to_csv(fi_path, index=False)

print("Saved final optimized attack-type artifacts:")
for path in [
    model_path,
    encoder_path,
    features_path,
    config_path,
    family_model_path,
    family_encoder_path,
    family_features_path,
    family_config_path,
    fi_path,
]:
    print(path)
display(fi.head(20))
"""


RELOAD_TEST = """
loaded_model = joblib.load(model_path)
loaded_encoder = joblib.load(encoder_path)
loaded_features = json.loads(features_path.read_text(encoding="utf-8"))
sample_X = clean_matrix(df.iloc[[0]], loaded_features)
pred_enc = predict_labels(loaded_model, sample_X)
pred_label = loaded_encoder.inverse_transform(np.clip(pred_enc, 0, len(loaded_encoder.classes_) - 1))[0]

print("Reload test prediction:", pred_label)
print("Allowed labels:", list(loaded_encoder.classes_))
assert pred_label in set(loaded_encoder.classes_)

loaded_family_model = joblib.load(family_model_path)
loaded_family_encoder = joblib.load(family_encoder_path)
loaded_family_features = json.loads(family_features_path.read_text(encoding="utf-8"))
family_X = clean_matrix(df.iloc[[0]], loaded_family_features)
family_pred_enc = predict_labels(loaded_family_model, family_X)
family_pred_label = loaded_family_encoder.inverse_transform(
    np.clip(family_pred_enc, 0, len(loaded_family_encoder.classes_) - 1)
)[0]

print("Reload family prediction:", family_pred_label)
print("Allowed family labels:", list(loaded_family_encoder.classes_))
assert family_pred_label in set(loaded_family_encoder.classes_)
"""


SUMMARY = """
summary = metrics_all.groupby(["scenario", "family"])[["bacc", "kappa", "f1_weighted"]].agg(["mean", "std"]).round(4)
group_summary = grouped_metrics_df.groupby(["scenario", "family"])[["bacc", "kappa", "f1_weighted"]].mean().round(4)
print("Current baseline reference from previous run:")
print("  RF leave-one-dataset-out mean bacc ~= 0.3704")
print("  XGB leave-one-dataset-out mean bacc ~= 0.3935")
print()
print("Optimized leaderboard:")
display(leaderboard)
display(summary)
print("Grouped attack labels: gradient_attack=(bim, fgm, pgd), blackbox_attack=(hsj, zoo), org=org")
display(group_summary)
"""


def main() -> None:
    embedded_table_path = Path(__file__).resolve().parents[1] / "results" / "attr_attacks_type_agr_nn.csv"
    embedded_table_json = pd.read_csv(embedded_table_path).to_json(orient="records")
    setup_cell = SETUP.replace(
        "__EMBEDDED_ATTACK_TYPE_TABLE_JSON__",
        json.dumps(embedded_table_json),
    )
    cells = [
        markdown(INTRO),
        markdown("## 1. Imports and Environment"),
        code(IMPORTS),
        code(setup_cell),
        markdown("## 2. Load Aggregated Diagnostic Table"),
        code(LOAD_DATA),
        markdown("## 3. Standalone Training and Evaluation Helpers"),
        code(HELPERS),
        code(EVALUATION),
        markdown("## 4. Optuna Search"),
        code(OPTUNA_SEARCH),
        markdown("## 5. Soft-Voting Candidate"),
        code(VOTING),
        markdown("## 6. Evaluate Best Candidates and Save Reports"),
        code(EVALUATE_SAVE),
        code(REPORTS),
        markdown("## 7. Fit and Save Final Deployable Model"),
        code(FINAL_MODEL),
        markdown("## 8. Reload Smoke Test"),
        code(RELOAD_TEST),
        markdown("## 9. Summary"),
        code(SUMMARY),
    ]
    out = Path(__file__).resolve().parent / "attack_type_classifier_optimized_optuna.ipynb"
    out.write_text(json.dumps(notebook(cells), indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
