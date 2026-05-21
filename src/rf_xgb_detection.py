from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.svm import SVC

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover - surfaced clearly in notebooks.
    xgb = None


RANDOM_STATE = 123

DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    "wilt": {
        "target": "target",
        "drop_cols": ["name", "is_train", "prediction"],
    },
    "banknote": {
        "target": "Class",
        "column_names": ["variance", "skewness", "curtosis", "entropy", "Class"],
    },
    "diabetes": {
        "target": "Outcome",
        "column_names": [
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age",
            "Outcome",
        ],
    },
}

MODEL_ORDER = ["lin", "svm", "xgb", "nn"]
NON_NN_MODELS = ["lin", "svm", "xgb"]

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

XGB_PARAM_GRID = {
    "max_depth": [6, 9, 12],
    "learning_rate": [0.1, 0.3, 0.5],
    "n_estimators": [100, 200, 500],
}

RF_PARAM_GRID = {
    "max_depth": [50, 80, 110],
    "min_samples_split": [2, 5, 8],
    "n_estimators": [100, 200, 500],
}


@dataclass
class CleanSplit:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    n_classes: int


@dataclass
class AttackFile:
    dataset: str
    model: str
    attack: str
    epsilon: Optional[float]
    path: Path


def q0(x: pd.Series) -> float:
    return float(x.quantile(0))


def q25(x: pd.Series) -> float:
    return float(x.quantile(0.25))


def q50(x: pd.Series) -> float:
    return float(x.quantile(0.5))


def q75(x: pd.Series) -> float:
    return float(x.quantile(0.75))


def q1(x: pd.Series) -> float:
    return float(x.quantile(1))


def minmax(x: pd.Series) -> float:
    return float(x.max() - x.min())


AGG_FUNCS = ["mean", q0, q25, q50, q75, q1, minmax]


def find_repo_root(start: Optional[Path] = None) -> Path:
    """Find a directory containing the repo data layout, including Kaggle inputs."""

    start = Path(start or Path.cwd()).resolve()
    candidates: List[Path] = [start, *start.parents]

    for base in [Path("/kaggle/working"), Path("/kaggle/input")]:
        if base.exists():
            candidates.append(base)
            candidates.extend([p for p in base.glob("*") if p.is_dir()])
            candidates.extend([p for p in base.glob("*/*") if p.is_dir()])

    try:
        module_path = Path(__file__).resolve()
        module_root = module_path.parents[1] if len(module_path.parents) > 1 else module_path.parent
        candidates.extend([module_root, *module_root.parents])
    except NameError:
        # __file__ is not defined when the full script is pasted into a notebook cell.
        pass

    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "data" / "adversarial_upstream").exists():
            return candidate

    raise FileNotFoundError(
        "Could not find repo root with data/adversarial_upstream. "
        "On Kaggle, add this repository as an input dataset."
    )


def default_results_dir(repo_root: Optional[Path] = None) -> Path:
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working") / "results"
    root = repo_root or find_repo_root()
    return root / "results"


def _read_raw_dataset(repo_root: Path, dataset: str) -> pd.DataFrame:
    info = DATASET_REGISTRY[dataset]
    raw_path = repo_root / "data" / "raw" / f"{dataset}.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw dataset: {raw_path}")
    if "column_names" in info:
        return pd.read_csv(raw_path, names=info["column_names"])
    return pd.read_csv(raw_path)


def load_clean_split(repo_root: Path, dataset: str) -> CleanSplit:
    info = DATASET_REGISTRY[dataset]
    df = _read_raw_dataset(repo_root, dataset).copy()
    target_col = info["target"]

    y = df[target_col]
    if y.dtype == object:
        df[target_col] = LabelEncoder().fit_transform(y)

    metadata_cols = [target_col, "name", "prediction", "is_train"]
    metadata_cols.extend([c for c in df.columns if str(c).startswith("score_")])

    if "is_train" in df.columns:
        train_df = df[df["is_train"] == 1]
        test_df = df[df["is_train"] == 0]
        X_train = train_df.drop(columns=[c for c in metadata_cols if c in train_df.columns])
        X_test = test_df.drop(columns=[c for c in metadata_cols if c in test_df.columns])
        y_train = train_df[target_col].to_numpy()
        y_test = test_df[target_col].to_numpy()
    else:
        drop_cols = info.get("drop_cols", [])
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        X = df.drop(columns=[target_col])
        y_values = df[target_col].to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_values,
            test_size=0.2,
            random_state=42,
            stratify=y_values,
        )

    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test).reindex(columns=X_train.columns, fill_value=0)

    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)
    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)
    n_classes = int(max(np.max(y_train), np.max(y_test)) + 1)

    return CleanSplit(
        X_train=X_train_s,
        X_test=X_test_s,
        y_train=y_train,
        y_test=y_test,
        feature_names=list(X_train.columns),
        n_classes=n_classes,
    )


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def _predict_nn(repo_root: Path, dataset: str, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    import torch
    import torch.nn as nn

    class TabDataModel2(nn.Module):
        def __init__(self, input_dim: int, num_classes: int):
            super().__init__()
            self.input = nn.Linear(input_dim, 32)
            self.relu = nn.ReLU()
            self.output = nn.Linear(32, num_classes)

        def forward(self, x):
            return self.output(self.relu(self.input(x)))

    model_path = repo_root / "models" / dataset / "upstream_nn.pt"
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    checkpoint = torch.load(model_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model = TabDataModel2(checkpoint["input_dim"], checkpoint["num_classes"])
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model = checkpoint
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32)).detach().cpu().numpy()
    probs = _softmax(logits)
    return probs.argmax(axis=1).astype(np.int64), probs.astype(np.float32)


def predict_upstream_model(
    repo_root: Path,
    dataset: str,
    model_name: str,
    X: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if model_name == "nn":
        return _predict_nn(repo_root, dataset, X)

    model_path = repo_root / "models" / dataset / f"upstream_{model_name}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    model = joblib.load(model_path)
    probs = model.predict_proba(X)
    return probs.argmax(axis=1).astype(np.int64), np.asarray(probs, dtype=np.float32)


def build_clean_org_dataframe(
    repo_root: Path,
    dataset: str,
    model_name: str,
    split: CleanSplit,
) -> pd.DataFrame:
    try:
        pred, probs = predict_upstream_model(repo_root, dataset, model_name, split.X_test)
    except Exception as exc:
        # Fallback for Kaggle inputs that include attacks but not the NN checkpoint.
        if model_name == "nn":
            eps0 = sorted(
                (repo_root / "data" / "adversarial_upstream" / dataset).glob(f"{model_name}_*_eps_0.00.csv")
            )
            if not eps0:
                raise exc
            fallback = pd.read_csv(eps0[0])
            return fallback.copy()

        fallback_model = train_fallback_upstream_model(model_name, split)
        probs = fallback_model.predict_proba(split.X_test)
        pred = probs.argmax(axis=1).astype(np.int64)

    df = pd.DataFrame(split.X_test, columns=split.feature_names)
    df["name"] = np.arange(len(df))
    df["is_train"] = 0
    df["target"] = split.y_test
    df["prediction"] = pred
    for class_idx in range(probs.shape[1]):
        df[f"score_{class_idx}"] = probs[:, class_idx]
    return df


def train_fallback_upstream_model(model_name: str, split: CleanSplit):
    """Retrain a compatible upstream model if an old pickle cannot be loaded."""

    if model_name == "lin":
        model = LogisticRegression(max_iter=1000, random_state=42)
    elif model_name == "svm":
        model = SVC(probability=True, random_state=42)
    elif model_name == "xgb":
        model = GradientBoostingClassifier(n_estimators=500, random_state=42)
    else:
        raise ValueError(f"No sklearn fallback for upstream model {model_name!r}")
    model.fit(split.X_train, split.y_train)
    return model


def parse_attack_file(path: Path) -> AttackFile:
    match = re.fullmatch(
        r"(?P<model>[^_]+)_(?P<attack>[a-z]+)(?:_eps_(?P<eps>[0-9.]+))?",
        path.stem,
    )
    if not match:
        raise ValueError(f"Cannot parse attack file name: {path.name}")
    eps = match.group("eps")
    return AttackFile(
        dataset=path.parent.name,
        model=match.group("model"),
        attack=match.group("attack"),
        epsilon=(float(eps) if eps is not None else None),
        path=path,
    )


def select_attack_files(repo_root: Path) -> List[AttackFile]:
    """Use one file per algorithm, selecting the strongest nonzero NN epsilon."""

    adv_root = repo_root / "data" / "adversarial_upstream"
    selected: List[AttackFile] = []
    grouped: Dict[Tuple[str, str, str], List[AttackFile]] = {}

    for path in sorted(adv_root.glob("*/*.csv")):
        record = parse_attack_file(path)
        grouped.setdefault((record.dataset, record.model, record.attack), []).append(record)

    for (_dataset, model_name, _attack), records in sorted(grouped.items()):
        if model_name == "nn":
            nonzero = [r for r in records if r.epsilon is not None and r.epsilon > 0]
            if nonzero:
                selected.append(max(nonzero, key=lambda r: r.epsilon or 0.0))
            else:
                selected.append(max(records, key=lambda r: r.epsilon or 0.0))
        else:
            selected.append(records[0])
    return selected


def _score_columns(df: pd.DataFrame) -> List[str]:
    return sorted(
        [c for c in df.columns if str(c).startswith("score_")],
        key=lambda c: int(str(c).split("_")[-1]),
    )


def _normalized_probs(row: pd.Series, score_cols: Sequence[str], n_classes: int) -> np.ndarray:
    if score_cols:
        probs = row[list(score_cols)].to_numpy(dtype=np.float64)
    else:
        probs = np.zeros(n_classes, dtype=np.float64)
        probs[int(row["prediction"])] = 1.0
    probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    probs = np.clip(probs, 1e-12, None)
    total = probs.sum()
    if total <= 0:
        probs = np.ones(n_classes, dtype=np.float64) / n_classes
    else:
        probs = probs / total
    return probs


def _normalized_entropy(probs: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 1e-12, 1.0)
    probs = probs / probs.sum()
    if len(probs) <= 1:
        return 0.0
    return float(-np.sum(probs * np.log(probs)) / math.log(len(probs)))


def _label_entropy(labels: np.ndarray, n_classes: int) -> float:
    if len(labels) == 0:
        return 0.0
    counts = np.bincount(labels.astype(int), minlength=n_classes).astype(np.float64)
    probs = counts / max(counts.sum(), 1.0)
    return _normalized_entropy(probs)


def _safe_mode(labels: np.ndarray, n_classes: int) -> int:
    if len(labels) == 0:
        return 0
    counts = np.bincount(labels.astype(int), minlength=n_classes)
    return int(np.argmax(counts))


class NeighborhoodDiagnostics:
    def __init__(
        self,
        clean_df: pd.DataFrame,
        feature_names: Sequence[str],
        n_classes: int,
        max_neighbors: int = 32,
    ):
        self.clean_df = clean_df.reset_index(drop=True).copy()
        self.feature_names = list(feature_names)
        self.n_classes = int(n_classes)
        self.max_neighbors = int(max_neighbors)
        self.clean_X = self.clean_df[self.feature_names].to_numpy(dtype=np.float32)
        self.clean_target = self.clean_df["target"].to_numpy(dtype=np.int64)
        self.clean_pred = self.clean_df["prediction"].to_numpy(dtype=np.int64)
        self.score_cols = _score_columns(clean_df)
        self.clean_entropy = np.array(
            [
                _normalized_entropy(_normalized_probs(row, self.score_cols, self.n_classes))
                for _, row in self.clean_df.iterrows()
            ],
            dtype=np.float64,
        )
        self.n_neighbors = max(1, min(self.max_neighbors, len(self.clean_df)))
        self.nn = NearestNeighbors(n_neighbors=self.n_neighbors, metric="euclidean")
        self.nn.fit(self.clean_X)
        clean_distances, clean_indices = self.nn.kneighbors(self.clean_X, return_distance=True)
        kth = clean_distances[:, -1]
        radius = float(np.nanmedian(kth))
        if radius <= 0:
            radius = float(np.nanpercentile(kth, 75))
        self.radius = radius if radius > 0 else 1e-6
        self.clean_neighborhood_sizes = self._neighborhood_sizes(clean_distances)
        self.mean_neighborhood_size = float(np.mean(self.clean_neighborhood_sizes))

    def _neighborhood_sizes(self, distances: np.ndarray) -> np.ndarray:
        return np.maximum((distances <= self.radius).sum(axis=1), 1)

    def diagnose(
        self,
        df: pd.DataFrame,
        dataset: str,
        model_name: str,
        attack: str,
        bacc_test: float,
    ) -> pd.DataFrame:
        df = df.reset_index(drop=True).copy()
        X = df[self.feature_names].to_numpy(dtype=np.float32)
        distances, indices = self.nn.kneighbors(X, return_distance=True)
        score_cols = _score_columns(df)
        rows: List[Dict[str, Any]] = []
        overall_mean_target = float(np.mean(self.clean_target))

        for row_idx, (_, row) in enumerate(df.iterrows()):
            selected = indices[row_idx][distances[row_idx] <= self.radius]
            if len(selected) == 0:
                selected = indices[row_idx][:1]

            neigh_targets = self.clean_target[selected]
            neigh_preds = self.clean_pred[selected]
            target = int(row["target"])
            pred = int(row["prediction"])
            probs = _normalized_probs(row, score_cols, self.n_classes)
            entropy = _normalized_entropy(probs)
            approx = _safe_mode(neigh_preds, self.n_classes)
            neighborhood_size = int(len(selected))

            mean_target = float(np.mean(neigh_targets)) if len(neigh_targets) else overall_mean_target
            mean_approx = float(np.mean(neigh_preds)) if len(neigh_preds) else float(approx)
            r_centered_entropy = float(abs(entropy - np.mean(self.clean_entropy)))

            rows.append(
                {
                    "name": row.get("name", row_idx),
                    "approx": approx,
                    "target": target,
                    "pred": pred,
                    "error": int(pred != target),
                    "scores": "|".join(f"{p:.8f}" for p in probs),
                    "overall_mean_target": overall_mean_target,
                    "mean_target_in_neighborhood": mean_target,
                    "mean_approx_in_neighborhood": mean_approx,
                    "neighborhood_size": neighborhood_size,
                    "neighborhood_size_pct": neighborhood_size / len(self.clean_df) * 100.0,
                    "neighborhood_size_div_model_avg": neighborhood_size
                    / max(self.mean_neighborhood_size, 1e-12),
                    "entropy": entropy,
                    "uncertainty": entropy,
                    "r_centered_entropy": r_centered_entropy,
                    "logk_r_centered_entropy": math.log1p(self.n_classes * r_centered_entropy),
                    "target_approx_consistency_in_neighborhood": float(np.mean(neigh_preds == target)),
                    "pred_targets_consistency_in_neighborhood": float(np.mean(neigh_targets == pred)),
                    "target_targets_consistency_in_neighborhood": float(np.mean(neigh_targets == target)),
                    "targets_and_approxs_consistency_in_neighborhood": float(
                        np.mean(neigh_targets == neigh_preds)
                    ),
                    "target_diversity_in_neighborhood": _label_entropy(neigh_targets, self.n_classes),
                    "approx_diversity_in_neighborhood": _label_entropy(neigh_preds, self.n_classes),
                    "dataset": dataset,
                    "model": model_name,
                    "attack": attack,
                    "n_test": len(self.clean_df),
                    "n_classes": self.n_classes,
                    "bacc_test": bacc_test,
                }
            )

        return pd.DataFrame(rows)


def generate_diagnostic_csvs(
    repo_root: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    recompute: bool = True,
) -> Tuple[Path, Path]:
    repo_root = Path(repo_root or find_repo_root()).resolve()
    output_dir = Path(output_dir or default_results_dir(repo_root)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base_out = output_dir / "attacks_diagnoses.csv"
    nn_out = output_dir / "attacks_diagnoses_nn.csv"

    if not recompute and base_out.exists() and nn_out.exists():
        return base_out, nn_out

    attack_files = select_attack_files(repo_root)
    attack_by_dataset_model: Dict[Tuple[str, str], List[AttackFile]] = {}
    for record in attack_files:
        attack_by_dataset_model.setdefault((record.dataset, record.model), []).append(record)

    base_frames: List[pd.DataFrame] = []
    nn_frames: List[pd.DataFrame] = []
    datasets = sorted(p.name for p in (repo_root / "data" / "adversarial_upstream").iterdir() if p.is_dir())

    for dataset in datasets:
        split = load_clean_split(repo_root, dataset)
        for model_name in MODEL_ORDER:
            clean_org = build_clean_org_dataframe(repo_root, dataset, model_name, split)
            clean_org = clean_org.reindex(columns=[*split.feature_names, "name", "is_train", "target", "prediction", *_score_columns(clean_org)])
            context = NeighborhoodDiagnostics(clean_org, split.feature_names, split.n_classes)
            clean_bacc = balanced_accuracy_score(clean_org["target"], clean_org["prediction"])
            org_diag = context.diagnose(clean_org, dataset, model_name, "org", clean_bacc)
            target_frames = nn_frames if model_name == "nn" else base_frames
            target_frames.append(org_diag)

            for record in sorted(attack_by_dataset_model.get((dataset, model_name), []), key=lambda r: r.attack):
                adv_df = pd.read_csv(record.path)
                missing = [c for c in split.feature_names if c not in adv_df.columns]
                if missing:
                    raise ValueError(f"{record.path} is missing feature columns: {missing}")
                bacc = balanced_accuracy_score(adv_df["target"], adv_df["prediction"])
                diag = context.diagnose(adv_df, dataset, model_name, record.attack, bacc)
                target_frames.append(diag)

    pd.concat(base_frames, ignore_index=True).to_csv(base_out, index=False)
    pd.concat(nn_frames, ignore_index=True).to_csv(nn_out, index=False)
    return base_out, nn_out


def _aggregate(
    df: pd.DataFrame,
    group_cols: Sequence[str],
) -> pd.DataFrame:
    attrs = df.drop(columns=DIAGNOSTIC_DROP_COLUMNS, errors="ignore").copy()
    feature_cols = [
        c
        for c in attrs.columns
        if c not in group_cols and pd.api.types.is_numeric_dtype(attrs[c])
    ]
    grouped = attrs[list(group_cols) + feature_cols].groupby(list(group_cols)).agg(AGG_FUNCS)
    out = grouped.copy()
    out.columns = list(out.columns.map("_".join))
    return out.reset_index()


def build_binary_aggregated_table(output_dir: Path) -> pd.DataFrame:
    base = pd.read_csv(output_dir / "attacks_diagnoses.csv")
    base = base[(base["dataset"] != "mfeat-morphological") & (base["attack"] != "lpf")].copy()
    base["attack_binary"] = np.where(base["attack"] == "org", 0, 1)
    base_agg = _aggregate(
        base,
        ["dataset", "model", "attack", "n_test", "n_classes", "attack_binary"],
    )

    nn_df = pd.read_csv(output_dir / "attacks_diagnoses_nn.csv")
    nn_df["attack_binary"] = np.where(nn_df["attack"] == "org", 0, 1)
    nn_agg = _aggregate(
        nn_df,
        ["dataset", "model", "attack", "n_test", "n_classes", "attack_binary"],
    )

    out = pd.concat([base_agg, nn_agg], ignore_index=True)
    out.to_csv(output_dir / "attr_attacks_binary_agr_nn_bacc.csv", index=False)
    return out


def build_attack_type_aggregated_table(output_dir: Path) -> pd.DataFrame:
    base = pd.read_csv(output_dir / "attacks_diagnoses.csv")
    base = base[(base["dataset"] != "mfeat-morphological") & (base["attack"] != "lpf")].copy()
    base_agg = _aggregate(
        base,
        ["dataset", "model", "attack", "bacc_test", "n_test", "n_classes"],
    )

    nn_df = pd.read_csv(output_dir / "attacks_diagnoses_nn.csv")
    nn_agg = _aggregate(
        nn_df,
        ["dataset", "model", "attack", "bacc_test", "n_test", "n_classes"],
    )

    out = pd.concat([base_agg, nn_agg], ignore_index=True)
    out.to_csv(output_dir / "attr_attacks_type_agr_nn.csv", index=False)
    return out


def false_negative_rate(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape != (2, 2):
        return float("nan")
    _tn, _fp, fn, tp = cm.ravel()
    return float(fn / (fn + tp)) if (fn + tp) else float("nan")


def _iter_param_grid(grid: Dict[str, Sequence[Any]]) -> Iterable[Dict[str, Any]]:
    keys = list(grid.keys())
    for values in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, values))


def make_model(model_class: str, params: Optional[Dict[str, Any]] = None):
    params = dict(params or {})
    if model_class == "rf":
        return RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **params)
    if model_class == "xgb":
        if xgb is None:
            raise ImportError("xgboost is required for XGBClassifier")
        defaults = {
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "tree_method": "hist",
        }
        defaults.update(params)
        return xgb.XGBClassifier(**defaults)
    raise ValueError(f"Unknown model_class: {model_class}")


def _feature_columns(df: pd.DataFrame, task: str) -> List[str]:
    drop = ["dataset", "model", "attack"]
    if task == "binary":
        drop.append("attack_binary")
    cols = [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
    return cols


def _target(df: pd.DataFrame, task: str) -> pd.Series:
    if task == "binary":
        return df["attack_binary"]
    return df["attack"]


def _binary_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {
            "bacc": balanced_accuracy_score(y_true, y_pred),
            "kappa": cohen_kappa_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "fnr": false_negative_rate(y_true, y_pred),
        }


def _type_metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> Dict[str, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {
            "bacc": balanced_accuracy_score(y_true, y_pred),
            "kappa": cohen_kappa_score(y_true, y_pred),
            "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }


def _fit_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    task: str,
    model_class: str,
    params: Optional[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray, List[str], Any]:
    feature_cols = _feature_columns(train_df, task)
    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    y_train = _target(train_df, task)
    encoder = LabelEncoder()
    y_train_enc = encoder.fit_transform(y_train)
    model = make_model(model_class, params)
    model.fit(X_train, y_train_enc)
    pred_enc = model.predict(X_test)
    pred_label = encoder.inverse_transform(pred_enc.astype(int))
    return np.asarray(_target(test_df, task)), np.asarray(pred_label), feature_cols, model


def _feature_importance(model: Any, feature_cols: Sequence[str], **id_cols: Any) -> pd.DataFrame:
    importance = getattr(model, "feature_importances_", None)
    if importance is None:
        return pd.DataFrame()
    out = pd.DataFrame({"var": list(feature_cols), "fi": importance})
    out["fi_rank"] = out["fi"].rank(ascending=False)
    for key, value in id_cols.items():
        out[key] = value
    return out


def _record_predictions(
    test_df: pd.DataFrame,
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    scenario: str,
    split_value: Any,
    model_class: str,
) -> pd.DataFrame:
    keep = [c for c in ["dataset", "model", "attack", "attack_binary"] if c in test_df.columns]
    out = test_df[keep].copy()
    out["true"] = list(y_true)
    out["pred"] = list(y_pred)
    out["scenario"] = scenario
    out["split"] = split_value
    out["model_class"] = model_class.upper()
    return out


def _adaptive_stratified_folds(y: Sequence[Any], max_splits: int = 10) -> int:
    counts = pd.Series(y).value_counts()
    if counts.empty:
        return 0
    return int(max(0, min(max_splits, counts.min())))


def _evaluate_leave_group(
    df: pd.DataFrame,
    task: str,
    model_class: str,
    params: Optional[Dict[str, Any]],
    group_col: str,
    scenario: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    fi_frames = []
    pred_frames = []
    for value in sorted(df[group_col].unique()):
        train = df[df[group_col] != value]
        test = df[df[group_col] == value]
        y_true, y_pred, feature_cols, model = _fit_predict(train, test, task, model_class, params)
        metrics = _binary_metrics(y_true, y_pred) if task == "binary" else _type_metrics(y_true, y_pred)
        metric_rows.append({group_col: value, **metrics})
        fi_key = "heldout_model" if group_col == "model" else group_col
        fi_frames.append(_feature_importance(model, feature_cols, **{fi_key: value}))
        pred_frames.append(_record_predictions(test, y_true, y_pred, scenario, value, model_class))
    return (
        pd.DataFrame(metric_rows),
        pd.concat(fi_frames, ignore_index=True) if fi_frames else pd.DataFrame(),
        pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame(),
    )


def _evaluate_cross_validation(
    df: pd.DataFrame,
    task: str,
    model_class: str,
    params: Optional[Dict[str, Any]],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_cols = _feature_columns(df, task)
    y = _target(df, task)
    n_splits = _adaptive_stratified_folds(y)
    if n_splits < 2:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    metric_rows = []
    fi_frames = []
    pred_frames = []

    for cv_idx, (train_idx, test_idx) in enumerate(skf.split(df[feature_cols], y_enc)):
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]
        y_true, y_pred, fold_features, model = _fit_predict(train, test, task, model_class, params)
        metrics = _binary_metrics(y_true, y_pred) if task == "binary" else _type_metrics(y_true, y_pred)
        metric_rows.append({"cv": cv_idx, **metrics})
        fi_frames.append(_feature_importance(model, fold_features, cv=cv_idx))
        pred_frames.append(
            _record_predictions(test, y_true, y_pred, "10-fold cross-validation", cv_idx, model_class)
        )

    return pd.DataFrame(metric_rows), pd.concat(fi_frames, ignore_index=True), pd.concat(pred_frames, ignore_index=True)


def _evaluate_leave_attack_binary(
    df: pd.DataFrame,
    model_class: str,
    params: Optional[Dict[str, Any]],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    fi_frames = []
    pred_frames = []
    attacks = sorted([a for a in df["attack"].unique() if a != "org"])
    for attack in attacks:
        train = df[df["attack"] != attack]
        test = df[df["attack"] == attack]
        y_true, y_pred, feature_cols, model = _fit_predict(train, test, "binary", model_class, params)
        metric_rows.append({"attack": attack, **_binary_metrics(y_true, y_pred)})
        fi_frames.append(_feature_importance(model, feature_cols, attack=attack))
        pred_frames.append(_record_predictions(test, y_true, y_pred, "one-attack-out", attack, model_class))
    return pd.DataFrame(metric_rows), pd.concat(fi_frames, ignore_index=True), pd.concat(pred_frames, ignore_index=True)


def _add_output_labels(df: pd.DataFrame, scenario: str, model_class: str) -> pd.DataFrame:
    out = df.copy()
    out["scenario"] = scenario
    out["model_class"] = model_class.upper()
    return out


def tune_on_leave_dataset(
    df: pd.DataFrame,
    task: str,
    model_class: str,
    grid: Dict[str, Sequence[Any]],
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    rows = []
    best_params: Dict[str, Any] = {}
    best_score = -np.inf
    for params in _iter_param_grid(grid):
        metrics, _, _ = _evaluate_leave_group(
            df,
            task=task,
            model_class=model_class,
            params=params,
            group_col="dataset",
            scenario="one-data-set-out",
        )
        score = float(metrics["kappa"].mean()) if not metrics.empty else -np.inf
        rows.append({**params, "mean_kappa": score})
        if score > best_score:
            best_score = score
            best_params = params
    return best_params, pd.DataFrame(rows)


def evaluate_binary(
    df: pd.DataFrame,
    model_class: str,
    params: Optional[Dict[str, Any]],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_frames = []
    fi_frames = []
    pred_frames = []

    metrics, fi, preds = _evaluate_leave_group(
        df, "binary", model_class, params, "dataset", "one-data-set-out"
    )
    metric_frames.append(_add_output_labels(metrics, "one-data-set-out", model_class))
    fi_frames.append(_add_output_labels(fi, "one-data-set-out", model_class))
    pred_frames.append(preds)

    metrics, fi, preds = _evaluate_leave_group(
        df, "binary", model_class, params, "model", "one-model-out"
    )
    metric_frames.append(_add_output_labels(metrics, "one-model-out", model_class))
    fi_frames.append(_add_output_labels(fi, "one-model-out", model_class))
    pred_frames.append(preds)

    metrics, fi, preds = _evaluate_cross_validation(df, "binary", model_class, params)
    metric_frames.append(_add_output_labels(metrics, "10-fold cross-validation", model_class))
    fi_frames.append(_add_output_labels(fi, "10-fold cross-validation", model_class))
    pred_frames.append(preds)

    metrics, fi, preds = _evaluate_leave_attack_binary(df, model_class, params)
    metric_frames.append(_add_output_labels(metrics, "one-attack-out", model_class))
    fi_frames.append(_add_output_labels(fi, "one-attack-out", model_class))
    pred_frames.append(preds)

    return (
        pd.concat(metric_frames, ignore_index=True),
        pd.concat(fi_frames, ignore_index=True),
        pd.concat(pred_frames, ignore_index=True),
    )


def evaluate_attack_type(
    df: pd.DataFrame,
    model_class: str,
    params: Optional[Dict[str, Any]],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_frames = []
    fi_frames = []
    pred_frames = []

    metrics, fi, preds = _evaluate_leave_group(
        df, "type", model_class, params, "dataset", "one-data-set-out"
    )
    metric_frames.append(_add_output_labels(metrics, "one-data-set-out", model_class))
    fi_frames.append(_add_output_labels(fi, "one-data-set-out", model_class))
    pred_frames.append(preds)

    metrics, fi, preds = _evaluate_leave_group(
        df, "type", model_class, params, "model", "one-model-out"
    )
    metric_frames.append(_add_output_labels(metrics, "one-model-out", model_class))
    fi_frames.append(_add_output_labels(fi, "one-model-out", model_class))
    pred_frames.append(preds)

    metrics, fi, preds = _evaluate_cross_validation(df, "type", model_class, params)
    metric_frames.append(_add_output_labels(metrics, "10-fold cross-validation", model_class))
    fi_frames.append(_add_output_labels(fi, "10-fold cross-validation", model_class))
    pred_frames.append(preds)

    return (
        pd.concat(metric_frames, ignore_index=True),
        pd.concat(fi_frames, ignore_index=True),
        pd.concat(pred_frames, ignore_index=True),
    )


def save_confusion_matrices(predictions: pd.DataFrame, labels: Sequence[Any], out_path: Path) -> pd.DataFrame:
    rows = []
    for (scenario, model_class), group in predictions.groupby(["scenario", "model_class"]):
        cm = confusion_matrix(group["true"], group["pred"], labels=list(labels))
        for i, true_label in enumerate(labels):
            for j, pred_label in enumerate(labels):
                rows.append(
                    {
                        "scenario": scenario,
                        "model_class": model_class,
                        "true": true_label,
                        "pred": pred_label,
                        "count": int(cm[i, j]),
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    return out


def fit_and_save_final_model(
    df: pd.DataFrame,
    task: str,
    model_class: str,
    params: Dict[str, Any],
    output_dir: Path,
    artifact_name: str,
) -> Dict[str, Path]:
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    feature_cols = _feature_columns(df, task)
    X = df[feature_cols]
    y = _target(df, task)
    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)
    model = make_model(model_class, params)
    model.fit(X, y_enc)

    model_path = model_dir / f"{artifact_name}.joblib"
    encoder_path = model_dir / f"{artifact_name}_label_encoder.joblib"
    features_path = model_dir / f"{artifact_name}_features.json"
    config_path = model_dir / f"{artifact_name}_config.json"

    joblib.dump(model, model_path)
    joblib.dump(encoder, encoder_path)
    features_path.write_text(json.dumps(feature_cols, indent=2), encoding="utf-8")
    config_path.write_text(
        json.dumps(
            {
                "task": task,
                "model_class": model_class,
                "params": params,
                "drop_columns": DIAGNOSTIC_DROP_COLUMNS,
                "classes": list(map(str, encoder.classes_)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "model": model_path,
        "label_encoder": encoder_path,
        "features": features_path,
        "config": config_path,
    }


def run_binary_experiment(
    repo_root: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    recompute_diagnostics: bool = True,
    tune: bool = True,
) -> Dict[str, Any]:
    repo_root = Path(repo_root or find_repo_root()).resolve()
    output_dir = Path(output_dir or default_results_dir(repo_root)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    generate_diagnostic_csvs(repo_root, output_dir, recompute=recompute_diagnostics)
    agg = build_binary_aggregated_table(output_dir)

    best_params: Dict[str, Dict[str, Any]] = {}
    tuning_frames = []
    metric_frames = []
    fi_frames = []
    pred_frames = []
    for model_class, grid in [("xgb", XGB_PARAM_GRID), ("rf", RF_PARAM_GRID)]:
        params, tuning = tune_on_leave_dataset(agg, "binary", model_class, grid) if tune else ({}, pd.DataFrame())
        best_params[model_class] = params
        if not tuning.empty:
            tuning["model_class"] = model_class.upper()
            tuning_frames.append(tuning)
        metrics, fi, preds = evaluate_binary(agg, model_class, params)
        metric_frames.append(metrics)
        fi_frames.append(fi)
        pred_frames.append(preds)

    metrics_all = pd.concat(metric_frames, ignore_index=True)
    fi_all = pd.concat(fi_frames, ignore_index=True)
    preds_all = pd.concat(pred_frames, ignore_index=True)
    metrics_all.to_csv(output_dir / "detection_bacc_with_bacc.csv", index=False)
    fi_all.to_csv(output_dir / "detection_fi_with_bacc.csv", index=False)
    preds_all.to_csv(output_dir / "detection_predictions_with_bacc.csv", index=False)
    save_confusion_matrices(preds_all, [0, 1], output_dir / "detection_confusion_matrices_with_bacc.csv")
    if tuning_frames:
        pd.concat(tuning_frames, ignore_index=True).to_csv(output_dir / "binary_hyperparameter_search.csv", index=False)
    (output_dir / "binary_best_params.json").write_text(json.dumps(best_params, indent=2), encoding="utf-8")

    artifacts = {
        "binary_rf": fit_and_save_final_model(agg, "binary", "rf", best_params["rf"], output_dir, "binary_rf"),
        "binary_xgb": fit_and_save_final_model(agg, "binary", "xgb", best_params["xgb"], output_dir, "binary_xgb"),
    }
    return {
        "aggregated_table": output_dir / "attr_attacks_binary_agr_nn_bacc.csv",
        "metrics": output_dir / "detection_bacc_with_bacc.csv",
        "feature_importance": output_dir / "detection_fi_with_bacc.csv",
        "predictions": output_dir / "detection_predictions_with_bacc.csv",
        "best_params": best_params,
        "artifacts": artifacts,
    }


def run_attack_type_experiment(
    repo_root: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    recompute_diagnostics: bool = False,
    tune: bool = True,
) -> Dict[str, Any]:
    repo_root = Path(repo_root or find_repo_root()).resolve()
    output_dir = Path(output_dir or default_results_dir(repo_root)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    generate_diagnostic_csvs(repo_root, output_dir, recompute=recompute_diagnostics)
    agg = build_attack_type_aggregated_table(output_dir)

    best_params: Dict[str, Dict[str, Any]] = {}
    tuning_frames = []
    metric_frames = []
    fi_frames = []
    pred_frames = []
    for model_class, grid in [("xgb", XGB_PARAM_GRID), ("rf", RF_PARAM_GRID)]:
        params, tuning = tune_on_leave_dataset(agg, "type", model_class, grid) if tune else ({}, pd.DataFrame())
        best_params[model_class] = params
        if not tuning.empty:
            tuning["model_class"] = model_class.upper()
            tuning_frames.append(tuning)
        metrics, fi, preds = evaluate_attack_type(agg, model_class, params)
        metric_frames.append(metrics)
        fi_frames.append(fi)
        pred_frames.append(preds)

    metrics_all = pd.concat(metric_frames, ignore_index=True)
    fi_all = pd.concat(fi_frames, ignore_index=True)
    preds_all = pd.concat(pred_frames, ignore_index=True)
    labels = sorted(agg["attack"].unique())
    metrics_all.to_csv(output_dir / "isolation_bacc_nn.csv", index=False)
    fi_all.to_csv(output_dir / "isolation_fi_nn.csv", index=False)
    preds_all.to_csv(output_dir / "isolation_predictions_nn.csv", index=False)
    save_confusion_matrices(preds_all, labels, output_dir / "isolation_confusion_matrices_nn.csv")
    if tuning_frames:
        pd.concat(tuning_frames, ignore_index=True).to_csv(output_dir / "attack_type_hyperparameter_search.csv", index=False)
    (output_dir / "attack_type_best_params.json").write_text(json.dumps(best_params, indent=2), encoding="utf-8")

    artifacts = {
        "attack_type_rf": fit_and_save_final_model(agg, "type", "rf", best_params["rf"], output_dir, "attack_type_rf"),
        "attack_type_xgb": fit_and_save_final_model(agg, "type", "xgb", best_params["xgb"], output_dir, "attack_type_xgb"),
    }
    return {
        "aggregated_table": output_dir / "attr_attacks_type_agr_nn.csv",
        "metrics": output_dir / "isolation_bacc_nn.csv",
        "feature_importance": output_dir / "isolation_fi_nn.csv",
        "predictions": output_dir / "isolation_predictions_nn.csv",
        "best_params": best_params,
        "artifacts": artifacts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate diagnostic-style attack meta-tables and train RF/XGBoost "
            "binary and attack-type detectors."
        )
    )
    parser.add_argument(
        "--task",
        choices=["both", "binary", "type", "diagnostics"],
        default="both",
        help="Which pipeline to run.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help=(
            "Repository/data root containing data/adversarial_upstream, data/raw, "
            "and models. If omitted, the script searches common local and Kaggle paths."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for CSVs, metrics, and saved models. Defaults to /kaggle/working/results on Kaggle.",
    )
    parser.add_argument(
        "--no-recompute-diagnostics",
        action="store_true",
        help="Reuse existing attacks_diagnoses CSVs when present.",
    )
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Skip RF/XGB grid search and train with default parameters. Useful for quick smoke runs.",
    )
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[setup] ignoring notebook/kernel arguments: {unknown}")
    return args


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve() if args.repo_root else find_repo_root()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else default_results_dir(repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    recompute = not args.no_recompute_diagnostics
    tune = not args.no_tune

    print(f"[setup] repo_root={repo_root}")
    print(f"[setup] output_dir={output_dir}")
    print(f"[setup] task={args.task} recompute_diagnostics={recompute} tune={tune}")

    if args.task == "diagnostics":
        base_path, nn_path = generate_diagnostic_csvs(repo_root, output_dir, recompute=recompute)
        print(f"[done] wrote {base_path}")
        print(f"[done] wrote {nn_path}")
        return

    if args.task in {"both", "binary"}:
        print("[binary] running binary attack detector pipeline")
        result = run_binary_experiment(
            repo_root=repo_root,
            output_dir=output_dir,
            recompute_diagnostics=recompute,
            tune=tune,
        )
        print(f"[binary] metrics={result['metrics']}")
        for name, artifacts in result["artifacts"].items():
            print(f"[binary] {name}")
            for kind, path in artifacts.items():
                print(f"  {kind}: {path}")

    if args.task in {"both", "type"}:
        print("[type] running attack-type classifier pipeline")
        result = run_attack_type_experiment(
            repo_root=repo_root,
            output_dir=output_dir,
            recompute_diagnostics=(recompute if args.task == "type" else False),
            tune=tune,
        )
        print(f"[type] metrics={result['metrics']}")
        for name, artifacts in result["artifacts"].items():
            print(f"[type] {name}")
            for kind, path in artifacts.items():
                print(f"  {kind}: {path}")


if __name__ == "__main__":
    main()
