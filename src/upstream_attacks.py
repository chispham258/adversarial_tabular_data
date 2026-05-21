import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from art.attacks.evasion import (
    BasicIterativeMethod,
    FastGradientMethod,
    HopSkipJump,
    LowProFool,
    ProjectedGradientDescent,
    ZooAttack,
)
from art.estimators.classification import PyTorchClassifier, SklearnClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset

from src.metrics import compute_attack_metrics
from src.utils import DATASET_REGISTRY, BASE, download_dataset, get_paths, set_seed

UPSTREAM_RESULTS = BASE / "results" / "upstream_attack_results.csv"
UPSTREAM_ADV_DIR = BASE / "data" / "adversarial_upstream"
UPSTREAM_MODEL_DIR = BASE / "models"
FIELDNAMES = [
    "dataset",
    "model",
    "attack",
    "epsilon",
    "steps",
    "n_samples",
    "clean_acc",
    "adv_acc",
    "clean_bacc",
    "adv_bacc",
    "acc_drop",
    "bacc_drop",
    "asr",
    "l0_mean",
    "l2_mean",
    "linf_mean",
]


@dataclass
class UpstreamData:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    y_train_one: np.ndarray
    y_test_one: np.ndarray
    feature_names: List[str]


class TabDataModel2(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.input = nn.Linear(input_dim, 32)
        self.relu = nn.ReLU()
        self.output = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.relu(self.input(x)))


def _one_hot(y: np.ndarray) -> np.ndarray:
    y = y.astype(int)
    y_one = np.zeros((y.size, int(y.max()) + 1), dtype=np.float32)
    y_one[np.arange(y.size), y] = 1.0
    return y_one


def prepare_upstream_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> UpstreamData:
    df = df.copy()
    y = df[target_col]
    if y.dtype == object:
        df[target_col] = LabelEncoder().fit_transform(y)

    metadata_cols = [target_col, "name", "prediction", "is_train"]
    metadata_cols.extend([c for c in df.columns if c.startswith("score_")])

    if "is_train" in df.columns:
        train_df = df[df["is_train"] == 1]
        test_df = df[df["is_train"] == 0]
        X_train = train_df.drop(columns=[c for c in metadata_cols if c in train_df.columns])
        X_test = test_df.drop(columns=[c for c in metadata_cols if c in test_df.columns])
        y_train = train_df[target_col].to_numpy()
        y_test = test_df[target_col].to_numpy()
    else:
        X = df.drop(columns=[c for c in metadata_cols if c in df.columns])
        y_values = df[target_col].to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_values,
            test_size=test_size,
            random_state=random_state,
            stratify=y_values,
        )

    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test).reindex(columns=X_train.columns, fill_value=0)

    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)
    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)

    return UpstreamData(
        X_train=X_train_s,
        X_test=X_test_s,
        y_train=y_train,
        y_test=y_test,
        y_train_one=_one_hot(y_train),
        y_test_one=_one_hot(y_test),
        feature_names=list(X_train.columns),
    )


def load_upstream_data(dataset: str) -> UpstreamData:
    info = DATASET_REGISTRY[dataset]
    raw_path = download_dataset(dataset)
    df = pd.read_csv(raw_path, names=info.get("column_names"))
    drop_cols = info.get("drop_cols", [])
    if drop_cols and "is_train" not in df.columns:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return prepare_upstream_data(df, target_col=info["target"])


def train_nn_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 0.01,
) -> Tuple[TabDataModel2, PyTorchClassifier]:
    num_classes = len(np.unique(y_train))
    model = TabDataModel2(X_train.shape[1], num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

    classifier = PyTorchClassifier(
        model=model,
        loss=loss_fn,
        optimizer=optimizer,
        input_shape=(X_train.shape[1],),
        nb_classes=num_classes,
    )
    return model, classifier


def train_sklearn_classifier(model_name: str, X_train: np.ndarray, y_train: np.ndarray):
    if model_name == "lin":
        model = LogisticRegression(max_iter=1000, random_state=42)
    elif model_name == "svm":
        model = SVC(probability=True, random_state=42)
    elif model_name == "xgb":
        model = GradientBoostingClassifier(n_estimators=500, random_state=42)
    else:
        raise ValueError(f"Unknown upstream sklearn model: {model_name}")
    model.fit(X_train, y_train)
    return model, SklearnClassifier(model=model)


def save_upstream_model(
    dataset: str,
    model_name: str,
    model,
    model_root: Optional[Path] = None,
) -> Path:
    out_dir = (model_root or UPSTREAM_MODEL_DIR) / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    if model_name == "nn":
        out_path = out_dir / "upstream_nn.pt"
        torch.save(
            {
                "model": "TabDataModel2",
                "input_dim": model.input.in_features,
                "num_classes": model.output.out_features,
                "state_dict": model.state_dict(),
            },
            out_path,
        )
        return out_path

    out_path = out_dir / f"upstream_{model_name}.pkl"
    joblib.dump(model, out_path)
    return out_path


def predict_nn(classifier: PyTorchClassifier, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    scores = classifier.predict(X)
    exp = np.exp(scores - scores.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    return probs.argmax(axis=1), probs


def predict_sklearn(model, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    probs = model.predict_proba(X)
    return probs.argmax(axis=1), probs


def build_upstream_adv_dataframe(
    X_adv: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    feature_names: List[str],
) -> pd.DataFrame:
    df = pd.DataFrame(X_adv, columns=feature_names)
    df["name"] = np.arange(len(df))
    df["is_train"] = 0
    df["target"] = y_true
    df["prediction"] = y_pred
    for idx in range(y_score.shape[1]):
        df[f"score_{idx}"] = y_score[:, idx]
    return df


def _append_result(row: dict) -> None:
    UPSTREAM_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    write_header = not UPSTREAM_RESULTS.exists()
    with open(UPSTREAM_RESULTS, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in FIELDNAMES})


def _save_adv(dataset: str, model: str, attack: str, eps: Optional[float], df: pd.DataFrame) -> None:
    out_dir = UPSTREAM_ADV_DIR / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = attack if eps is None else f"{attack}_eps_{eps:.2f}"
    df.to_csv(out_dir / f"{model}_{suffix}.csv", index=False)


def _attack_metrics_row(
    dataset: str,
    model_name: str,
    attack_name: str,
    epsilon: Optional[float],
    steps: int,
    X_clean: np.ndarray,
    X_adv: np.ndarray,
    y_true: np.ndarray,
    clean_preds: np.ndarray,
    adv_preds: np.ndarray,
) -> dict:
    metrics = compute_attack_metrics(X_clean, X_adv, y_true, clean_preds, adv_preds)
    clean_bacc = balanced_accuracy_score(y_true, clean_preds)
    adv_bacc = balanced_accuracy_score(y_true, adv_preds)
    return {
        "dataset": dataset,
        "model": model_name,
        "attack": attack_name,
        "epsilon": epsilon,
        "steps": steps,
        "n_samples": len(y_true),
        "clean_bacc": clean_bacc,
        "adv_bacc": adv_bacc,
        "bacc_drop": clean_bacc - adv_bacc,
        **metrics,
    }


def make_attack(
    classifier,
    attack_name: str,
    eps: Optional[float],
    steps: int,
    zoo_max_iter: int,
    zoo_nb_parallel: int = 5,
):
    if attack_name == "fgm":
        return FastGradientMethod(estimator=classifier, eps=eps)
    if attack_name == "pgd":
        return ProjectedGradientDescent(
            estimator=classifier,
            eps=eps,
            eps_step=0.1,
            max_iter=steps,
            num_random_init=1,
            targeted=False,
            verbose=True,
        )
    if attack_name == "bim":
        return BasicIterativeMethod(
            estimator=classifier,
            eps=eps,
            eps_step=0.1,
            max_iter=steps,
            targeted=False,
            verbose=True,
        )
    if attack_name == "zoo":
        return ZooAttack(
            classifier=classifier,
            confidence=0.0,
            targeted=False,
            learning_rate=1e-1,
            max_iter=zoo_max_iter,
            binary_search_steps=10,
            initial_const=1e-3,
            abort_early=True,
            use_resize=False,
            use_importance=False,
            nb_parallel=zoo_nb_parallel,
            batch_size=1,
            variable_h=0.01,
        )
    if attack_name == "hsj":
        return HopSkipJump(classifier=classifier)
    if attack_name == "lpf":
        return LowProFool(classifier=classifier)
    raise ValueError(f"Unknown upstream attack: {attack_name}")


def run_one_attack(
    dataset: str,
    model_name: str,
    attack_name: str,
    data: UpstreamData,
    eps: Optional[float],
    steps: int,
    blackbox_sample_limit: int,
    zoo_max_iter: int,
    nn_epochs: int,
) -> Optional[dict]:
    if model_name == "nn":
        model, classifier = train_nn_classifier(data.X_train, data.y_train, epochs=nn_epochs)
        save_upstream_model(dataset, model_name, model)
        clean_preds, _ = predict_nn(classifier, data.X_test)
        predict_fn = lambda X: predict_nn(classifier, X)
    else:
        model, classifier = train_sklearn_classifier(model_name, data.X_train, data.y_train)
        save_upstream_model(dataset, model_name, model)
        clean_preds, _ = predict_sklearn(model, data.X_test)
        predict_fn = lambda X: predict_sklearn(model, X)

    X_attack = data.X_test
    y_attack = data.y_test
    clean_attack_preds = clean_preds
    y_attack_one = data.y_test_one
    if attack_name in {"zoo", "hsj", "lpf"} and blackbox_sample_limit > 0:
        X_attack = X_attack[:blackbox_sample_limit]
        y_attack = y_attack[:blackbox_sample_limit]
        clean_attack_preds = clean_attack_preds[:blackbox_sample_limit]
        y_attack_one = y_attack_one[:blackbox_sample_limit]

    attack = make_attack(
        classifier,
        attack_name,
        eps,
        steps,
        zoo_max_iter,
        zoo_nb_parallel=min(5, X_attack.shape[1]),
    )
    if attack_name == "lpf":
        attack = attack.fit_importances(x=X_attack, y=y_attack)
        X_adv = attack.generate(x=X_attack, y=y_attack_one)
    elif attack_name in {"zoo", "hsj"}:
        X_adv = attack.generate(x=X_attack, y=y_attack)
    else:
        X_adv = attack.generate(x=X_attack)

    X_adv = np.clip(np.nan_to_num(X_adv, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    adv_preds, adv_scores = predict_fn(X_adv)
    row = _attack_metrics_row(
        dataset,
        model_name,
        attack_name,
        eps,
        steps,
        X_attack,
        X_adv,
        y_attack,
        clean_attack_preds,
        adv_preds,
    )
    df_adv = build_upstream_adv_dataframe(
        X_adv,
        y_attack,
        adv_preds,
        adv_scores,
        data.feature_names,
    )
    _save_adv(dataset, model_name, attack_name, eps, df_adv)
    _append_result(row)
    return row


def run_upstream_experiment(args) -> None:
    set_seed(42)
    for dataset in args.datasets:
        data = load_upstream_data(dataset)
        for model_name in args.models:
            attack_names = args.nn_attacks if model_name == "nn" else args.sklearn_attacks
            for attack_name in attack_names:
                eps_values = args.eps if attack_name in {"fgm", "pgd", "bim"} else [None]
                for eps in eps_values:
                    print(f"[upstream] {dataset}/{model_name}/{attack_name} eps={eps}")
                    try:
                        row = run_one_attack(
                            dataset=dataset,
                            model_name=model_name,
                            attack_name=attack_name,
                            data=data,
                            eps=eps,
                            steps=args.steps,
                            blackbox_sample_limit=args.blackbox_sample_limit,
                            zoo_max_iter=args.zoo_max_iter,
                            nn_epochs=args.nn_epochs,
                        )
                        print(
                            f"  clean_acc={row['clean_acc']:.4f} adv_acc={row['adv_acc']:.4f} "
                            f"clean_bacc={row['clean_bacc']:.4f} adv_bacc={row['adv_bacc']:.4f} "
                            f"asr={row['asr']:.4f}"
                        )
                    except Exception as exc:
                        print(f"  skipped: {type(exc).__name__}: {exc}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["wilt", "banknote", "diabetes"])
    parser.add_argument("--models", nargs="+", default=["nn", "lin", "svm", "xgb"])
    parser.add_argument("--nn-attacks", nargs="+", default=["fgm", "pgd", "bim"])
    parser.add_argument("--sklearn-attacks", nargs="+", default=["zoo", "hsj", "lpf"])
    parser.add_argument("--eps", nargs="+", type=float, default=[0.0, 0.1, 0.2])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--nn-epochs", type=int, default=200)
    parser.add_argument("--zoo-max-iter", type=int, default=200)
    parser.add_argument("--blackbox-sample-limit", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    run_upstream_experiment(parse_args())
