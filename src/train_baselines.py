import argparse
import csv
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

from src.metrics import compute_baseline_metrics
from src.utils import ensure_dirs, get_paths, set_seed

BASELINE_RESULTS = Path(__file__).parent.parent / "results" / "baseline_results.csv"


class MLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_logistic_regression(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray) -> XGBClassifier:
    model = XGBClassifier(random_state=42, eval_metric="logloss", verbosity=0)
    model.fit(X_train, y_train)
    return model


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    device: torch.device,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> MLP:
    num_classes = len(np.unique(y_train))
    model = MLP(input_dim=X_train.shape[1], num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1}/{epochs}  loss={total_loss/len(loader):.4f}")
    return model


def _append_baseline_result(row: dict) -> None:
    BASELINE_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    write_header = not BASELINE_RESULTS.exists()
    with open(BASELINE_RESULTS, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def evaluate_and_save(
    model_name: str,
    model,
    dataset: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device | None = None,
) -> dict:
    if model_name == "mlp":
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(X_test, dtype=torch.float32).to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        y_pred = probs.argmax(axis=1)
        y_prob = probs[:, 1]
    else:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
    m = compute_baseline_metrics(y_test, y_pred, y_prob)
    row = {"dataset": dataset, "model": model_name, **m}
    _append_baseline_result(row)
    return row


def train_all(dataset: str, model_names: list[str]) -> None:
    set_seed(42)
    paths = get_paths(dataset)
    ensure_dirs(dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}")

    X_train = pd.read_csv(paths["X_train"]).values.astype(np.float32)
    X_test = pd.read_csv(paths["X_test"]).values.astype(np.float32)
    y_train = pd.read_csv(paths["y_train"]).values.ravel().astype(np.int64)
    y_test = pd.read_csv(paths["y_test"]).values.ravel().astype(np.int64)

    for name in model_names:
        print(f"\n[train] Training {name} on {dataset} ...")
        if name == "logistic_regression":
            model = train_logistic_regression(X_train, y_train)
            joblib.dump(model, paths["model_dir"] / "logistic_regression.pkl")
        elif name == "xgboost":
            model = train_xgboost(X_train, y_train)
            joblib.dump(model, paths["model_dir"] / "xgboost.pkl")
        elif name == "mlp":
            model = train_mlp(X_train, y_train, device=device)
            torch.save(model, paths["model_dir"] / "mlp.pt")
            print("[train] Training surrogate MLP ...")
            set_seed(0)
            surrogate = train_mlp(X_train, y_train, device=device)
            torch.save(surrogate, paths["model_dir"] / "surrogate_mlp.pt")
        else:
            raise ValueError(f"Unknown model: {name}")

        m = evaluate_and_save(
            name, model, dataset, X_test, y_test,
            device=(device if name == "mlp" else None),
        )
        print(f"[train] {name}  acc={m['accuracy']:.4f}  f1={m['f1']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--models", nargs="+",
        default=["logistic_regression", "xgboost", "mlp"],
    )
    args = parser.parse_args()
    train_all(args.dataset, args.models)
