import argparse
import csv
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from src.attacks import apply_constraints, run_fgsm, run_pgd, wrap_pytorch, wrap_sklearn
from src.metrics import compute_attack_metrics
from src.train_baselines import MLP
from src.utils import ensure_dirs, get_paths, set_seed

ATTACK_RESULTS = Path(__file__).parent.parent / "results" / "attack_results.csv"
ATTACK_FIELDNAMES = [
    "dataset", "target_model", "attack", "epsilon", "steps",
    "clean_acc", "adv_acc", "acc_drop", "asr",
    "l0_mean", "l2_mean", "linf_mean",
]


def build_adv_dataframe(
    X_adv: np.ndarray,
    original_labels: np.ndarray,
    clean_preds: np.ndarray,
    adv_preds: np.ndarray,
    feat_names: list,
) -> pd.DataFrame:
    df = pd.DataFrame(X_adv, columns=feat_names)
    df.insert(0, "sample_id", np.arange(len(X_adv)))
    df.insert(1, "original_label", original_labels)
    df.insert(2, "clean_prediction", clean_preds)
    df.insert(3, "adv_prediction", adv_preds)
    df.insert(
        4, "attack_success",
        (clean_preds == original_labels) & (adv_preds != original_labels),
    )
    return df


def save_adv_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[generate] Saved → {path}")


def _append_attack_result(row: dict) -> None:
    ATTACK_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    write_header = not ATTACK_RESULTS.exists()
    with open(ATTACK_RESULTS, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ATTACK_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in ATTACK_FIELDNAMES})


def _predict_mlp(model: MLP, X: np.ndarray, device: torch.device):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32).to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs.argmax(axis=1), probs


def run_attack_for_config(
    dataset: str,
    target_model_name: str,
    attack_name: str,
    eps: float,
    steps: int,
    paths: dict,
    device: torch.device,
) -> None:
    X_test = pd.read_csv(paths["X_test"]).values.astype(np.float32)
    X_train = pd.read_csv(paths["X_train"]).values.astype(np.float32)
    y_test = pd.read_csv(paths["y_test"]).values.ravel().astype(np.int64)
    feat_names = pd.read_csv(paths["X_test"]).columns.tolist()
    nb_classes = len(np.unique(y_test))
    dev_str = "gpu" if device.type == "cuda" else "cpu"

    if target_model_name == "mlp":
        model = torch.load(paths["model_dir"] / "mlp.pt", map_location=device)
        model.eval()
        clf = wrap_pytorch(model, input_shape=(X_test.shape[1],),
                           nb_classes=nb_classes, device=dev_str)
        clean_preds, _ = _predict_mlp(model, X_test, device)

    elif target_model_name == "logistic_regression":
        model = joblib.load(paths["model_dir"] / "logistic_regression.pkl")
        clf = wrap_sklearn(model)
        clean_preds = model.predict(X_test)

    elif target_model_name == "xgboost":
        surrogate = torch.load(paths["model_dir"] / "surrogate_mlp.pt", map_location=device)
        surrogate.eval()
        clf = wrap_pytorch(surrogate, input_shape=(X_test.shape[1],),
                           nb_classes=nb_classes, device=dev_str)
        xgb_model = joblib.load(paths["model_dir"] / "xgboost.pkl")
        clean_preds = xgb_model.predict(X_test)
    else:
        raise ValueError(f"Unknown target model: {target_model_name}")

    print(f"[generate] {dataset}/{target_model_name} {attack_name} eps={eps}")

    if attack_name == "fgsm":
        X_adv = run_fgsm(clf, X_test, eps=eps)
    elif attack_name in ("pgd", "transfer_pgd"):
        X_adv = run_pgd(clf, X_test, eps=eps, max_iter=steps, eps_step=eps / 10)
    else:
        raise ValueError(f"Unknown attack: {attack_name}")

    X_adv = apply_constraints(X_adv, X_train)

    if target_model_name == "xgboost":
        adv_preds = xgb_model.predict(X_adv)
    elif target_model_name == "mlp":
        adv_preds, _ = _predict_mlp(model, X_adv, device)
    else:
        adv_preds = model.predict(X_adv)

    m = compute_attack_metrics(X_test, X_adv, y_test, clean_preds, adv_preds)
    print(f"  clean_acc={m['clean_acc']:.4f}  adv_acc={m['adv_acc']:.4f}  "
          f"asr={m['asr']:.4f}  l2={m['l2_mean']:.4f}")

    df_adv = build_adv_dataframe(X_adv, y_test, clean_preds, adv_preds, feat_names)
    fname = f"{target_model_name}_{attack_name}_eps_{eps}.csv"
    save_adv_csv(df_adv, paths["adv_dir"] / fname)

    _append_attack_result({
        "dataset": dataset,
        "target_model": target_model_name,
        "attack": attack_name,
        "epsilon": eps,
        "steps": steps,
        **m,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--target_model", required=True,
                        choices=["mlp", "logistic_regression", "xgboost"])
    parser.add_argument("--attack", required=True,
                        choices=["fgsm", "pgd", "transfer_pgd"])
    parser.add_argument("--eps", nargs="+", type=float, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--surrogate_model", default="mlp")
    args = parser.parse_args()

    set_seed(42)
    ensure_dirs(args.dataset)
    paths = get_paths(args.dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for eps in args.eps:
        run_attack_for_config(
            dataset=args.dataset,
            target_model_name=args.target_model,
            attack_name=args.attack,
            eps=eps,
            steps=args.steps,
            paths=paths,
            device=device,
        )
