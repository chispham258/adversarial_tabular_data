import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_baseline_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }


def compute_attack_metrics(
    X_clean: np.ndarray,
    X_adv: np.ndarray,
    y_true: np.ndarray,
    clean_preds: np.ndarray,
    adv_preds: np.ndarray,
) -> dict:
    clean_acc = accuracy_score(y_true, clean_preds)
    adv_acc = accuracy_score(y_true, adv_preds)
    acc_drop = clean_acc - adv_acc

    correct_mask = clean_preds == y_true
    if correct_mask.sum() == 0:
        asr = 0.0
    else:
        asr = float(((correct_mask) & (adv_preds != y_true)).sum() / correct_mask.sum())

    diff = X_adv - X_clean
    l0_mean = float(np.mean(np.sum(np.abs(diff) > 1e-6, axis=1)))
    l2_mean = float(np.mean(np.linalg.norm(diff, axis=1)))
    linf_mean = float(np.mean(np.max(np.abs(diff), axis=1)))

    return {
        "clean_acc": clean_acc,
        "adv_acc": adv_acc,
        "acc_drop": acc_drop,
        "asr": asr,
        "l0_mean": l0_mean,
        "l2_mean": l2_mean,
        "linf_mean": linf_mean,
    }
