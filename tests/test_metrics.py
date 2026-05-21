import numpy as np
import pytest
from src.metrics import compute_baseline_metrics, compute_attack_metrics


def test_baseline_metrics_perfect():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.1, 0.9, 0.9])
    m = compute_baseline_metrics(y_true, y_pred, y_prob)
    assert m["accuracy"] == 1.0
    assert m["f1"] == 1.0
    assert m["roc_auc"] == 1.0


def test_baseline_metrics_keys():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0])
    y_prob = np.array([0.2, 0.8, 0.6, 0.4])
    m = compute_baseline_metrics(y_true, y_pred, y_prob)
    for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "balanced_accuracy"]:
        assert k in m


def test_attack_metrics_full_success():
    X_clean = np.zeros((4, 3))
    X_adv = np.ones((4, 3))
    y_true = np.array([0, 0, 1, 1])
    clean_preds = np.array([0, 0, 1, 1])
    adv_preds = np.array([1, 1, 0, 0])
    m = compute_attack_metrics(X_clean, X_adv, y_true, clean_preds, adv_preds)
    assert m["asr"] == pytest.approx(1.0)
    assert m["acc_drop"] > 0
    assert m["linf_mean"] == pytest.approx(1.0)


def test_attack_metrics_l0():
    X_clean = np.zeros((2, 5))
    X_adv = np.array([[1, 0, 0, 0, 1], [0, 1, 1, 0, 0]], dtype=float)
    y_true = np.array([0, 1])
    clean_preds = np.array([0, 1])
    adv_preds = np.array([1, 0])
    m = compute_attack_metrics(X_clean, X_adv, y_true, clean_preds, adv_preds)
    assert m["l0_mean"] == pytest.approx(2.0)
