import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.attacks import apply_constraints, run_fgsm, run_pgd, wrap_pytorch, wrap_sklearn


def test_wrap_sklearn():
    X = np.random.randn(50, 5).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    lr = LogisticRegression().fit(X, y)
    clf = wrap_sklearn(lr, nb_classes=2)
    assert clf is not None


def test_fgsm_output_shape(tiny_mlp):
    X = np.random.randn(10, 5).astype(np.float32)
    clf = wrap_pytorch(tiny_mlp, input_shape=(5,), nb_classes=2)
    X_adv = run_fgsm(clf, X, eps=0.1)
    assert X_adv.shape == X.shape


def test_pgd_output_shape(tiny_mlp):
    X = np.random.randn(10, 5).astype(np.float32)
    clf = wrap_pytorch(tiny_mlp, input_shape=(5,), nb_classes=2)
    X_adv = run_pgd(clf, X, eps=0.1, max_iter=5, eps_step=0.01)
    assert X_adv.shape == X.shape


def test_apply_constraints():
    X_train = np.array([[0.0, 0.0], [1.0, 1.0]])
    X_adv = np.array([[2.0, -1.0], [0.5, float("nan")]])
    result = apply_constraints(X_adv, X_train)
    assert result[0, 0] == pytest.approx(1.0)
    assert result[0, 1] == pytest.approx(0.0)
    assert not np.isnan(result).any()


def test_fgsm_perturbation_bounded(tiny_mlp):
    X = np.random.randn(20, 5).astype(np.float32)
    clf = wrap_pytorch(tiny_mlp, input_shape=(5,), nb_classes=2)
    eps = 0.1
    X_adv = run_fgsm(clf, X, eps=eps)
    assert np.abs(X_adv - X).max() <= eps + 1e-5
