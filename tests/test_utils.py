import numpy as np
import pytest
from src.utils import DATASET_REGISTRY, get_paths, set_seed


def test_registry_has_wilt():
    assert "wilt" in DATASET_REGISTRY
    assert "url" in DATASET_REGISTRY["wilt"]
    assert "target" in DATASET_REGISTRY["wilt"]


def test_get_paths_returns_all_keys():
    paths = get_paths("wilt")
    for key in ["raw", "processed_dir", "model_dir", "adv_dir",
                "X_train", "X_test", "y_train", "y_test", "scaler"]:
        assert key in paths


def test_set_seed_reproducible():
    set_seed(42)
    a = np.random.randn(5)
    set_seed(42)
    b = np.random.randn(5)
    assert np.allclose(a, b)
