import numpy as np
import pandas as pd
import tempfile
from pathlib import Path

from src.generate_adv_samples import build_adv_dataframe, save_adv_csv


def test_build_adv_dataframe():
    X_adv = np.random.randn(10, 4)
    original_labels = np.array([0] * 5 + [1] * 5)
    clean_preds = np.array([0] * 5 + [1] * 5)
    adv_preds = np.array([1] * 5 + [0] * 5)
    feat_names = [f"feat_{i}" for i in range(4)]
    df = build_adv_dataframe(X_adv, original_labels, clean_preds, adv_preds, feat_names)
    assert len(df) == 10
    assert "sample_id" in df.columns
    assert "attack_success" in df.columns
    assert df["attack_success"].sum() == 10


def test_save_adv_csv():
    with tempfile.TemporaryDirectory() as tmpdir:
        df = pd.DataFrame({
            "sample_id": [0, 1],
            "original_label": [0, 1],
            "clean_prediction": [0, 1],
            "adv_prediction": [1, 0],
            "attack_success": [True, True],
            "feat_0": [0.1, 0.2],
        })
        path = Path(tmpdir) / "test_adv.csv"
        save_adv_csv(df, path)
        loaded = pd.read_csv(path)
        assert len(loaded) == 2
        assert "sample_id" in loaded.columns
