import numpy as np
import pandas as pd
import torch

from src.upstream_attacks import (
    TabDataModel2,
    build_upstream_adv_dataframe,
    prepare_upstream_data,
)


def test_prepare_upstream_data_uses_existing_train_split():
    df = pd.DataFrame({
        "f0": [0.0, 1.0, 2.0, 3.0],
        "f1": [10.0, 11.0, 12.0, 13.0],
        "name": [0, 1, 2, 3],
        "prediction": [0, 0, 1, 1],
        "is_train": [1, 1, 0, 0],
        "target": [0, 1, 0, 1],
    })

    data = prepare_upstream_data(df, target_col="target")

    assert data.X_train.shape == (2, 2)
    assert data.X_test.shape == (2, 2)
    assert data.feature_names == ["f0", "f1"]
    assert data.X_train.min() >= 0.0
    assert data.X_train.max() <= 1.0


def test_tab_data_model2_shape():
    model = TabDataModel2(input_dim=5, num_classes=3)
    out = model(torch.randn(4, 5))
    assert out.shape == (4, 3)


def test_build_upstream_adv_dataframe_contains_scores():
    X_adv = np.array([[0.1, 0.2], [0.3, 0.4]])
    y_true = np.array([0, 1])
    y_pred = np.array([1, 1])
    probs = np.array([[0.25, 0.75], [0.1, 0.9]])

    df = build_upstream_adv_dataframe(
        X_adv=X_adv,
        y_true=y_true,
        y_pred=y_pred,
        y_score=probs,
        feature_names=["f0", "f1"],
    )

    assert list(df.columns) == [
        "f0",
        "f1",
        "name",
        "is_train",
        "target",
        "prediction",
        "score_0",
        "score_1",
    ]
    assert df["prediction"].tolist() == [1, 1]
