import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn


@pytest.fixture
def synthetic_df():
    np.random.seed(42)
    n = 200
    X = np.random.randn(n, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(5)])
    df["class"] = y
    return df


@pytest.fixture
def synthetic_splits(synthetic_df):
    from src.preprocess import split_and_scale
    return split_and_scale(synthetic_df, target_col="class", random_state=42)


@pytest.fixture
def tiny_mlp():
    class TinyMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(5, 8), nn.ReLU(), nn.Linear(8, 2)
            )

        def forward(self, x):
            return self.net(x)

    model = TinyMLP()
    model.eval()
    return model
