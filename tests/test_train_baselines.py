import numpy as np
import pytest
import torch
from src.train_baselines import MLP, train_logistic_regression, train_xgboost, train_mlp


def test_logistic_regression_smoke(synthetic_splits):
    X_train, X_test, y_train, y_test, scaler, le = synthetic_splits
    model = train_logistic_regression(X_train.values, y_train.values)
    preds = model.predict(X_test.values)
    assert len(preds) == len(X_test)


def test_xgboost_smoke(synthetic_splits):
    X_train, X_test, y_train, y_test, scaler, le = synthetic_splits
    model = train_xgboost(X_train.values, y_train.values)
    preds = model.predict(X_test.values)
    assert len(preds) == len(X_test)


def test_mlp_smoke(synthetic_splits):
    X_train, X_test, y_train, y_test, scaler, le = synthetic_splits
    device = torch.device("cpu")
    model = train_mlp(X_train.values, y_train.values, device=device, epochs=2)
    assert isinstance(model, MLP)
    model.eval()
    with torch.no_grad():
        x = torch.tensor(X_test.values[:4], dtype=torch.float32)
        out = model(x)
    assert out.shape == (4, 2)


def test_mlp_architecture():
    model = MLP(input_dim=5, num_classes=2)
    x = torch.randn(3, 5)
    out = model(x)
    assert out.shape == (3, 2)
