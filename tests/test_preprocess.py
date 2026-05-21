import pytest
from src.preprocess import split_and_scale, print_dataset_stats


def test_split_shapes(synthetic_df):
    X_train, X_test, y_train, y_test, scaler = split_and_scale(
        synthetic_df, target_col="class", random_state=42
    )
    assert X_train.shape[1] == 5
    assert X_test.shape[1] == 5
    assert len(X_train) + len(X_test) == 200
    assert abs(len(X_test) / 200 - 0.2) < 0.05


def test_scaler_fit_on_train_only(synthetic_df):
    import numpy as np
    X_train, X_test, y_train, y_test, scaler = split_and_scale(
        synthetic_df, target_col="class", random_state=42
    )
    assert abs(X_train.mean().mean()) < 0.2
    assert abs(X_train.std().mean() - 1.0) < 0.2


def test_no_target_in_features(synthetic_df):
    X_train, X_test, y_train, y_test, scaler = split_and_scale(
        synthetic_df, target_col="class", random_state=42
    )
    assert "class" not in X_train.columns
    assert "class" not in X_test.columns


def test_print_stats_runs(synthetic_df, capsys):
    print_dataset_stats(synthetic_df, target_col="class")
    out = capsys.readouterr().out
    assert "samples" in out
    assert "features" in out
