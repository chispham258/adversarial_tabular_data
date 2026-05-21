import argparse
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.utils import DATASET_REGISTRY, download_dataset, ensure_dirs, get_paths, set_seed


def print_dataset_stats(df: pd.DataFrame, target_col: str) -> None:
    print(f"  samples    : {len(df)}")
    print(f"  features   : {df.shape[1] - 1}")
    print(f"  class dist : {df[target_col].value_counts().to_dict()}")
    print(f"  missing    : {df.isnull().sum().sum()}")


def split_and_scale(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    if y.dtype == object:
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), name=target_col)
    else:
        le = None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # apply dummies after split, align test to train columns
    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    scaler = StandardScaler()
    X_train_s = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_s = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )
    return X_train_s, X_test_s, y_train, y_test, scaler, le


def preprocess_dataset(dataset: str, output_dir: Optional[Path] = None) -> None:
    set_seed(42)
    info = DATASET_REGISTRY[dataset]
    raw_path = download_dataset(dataset)
    df = pd.read_csv(raw_path)

    print(f"\n[preprocess] Dataset: {dataset}")
    print_dataset_stats(df, target_col=info["target"])

    X_train, X_test, y_train, y_test, scaler, le = split_and_scale(
        df, target_col=info["target"]
    )
    ensure_dirs(dataset)
    paths = get_paths(dataset)
    out = Path(output_dir) if output_dir else paths["processed_dir"]
    out.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(out / "X_train.csv", index=False)
    X_test.to_csv(out / "X_test.csv", index=False)
    y_train.to_csv(out / "y_train.csv", index=False)
    y_test.to_csv(out / "y_test.csv", index=False)
    joblib.dump(scaler, out / "scaler.pkl")
    if le is not None:
        joblib.dump(le, out / "label_encoder.pkl")
    print(f"[preprocess] Saved splits to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    preprocess_dataset(args.dataset, output_dir=args.output)
