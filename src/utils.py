import random
from pathlib import Path

import numpy as np
import requests
import torch

BASE = Path(__file__).parent.parent

DATASET_REGISTRY = {
    "wilt": {
        "url": "https://raw.githubusercontent.com/lwawrowski/adversarial_attacks_detection/master/data/wilt.csv",
        "target": "target",
        "drop_cols": ["name", "is_train", "prediction"],
    },
    "banknote": {
        "url": "https://huggingface.co/datasets/farish07/banknote-authentication-dataset/resolve/main/data_banknote_authentication.csv",
        "target": "Class",
        "column_names": ["variance", "skewness", "curtosis", "entropy", "Class"],
    },
    "diabetes": {
        "url": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
        "target": "Outcome",
        "column_names": [
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age",
            "Outcome",
        ],
    },
}


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_paths(dataset: str) -> dict:
    processed_dir = BASE / "data" / "processed" / dataset
    return {
        "raw": BASE / "data" / "raw" / f"{dataset}.csv",
        "processed_dir": processed_dir,
        "model_dir": BASE / "models" / dataset,
        "adv_dir": BASE / "data" / "adversarial" / dataset,
        "X_train": processed_dir / "X_train.csv",
        "X_test": processed_dir / "X_test.csv",
        "y_train": processed_dir / "y_train.csv",
        "y_test": processed_dir / "y_test.csv",
        "scaler": processed_dir / "scaler.pkl",
    }


def download_dataset(dataset: str) -> Path:
    info = DATASET_REGISTRY[dataset]
    raw_path = get_paths(dataset)["raw"]
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if raw_path.exists():
        print(f"[utils] {dataset} already at {raw_path}")
        return raw_path
    print(f"[utils] Downloading {dataset} ...")
    response = requests.get(info["url"], timeout=30)
    response.raise_for_status()
    raw_path.write_bytes(response.content)
    print(f"[utils] Saved to {raw_path}")
    return raw_path


def ensure_dirs(dataset: str) -> None:
    paths = get_paths(dataset)
    for key in ["processed_dir", "model_dir", "adv_dir"]:
        paths[key].mkdir(parents=True, exist_ok=True)
    (BASE / "results").mkdir(parents=True, exist_ok=True)
