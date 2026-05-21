import json
from pathlib import Path


INPUT_DATA_DIR = "/kaggle/input/datasets/hongthnguyn/attack-data/data"


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip().splitlines(True),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def standalone_helper_code() -> str:
    source = (Path(__file__).resolve().parent / "rf_xgb_detection.py").read_text(encoding="utf-8")
    return source.split("\ndef parse_args()")[0].rstrip()


SETUP_CELL = f"""
KAGGLE_DATA_DIR = Path("{INPUT_DATA_DIR}")

if KAGGLE_DATA_DIR.exists():
    DATA_DIR = KAGGLE_DATA_DIR
    REPO_ROOT = DATA_DIR.parent
else:
    REPO_ROOT = find_repo_root()
    DATA_DIR = REPO_ROOT / "data"

RESULTS_DIR = Path("/kaggle/working/results") if Path("/kaggle/working").exists() else REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = RESULTS_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RUN_FULL_GRID = True

print(f"DATA_DIR={{DATA_DIR}}")
print(f"REPO_ROOT={{REPO_ROOT}}")
print(f"RESULTS_DIR={{RESULTS_DIR}}")
print(f"MODEL_DIR={{MODEL_DIR}}")
print(f"RUN_FULL_GRID={{RUN_FULL_GRID}}")
"""


SCAN_CELL = """
attack_files = select_attack_files(REPO_ROOT)
attack_scan = pd.DataFrame([
    {
        "dataset": item.dataset,
        "model": item.model,
        "attack": item.attack,
        "epsilon": item.epsilon,
        "path": str(item.path),
    }
    for item in attack_files
])

display(attack_scan.sort_values(["dataset", "model", "attack"]))
display(
    attack_scan
    .groupby(["dataset", "model", "attack"], dropna=False)
    .size()
    .reset_index(name="selected_files")
)
"""


DIAGNOSTIC_CELL = """
base_diag_path, nn_diag_path = generate_diagnostic_csvs(
    repo_root=REPO_ROOT,
    output_dir=RESULTS_DIR,
    recompute=True,
)

base_diag = pd.read_csv(base_diag_path)
nn_diag = pd.read_csv(nn_diag_path)

print(base_diag_path, base_diag.shape)
print(nn_diag_path, nn_diag.shape)
display(base_diag.groupby(["dataset", "model", "attack"]).size().reset_index(name="rows"))
display(nn_diag.groupby(["dataset", "model", "attack"]).size().reset_index(name="rows"))
"""


DIAGNOSTIC_REUSE_CELL = """
base_diag_path, nn_diag_path = generate_diagnostic_csvs(
    repo_root=REPO_ROOT,
    output_dir=RESULTS_DIR,
    recompute=False,
)

base_diag = pd.read_csv(base_diag_path)
nn_diag = pd.read_csv(nn_diag_path)

print(base_diag_path, base_diag.shape)
print(nn_diag_path, nn_diag.shape)
display(base_diag.groupby(["dataset", "model", "attack"]).size().reset_index(name="rows"))
display(nn_diag.groupby(["dataset", "model", "attack"]).size().reset_index(name="rows"))
"""


BINARY_AGG_CELL = """
binary_table = build_binary_aggregated_table(RESULTS_DIR)

print(binary_table.shape)
display(binary_table.head())
display(binary_table["attack_binary"].value_counts().rename_axis("attack_binary").reset_index(name="rows"))
display(binary_table["attack"].value_counts().rename_axis("attack").reset_index(name="rows"))
display(binary_table.groupby(["dataset", "model"]).size().reset_index(name="rows"))
"""


TYPE_AGG_CELL = """
attack_type_table = build_attack_type_aggregated_table(RESULTS_DIR)

print(attack_type_table.shape)
display(attack_type_table.head())
display(attack_type_table["attack"].value_counts().rename_axis("attack").reset_index(name="rows"))
display(attack_type_table.groupby(["dataset", "model"]).size().reset_index(name="rows"))
"""


def tune_cell(prefix: str, table_name: str, task: str, model_class: str, grid_name: str) -> str:
    return f"""
{prefix}_{model_class}_best_params, {prefix}_{model_class}_search = tune_on_leave_dataset(
    {table_name},
    task="{task}",
    model_class="{model_class}",
    grid={grid_name},
) if RUN_FULL_GRID else ({{}}, pd.DataFrame())

{prefix}_{model_class}_metrics, {prefix}_{model_class}_fi, {prefix}_{model_class}_predictions = evaluate_{'binary' if task == 'binary' else 'attack_type'}(
    {table_name},
    model_class="{model_class}",
    params={prefix}_{model_class}_best_params,
)

print("Best {model_class.upper()} params:", {prefix}_{model_class}_best_params)
if not {prefix}_{model_class}_search.empty:
    display({prefix}_{model_class}_search.sort_values("mean_kappa", ascending=False).head(10))
display({prefix}_{model_class}_metrics)
"""


BINARY_SAVE_CELL = """
binary_metrics = pd.concat([binary_xgb_metrics, binary_rf_metrics], ignore_index=True)
binary_fi = pd.concat([binary_xgb_fi, binary_rf_fi], ignore_index=True)
binary_predictions = pd.concat([binary_xgb_predictions, binary_rf_predictions], ignore_index=True)

binary_metrics_path = RESULTS_DIR / "detection_bacc_with_bacc.csv"
binary_fi_path = RESULTS_DIR / "detection_fi_with_bacc.csv"
binary_predictions_path = RESULTS_DIR / "detection_predictions_with_bacc.csv"
binary_confusion_path = RESULTS_DIR / "detection_confusion_matrices_with_bacc.csv"
binary_best_params_path = RESULTS_DIR / "binary_best_params.json"

binary_metrics.to_csv(binary_metrics_path, index=False)
binary_fi.to_csv(binary_fi_path, index=False)
binary_predictions.to_csv(binary_predictions_path, index=False)
save_confusion_matrices(binary_predictions, labels=[0, 1], out_path=binary_confusion_path)

binary_best_params = {"xgb": binary_xgb_best_params, "rf": binary_rf_best_params}
binary_best_params_path.write_text(json.dumps(binary_best_params, indent=2), encoding="utf-8")
binary_xgb_search.assign(model_class="XGB").to_csv(RESULTS_DIR / "binary_xgb_hyperparameter_search.csv", index=False)
binary_rf_search.assign(model_class="RF").to_csv(RESULTS_DIR / "binary_rf_hyperparameter_search.csv", index=False)

print(binary_metrics_path)
print(binary_fi_path)
print(binary_predictions_path)
print(binary_confusion_path)
print(binary_best_params_path)
"""


BINARY_MODEL_CELL = """
binary_xgb_artifacts = fit_and_save_final_model(
    binary_table,
    task="binary",
    model_class="xgb",
    params=binary_xgb_best_params,
    output_dir=RESULTS_DIR,
    artifact_name="binary_xgb",
)

binary_rf_artifacts = fit_and_save_final_model(
    binary_table,
    task="binary",
    model_class="rf",
    params=binary_rf_best_params,
    output_dir=RESULTS_DIR,
    artifact_name="binary_rf",
)

print("Saved final binary detector models and inference artifacts:")
for family, artifacts in {"binary_xgb": binary_xgb_artifacts, "binary_rf": binary_rf_artifacts}.items():
    print(family)
    for artifact_type, path in artifacts.items():
        print(f"  {artifact_type}: {path}")
"""


BINARY_SUMMARY_CELL = """
summary_cols = ["bacc", "kappa", "precision", "recall", "f1", "fnr"]
binary_summary = (
    binary_metrics
    .groupby(["scenario", "model_class"])[summary_cols]
    .agg(["mean", "std"])
    .round(4)
)
display(binary_summary)

display(
    binary_fi
    .groupby(["model_class", "var"])["fi_rank"]
    .mean()
    .reset_index()
    .sort_values(["model_class", "fi_rank"])
    .groupby("model_class")
    .head(20)
)
"""


TYPE_SAVE_CELL = """
attack_type_metrics = pd.concat([attack_type_xgb_metrics, attack_type_rf_metrics], ignore_index=True)
attack_type_fi = pd.concat([attack_type_xgb_fi, attack_type_rf_fi], ignore_index=True)
attack_type_predictions = pd.concat([attack_type_xgb_predictions, attack_type_rf_predictions], ignore_index=True)
attack_type_labels = sorted(attack_type_table["attack"].unique())

attack_type_metrics_path = RESULTS_DIR / "isolation_bacc_nn.csv"
attack_type_fi_path = RESULTS_DIR / "isolation_fi_nn.csv"
attack_type_predictions_path = RESULTS_DIR / "isolation_predictions_nn.csv"
attack_type_confusion_path = RESULTS_DIR / "isolation_confusion_matrices_nn.csv"
attack_type_best_params_path = RESULTS_DIR / "attack_type_best_params.json"

attack_type_metrics.to_csv(attack_type_metrics_path, index=False)
attack_type_fi.to_csv(attack_type_fi_path, index=False)
attack_type_predictions.to_csv(attack_type_predictions_path, index=False)
save_confusion_matrices(attack_type_predictions, labels=attack_type_labels, out_path=attack_type_confusion_path)

attack_type_best_params = {"xgb": attack_type_xgb_best_params, "rf": attack_type_rf_best_params}
attack_type_best_params_path.write_text(json.dumps(attack_type_best_params, indent=2), encoding="utf-8")
attack_type_xgb_search.assign(model_class="XGB").to_csv(RESULTS_DIR / "attack_type_xgb_hyperparameter_search.csv", index=False)
attack_type_rf_search.assign(model_class="RF").to_csv(RESULTS_DIR / "attack_type_rf_hyperparameter_search.csv", index=False)

print(attack_type_metrics_path)
print(attack_type_fi_path)
print(attack_type_predictions_path)
print(attack_type_confusion_path)
print(attack_type_best_params_path)
"""


TYPE_MODEL_CELL = """
attack_type_xgb_artifacts = fit_and_save_final_model(
    attack_type_table,
    task="type",
    model_class="xgb",
    params=attack_type_xgb_best_params,
    output_dir=RESULTS_DIR,
    artifact_name="attack_type_xgb",
)

attack_type_rf_artifacts = fit_and_save_final_model(
    attack_type_table,
    task="type",
    model_class="rf",
    params=attack_type_rf_best_params,
    output_dir=RESULTS_DIR,
    artifact_name="attack_type_rf",
)

print("Saved final attack-type classifier models and inference artifacts:")
for family, artifacts in {"attack_type_xgb": attack_type_xgb_artifacts, "attack_type_rf": attack_type_rf_artifacts}.items():
    print(family)
    for artifact_type, path in artifacts.items():
        print(f"  {artifact_type}: {path}")
"""


TYPE_SUMMARY_CELL = """
summary_cols = ["bacc", "kappa", "precision_weighted", "recall_weighted", "f1_weighted"]
attack_type_summary = (
    attack_type_metrics
    .groupby(["scenario", "model_class"])[summary_cols]
    .agg(["mean", "std"])
    .round(4)
)
display(attack_type_summary)

display(
    attack_type_fi
    .groupby(["model_class", "var"])["fi_rank"]
    .mean()
    .reset_index()
    .sort_values(["model_class", "fi_rank"])
    .groupby("model_class")
    .head(20)
)
"""


def build_binary_notebook() -> dict:
    helper = standalone_helper_code()
    return notebook(
        [
            markdown(
                """# Standalone Binary Attack Detection with RF/XGBoost

This notebook has no project-file imports. It embeds the diagnostic generation, aggregation, evaluation, and model-saving code directly in the notebook."""
            ),
            markdown("## Step 0 - Inline Helper Code"),
            code(helper),
            markdown("## Step 1 - Kaggle Paths and Run Settings"),
            code(SETUP_CELL),
            markdown("## Step 2 - Scan Available Upstream Attacks"),
            code(SCAN_CELL),
            markdown("## Step 3 - Generate Diagnostic-Style CSVs"),
            code(DIAGNOSTIC_CELL),
            markdown("## Step 4 - Aggregate Diagnostics for Binary Detection"),
            code(BINARY_AGG_CELL),
            markdown("## Step 5 - Tune and Evaluate XGBoost"),
            code(tune_cell("binary", "binary_table", "binary", "xgb", "XGB_PARAM_GRID")),
            markdown("## Step 6 - Tune and Evaluate Random Forest"),
            code(tune_cell("binary", "binary_table", "binary", "rf", "RF_PARAM_GRID")),
            markdown("## Step 7 - Save Metrics, Predictions, Feature Importances, and Confusion Matrices"),
            code(BINARY_SAVE_CELL),
            markdown("## Step 8 - Refit on Full Meta-Table and Save Final Models"),
            code(BINARY_MODEL_CELL),
            markdown("## Step 9 - Summary Views"),
            code(BINARY_SUMMARY_CELL),
        ]
    )


def build_attack_type_notebook() -> dict:
    helper = standalone_helper_code()
    return notebook(
        [
            markdown(
                """# Standalone Attack Type Classification with RF/XGBoost

This notebook has no project-file imports. It embeds the diagnostic generation, aggregation, evaluation, and model-saving code directly in the notebook."""
            ),
            markdown("## Step 0 - Inline Helper Code"),
            code(helper),
            markdown("## Step 1 - Kaggle Paths and Run Settings"),
            code(SETUP_CELL),
            markdown("## Step 2 - Scan Available Upstream Attacks"),
            code(SCAN_CELL),
            markdown("## Step 3 - Generate or Reuse Diagnostic-Style CSVs"),
            code(DIAGNOSTIC_REUSE_CELL),
            markdown("## Step 4 - Aggregate Diagnostics for Attack-Type Classification"),
            code(TYPE_AGG_CELL),
            markdown("## Step 5 - Tune and Evaluate XGBoost"),
            code(tune_cell("attack_type", "attack_type_table", "type", "xgb", "XGB_PARAM_GRID")),
            markdown("## Step 6 - Tune and Evaluate Random Forest"),
            code(tune_cell("attack_type", "attack_type_table", "type", "rf", "RF_PARAM_GRID")),
            markdown("## Step 7 - Save Metrics, Predictions, Feature Importances, and Confusion Matrices"),
            code(TYPE_SAVE_CELL),
            markdown("## Step 8 - Refit on Full Meta-Table and Save Final Models"),
            code(TYPE_MODEL_CELL),
            markdown("## Step 9 - Summary Views"),
            code(TYPE_SUMMARY_CELL),
        ]
    )


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    (out_dir / "attack_binary_classifier_with_bacc_rf_xgb.ipynb").write_text(
        json.dumps(build_binary_notebook(), indent=2),
        encoding="utf-8",
    )
    (out_dir / "attack_type_classifier_add_nn_rf_xgb.ipynb").write_text(
        json.dumps(build_attack_type_notebook(), indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
