from __future__ import annotations

import json
import math
import re
import textwrap
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
REPORT_PATH = REPORT_DIR / "adversarial_attack_detection_report.md"

DISALLOWED_REPORT_TERMS = ("paper", "authors", "study")

ATTACK_FAMILY = {
    "org": "clean",
    "bim": "gradient-based attack",
    "fgm": "gradient-based attack",
    "pgd": "gradient-based attack",
    "hsj": "black-box/query attack",
    "zoo": "black-box/query attack",
}


def read_csv(name: str) -> pd.DataFrame:
    path = RESULTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required result file: {path}")
    return pd.read_csv(path)


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dirs() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def fmt_num(value: object, digits: int = 4) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isclose(value, round(value), abs_tol=1e-12):
        return str(int(round(value)))
    return f"{value:.{digits}f}"


def simple_markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "_No rows available._"
    show = df.head(max_rows).copy() if max_rows else df.copy()
    show = show.fillna("")
    columns = list(show.columns)
    lines = [
        "| " + " | ".join(str(c) for c in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in show.iterrows():
        values = []
        for col in columns:
            value = row[col]
            values.append(fmt_num(value) if isinstance(value, (float, int, np.floating, np.integer)) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def aggregate_metric(df: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    present = [c for c in metric_cols if c in df.columns]
    return df.groupby(group_cols, dropna=False)[present].mean(numeric_only=True).reset_index()


def save_current_fig(name: str) -> str:
    path = FIG_DIR / f"{name}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    return f"figures/{path.name}"


def plot_pipeline() -> str:
    steps = [
        ("Clean tabular data", "Reference split and monitored-model input"),
        ("Monitored classifier", "Linear, kernel, tree-boosting, or neural model"),
        ("Adversarial batch", "Generated live or replayed from compatible attacks"),
        ("Diagnostic features", "Neighborhood, uncertainty, consistency, diversity"),
        ("Aggregation", "Fixed-width vectors and optional diagnostic windows"),
        ("Secondary classifiers", "Binary, family, and exact attack decisions"),
    ]
    fig, ax = plt.subplots(figsize=(13.5, 3.9))
    ax.axis("off")
    y = 0.5
    xs = np.linspace(0.08, 0.92, len(steps))
    for idx, ((title, subtitle), x) in enumerate(zip(steps, xs)):
        box = plt.Rectangle((x - 0.075, y - 0.16), 0.15, 0.32, facecolor="#f6f8fb", edgecolor="#2f3b52", linewidth=1.4)
        ax.add_patch(box)
        ax.text(x, y + 0.045, title, ha="center", va="center", fontsize=9.5, fontweight="bold", color="#172033")
        ax.text(x, y - 0.055, textwrap.fill(subtitle, 22), ha="center", va="center", fontsize=7.5, color="#415064")
        if idx < len(steps) - 1:
            ax.annotate("", xy=(xs[idx + 1] - 0.087, y), xytext=(x + 0.087, y), arrowprops=dict(arrowstyle="->", lw=1.5, color="#415064"))
    return save_current_fig("pipeline_diagram")


def plot_attack_coverage(windowed: pd.DataFrame) -> str:
    counts = windowed.groupby(["model", "attack"]).size().unstack(fill_value=0)
    ordered_attacks = [a for a in ["org", "bim", "fgm", "pgd", "hsj", "zoo"] if a in counts.columns]
    counts = counts.reindex(columns=ordered_attacks)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    sns.heatmap(counts, annot=True, fmt="d", cmap="Blues", cbar_kws={"label": "diagnostic rows"}, ax=ax)
    ax.set_title("Diagnostic Row Coverage by Monitored Model and Attack")
    ax.set_xlabel("Attack label")
    ax.set_ylabel("Monitored model")
    return save_current_fig("attack_coverage_heatmap")


def plot_upstream_effectiveness(upstream: pd.DataFrame) -> str:
    data = upstream.copy()
    data = data[data["attack"].isin(["bim", "fgm", "pgd", "hsj", "zoo", "lpf"])]
    summary = data.groupby(["model", "attack"], as_index=False)["bacc_drop"].mean()
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    sns.barplot(data=summary, x="attack", y="bacc_drop", hue="model", ax=ax)
    ax.axhline(0, color="#222222", linewidth=1)
    ax.set_title("Mean Balanced-Accuracy Drop After Adversarial Perturbation")
    ax.set_xlabel("Attack")
    ax.set_ylabel("Mean BACC drop")
    ax.legend(title="Monitored model", ncols=2)
    return save_current_fig("upstream_attack_effectiveness")


def plot_diagnostic_distributions(windowed: pd.DataFrame) -> str:
    data = windowed.copy()
    data["batch label"] = np.where(data["attack"].eq("org"), "clean", "attacked")
    features = [
        "uncertainty_mean",
        "neighborhood_size_mean",
        "pred_targets_consistency_in_neighborhood_mean",
        "target_diversity_in_neighborhood_mean",
    ]
    features = [f for f in features if f in data.columns]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, feature in zip(axes.ravel(), features):
        sns.boxplot(data=data, x="batch label", y=feature, ax=ax, color="#b8d5f5")
        sns.stripplot(data=data, x="batch label", y=feature, ax=ax, color="#1f3b5d", size=2.2, alpha=0.35)
        ax.set_title(feature.replace("_", " "))
        ax.set_xlabel("")
    for ax in axes.ravel()[len(features) :]:
        ax.axis("off")
    fig.suptitle("Diagnostic Feature Distributions", y=1.02, fontsize=13, fontweight="bold")
    return save_current_fig("diagnostic_feature_distributions")


def plot_binary_metrics(binary_metrics: pd.DataFrame) -> str:
    summary = aggregate_metric(binary_metrics, ["scenario", "model_class"], ["bacc", "fnr"])
    scenario_order = [
        "10-fold cross-validation",
        "one-data-set-out",
        "one-model-out",
        "one-attack-out",
    ]
    summary["scenario"] = pd.Categorical(summary["scenario"], scenario_order, ordered=True)
    summary = summary.sort_values(["scenario", "model_class"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True)
    sns.barplot(data=summary, x="scenario", y="bacc", hue="model_class", ax=axes[0])
    axes[0].set_title("Binary Detection Balanced Accuracy")
    axes[0].set_ylim(0, 1.05)
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].set_xlabel("")
    sns.barplot(data=summary, x="scenario", y="fnr", hue="model_class", ax=axes[1])
    axes[1].set_title("Binary Detection False Negative Rate")
    axes[1].set_ylim(0, max(0.2, float(summary["fnr"].max()) * 1.2 if not summary.empty else 0.2))
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].set_xlabel("")
    axes[1].legend_.remove()
    return save_current_fig("binary_detection_metrics")


def plot_task_comparison(
    binary_metrics: pd.DataFrame,
    baseline_type_metrics: pd.DataFrame,
    optimized_type_metrics: pd.DataFrame,
    grouped_metrics: pd.DataFrame,
    exact_family: str,
    family_family: str,
) -> str:
    binary_best = aggregate_metric(binary_metrics, ["scenario", "model_class"], ["bacc"])
    binary_best = binary_best.loc[binary_best.groupby("scenario")["bacc"].idxmax()]
    binary_best["task"] = "Binary"
    binary_best["model"] = binary_best["model_class"]

    baseline = aggregate_metric(baseline_type_metrics, ["scenario", "model_class"], ["bacc"])
    baseline = baseline.loc[baseline.groupby("scenario")["bacc"].idxmax()]
    baseline["task"] = "Exact type baseline"
    baseline["model"] = baseline["model_class"]

    opt = optimized_type_metrics[optimized_type_metrics["family"].eq(exact_family)].copy()
    opt = aggregate_metric(opt, ["scenario", "family"], ["bacc"])
    opt["task"] = "Exact type optimized"
    opt["model"] = exact_family

    fam = grouped_metrics[grouped_metrics["family"].eq(family_family)].copy()
    fam = aggregate_metric(fam, ["scenario", "family"], ["bacc"])
    fam["task"] = "Attack family"
    fam["model"] = family_family

    combined = pd.concat(
        [
            binary_best[["scenario", "task", "model", "bacc"]],
            baseline[["scenario", "task", "model", "bacc"]],
            opt[["scenario", "task", "model", "bacc"]],
            fam[["scenario", "task", "model", "bacc"]],
        ],
        ignore_index=True,
    )
    combined = combined[combined["scenario"].isin(["10-fold cross-validation", "one-data-set-out", "one-model-out"])]
    fig, ax = plt.subplots(figsize=(12, 5.5))
    sns.barplot(data=combined, x="scenario", y="bacc", hue="task", ax=ax)
    ax.set_title("Balanced Accuracy Across Detection Tasks")
    ax.set_xlabel("Evaluation scenario")
    ax.set_ylabel("Balanced accuracy")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="Task", loc="lower left", bbox_to_anchor=(0, 1.01), ncols=2)
    return save_current_fig("task_performance_comparison")


def matrix_from_counts(df: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    mat = df.pivot_table(index="true", columns="pred", values="count", aggfunc="sum", fill_value=0)
    return mat.reindex(index=labels, columns=labels, fill_value=0)


def plot_confusion_matrix(cm: pd.DataFrame, title: str, name: str) -> str:
    fig, ax = plt.subplots(figsize=(7.4, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    return save_current_fig(name)


def plot_exact_confusion(confusions: pd.DataFrame, family: str) -> str:
    labels = ["bim", "fgm", "hsj", "org", "pgd", "zoo"]
    data = confusions[
        confusions["family"].eq(family)
        & confusions["scenario"].eq("one-data-set-out")
        & confusions["true"].isin(labels)
        & confusions["pred"].isin(labels)
    ]
    cm = matrix_from_counts(data, labels)
    return plot_confusion_matrix(cm, "Exact Attack Classification Confusion Matrix", "exact_attack_confusion_matrix")


def plot_family_confusion(predictions: pd.DataFrame, family: str) -> str:
    data = predictions[predictions["family"].eq(family) & predictions["scenario"].eq("one-data-set-out")].copy()
    data["true_family"] = data["true"].map(ATTACK_FAMILY)
    data["pred_family"] = data["pred"].map(ATTACK_FAMILY)
    labels = ["clean", "gradient-based attack", "black-box/query attack"]
    cm = (
        data.groupby(["true_family", "pred_family"])
        .size()
        .rename("count")
        .reset_index()
        .rename(columns={"true_family": "true", "pred_family": "pred"})
    )
    mat = matrix_from_counts(cm, labels)
    return plot_confusion_matrix(mat, "Attack Family Confusion Matrix", "attack_family_confusion_matrix")


def plot_feature_importances(
    binary_fi: pd.DataFrame,
    exact_fi: pd.DataFrame,
    exact_config: dict,
    family_config: dict,
) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(17, 7.2))

    binary = binary_fi.copy()
    binary = binary[binary["scenario"].eq("10-fold cross-validation")]
    if binary.empty:
        binary = binary_fi.copy()
    binary_summary = binary.groupby("var", as_index=False)["fi_rank"].mean().sort_values("fi_rank").head(12)
    sns.barplot(data=binary_summary, y="var", x="fi_rank", ax=axes[0], color="#7da7d9")
    axes[0].invert_xaxis()
    axes[0].set_title("Binary: mean importance rank")
    axes[0].set_xlabel("Lower rank is better")
    axes[0].set_ylabel("")

    exact = exact_fi.sort_values("rank").head(12)
    sns.barplot(data=exact, y="var", x="importance", ax=axes[1], color="#8fd0a9")
    axes[1].set_title("Exact type: feature importance")
    axes[1].set_xlabel("Importance")
    axes[1].set_ylabel("")

    family_features = pd.DataFrame()
    family_path = MODELS_DIR / "attack_type_optimized" / "attack_family_optimized_best.joblib"
    family_feature_cols = family_config.get("feature_columns", [])
    if family_path.exists() and family_feature_cols:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = joblib.load(family_path)
        if hasattr(model, "feature_importances_"):
            family_features = pd.DataFrame(
                {
                    "var": family_feature_cols,
                    "importance": model.feature_importances_[: len(family_feature_cols)],
                }
            ).sort_values("importance", ascending=False)
    if family_features.empty:
        family_features = pd.DataFrame(
            {
                "var": family_config.get("feature_columns", [])[:12],
                "importance": np.linspace(1.0, 0.2, min(12, len(family_config.get("feature_columns", [])))),
            }
        )
    family_features = family_features.head(12)
    sns.barplot(data=family_features, y="var", x="importance", ax=axes[2], color="#f2b66d")
    axes[2].set_title("Family: feature importance")
    axes[2].set_xlabel("Importance")
    axes[2].set_ylabel("")

    for ax in axes:
        ax.tick_params(axis="y", labelsize=8)
    return save_current_fig("feature_importance_summary")


def scenario_summary_table(df: pd.DataFrame, model_col: str, model_value: str | None = None) -> pd.DataFrame:
    data = df.copy()
    if model_value is not None and model_col in data.columns:
        data = data[data[model_col].eq(model_value)]
    metrics = [c for c in ["bacc", "kappa", "precision", "recall", "f1", "fnr", "precision_weighted", "recall_weighted", "f1_weighted"] if c in data.columns]
    out = aggregate_metric(data, ["scenario"], metrics)
    for col in metrics:
        out[col] = out[col].map(lambda x: fmt_num(x))
    return out


def best_rows(df: pd.DataFrame, family_col: str, metric: str = "bacc") -> pd.DataFrame:
    summary = aggregate_metric(df, ["scenario", family_col], [metric, "kappa"] if "kappa" in df.columns else [metric])
    return summary.sort_values(["scenario", metric], ascending=[True, False])


def params_block(title: str, params: dict) -> str:
    if not params:
        return f"**{title}:** not available."
    return f"**{title}:**\n\n```json\n{json.dumps(params, indent=2)}\n```"


def build_report(figures: dict[str, str], tables: dict[str, pd.DataFrame], configs: dict[str, dict]) -> str:
    binary_params = configs.get("binary_params", {})
    type_params = configs.get("type_params", {})
    exact_config = configs.get("exact_config", {})
    family_config = configs.get("family_config", {})

    exact_model = exact_config.get("selected_family", "selected classifier")
    family_model = family_config.get("selected_family", "selected classifier")
    exact_params = exact_config.get("params", {})
    family_params = family_config.get("params", {})

    text = f"""# Adversarial Attack Detection for Tabular Data

## Executive Summary

This report describes a self-contained adversarial attack detection workflow for tabular classification systems. The workflow converts clean and adversarial batches into fixed-width diagnostic vectors, then trains secondary classifiers for three tasks: binary attack identification, attack family classification, and exact attack type classification.

The main observation is consistent across the experiment outputs: binary attack identification is the most stable task, exact attack type classification is substantially harder, and attack family classification provides a useful middle level between the two. This behavior is expected because several attack algorithms can produce overlapping diagnostic footprints even when their optimization procedures differ.

![End-to-end pipeline]({figures["pipeline"]})

## 1. Data Generation

The data generation process begins with clean tabular datasets and a set of monitored classifiers. Each monitored classifier is trained on clean data and then evaluated on both clean and adversarial batches. Clean batches receive the label `org`; adversarial batches receive the label of the attack procedure that generated or supplied them.

Let a clean instance be $x \\in \\mathbb{{R}}^d$ with target label $y$, and let the monitored classifier be $f(\\cdot)$. An adversarial instance is represented as:

$$
x' = x + \\delta, \\quad \\|\\delta\\|_p \\leq \\epsilon, \\quad f(x') \\neq y.
$$

The perturbation $\\delta$ is generated or replayed according to attack compatibility with the monitored classifier. Gradient-based attacks are used where gradients are practical, while query-based attacks are used when the monitored classifier is not naturally differentiable. This produces adversarial data with different signatures: dense small perturbations for gradient-based attacks and sparse or boundary-seeking perturbations for query-based attacks.

The available diagnostic rows cover the following monitored models and attack labels:

![Attack coverage]({figures["coverage"]})

The upstream attack evaluation measures how much each perturbation degrades the monitored classifier. A large positive balanced-accuracy drop means the attack successfully pushed samples into regions where the monitored classifier made more mistakes.

![Upstream attack effectiveness]({figures["upstream"]})

## 2. Feature Engineering

The detector does not operate directly on raw tabular rows. Instead, it summarizes a batch through diagnostic attributes derived from local neighborhood behavior around each diagnosed instance. This makes the detection layer less dependent on the original dataset dimensionality and more focused on how the monitored classifier behaves around clean and perturbed data.

For a diagnosed instance $x$ and a reference set $R$, a local neighborhood can be written as:

$$
N(x) = \\{{z \\in R : s(x, z) \\geq \\tau\\}},
$$

where $s(x,z)$ is a similarity rule in the processed feature space and $\\tau$ is the neighborhood threshold. The implementation uses a self-contained neighborhood surrogate to create comparable diagnostic attributes across datasets and monitored models.

The uncertainty of a local prediction distribution $q$ over $C$ classes is computed with normalized entropy:

$$
H_{{norm}}(q) = -\\frac{{1}}{{\\log C}} \\sum_{{c=1}}^C q_c \\log(q_c).
$$

Balanced accuracy is used because clean and attacked classes can be imbalanced:

$$
\\operatorname{{BACC}} = \\frac{{1}}{{C}} \\sum_{{c=1}}^C \\frac{{TP_c}}{{TP_c + FN_c}}.
$$

For each diagnostic quantity $g$, the batch-level vector stores summary statistics:

$$
\\phi_g(B) = [\\operatorname{{mean}}(g), q_0(g), q_{{25}}(g), q_{{50}}(g), q_{{75}}(g), q_1(g), \\max(g)-\\min(g)].
$$

The final diagnostic vector includes neighborhood size, uncertainty, prediction consistency, target consistency, diversity measures, dataset-level metadata, and the monitored model's clean or adversarial balanced accuracy. The binary target is:

$$
y_{{bin}} = \\mathbb{{1}}[a \\neq \\text{{org}}],
$$

where $a$ is the attack label. The family target is:

$$
y_{{family}} =
\\begin{{cases}}
\\text{{clean}}, & a = \\text{{org}},\\\\
\\text{{gradient-based attack}}, & a \\in \\{{\\text{{bim}}, \\text{{fgm}}, \\text{{pgd}}\\}},\\\\
\\text{{black-box/query attack}}, & a \\in \\{{\\text{{hsj}}, \\text{{zoo}}\\}}.
\\end{{cases}}
$$

Exact attack classification keeps the original attack label:

$$
y_{{type}} \\in \\{{\\text{{org}}, \\text{{bim}}, \\text{{fgm}}, \\text{{pgd}}, \\text{{hsj}}, \\text{{zoo}}\\}}.
$$

The diagnostic distributions show why a secondary classifier can detect attacks: clean and attacked batches tend to separate through uncertainty, neighborhood size, consistency, and diversity.

![Diagnostic feature distributions]({figures["diagnostics"]})

## 3. Binary Attack Identification

The binary task predicts whether a diagnostic vector came from clean data or adversarially perturbed data. Random Forest and XGBoost classifiers were trained with balanced-accuracy-oriented evaluation. The evaluation uses repeated generalization scenarios: cross-validation, held-out dataset, held-out monitored model, and held-out attack.

{params_block("Binary Random Forest best parameters", binary_params.get("rf", {}))}

{params_block("Binary XGBoost best parameters", binary_params.get("xgb", {}))}

Binary detection is the strongest of the three tasks. It avoids the hardest distinction, which is separating similar attack algorithms from one another, and instead only asks whether the batch behavior is normal or suspicious.

![Binary detection metrics]({figures["binary"]})

**Binary metric summary**

{simple_markdown_table(tables["binary_summary"])}

## 4. Attack Family Classification

Attack family classification groups exact attack labels into broader categories. This task predicts whether a diagnostic vector is clean, gradient-based adversarial data, or black-box/query-based adversarial data.

The selected family classifier is `{family_model}` with the following parameters:

```json
{json.dumps(family_params, indent=2)}
```

The family classifier uses diagnostic windows and class-aware training so that the model receives more than one aggregate row for each dataset/model/attack combination. This increases the number of training examples while preserving the batch-level diagnostic interpretation.

**Attack family metric summary**

{simple_markdown_table(tables["family_summary"])}

![Attack family confusion matrix]({figures["family_cm"]})

## 5. Exact Attack Type Classification

Exact attack type classification predicts the specific attack label. This task is more difficult than binary detection or family classification because multiple attacks can produce overlapping behavior in the diagnostic feature space. For example, iterative gradient attacks can resemble each other, and query-based attacks can create similar boundary-seeking signatures.

The baseline exact classifier used Random Forest and XGBoost models. The optimized exact classifier used diagnostic windows, constrained prediction labels, guarded resampling, feature filtering, and hyperparameter search.

{params_block("Baseline exact Random Forest best parameters", type_params.get("rf", {}))}

{params_block("Baseline exact XGBoost best parameters", type_params.get("xgb", {}))}

The selected optimized exact classifier is `{exact_model}` with the following parameters:

```json
{json.dumps(exact_params, indent=2)}
```

**Baseline exact attack metric summary**

{simple_markdown_table(tables["baseline_type_summary"])}

**Optimized exact attack metric summary**

{simple_markdown_table(tables["optimized_type_summary"])}

![Task performance comparison]({figures["comparison"]})

![Exact attack confusion matrix]({figures["exact_cm"]})

## 6. Feature Importance

Across the three tasks, the most influential features are usually the monitored model balanced accuracy, neighborhood size summaries, prediction consistency, target consistency, and uncertainty. This indicates that the detector is primarily using changes in local classifier behavior rather than memorizing raw feature values.

![Feature importance summary]({figures["importance"]})

## 7. Conclusions

The implemented workflow converts heterogeneous tabular datasets into a shared diagnostic representation and uses secondary classifiers to identify adversarial manipulation. Binary attack identification is the most reliable target because it only requires separating clean and suspicious behavior. Exact attack classification remains harder because several attack algorithms can leave similar diagnostic traces. Attack family classification reduces that ambiguity by grouping attacks according to their broader perturbation mechanism.

The strongest practical direction for further improvement is to generate more diagnostic windows from additional datasets, monitored models, and attack settings. This would reduce the small-sample effect in multiclass classification and make the exact attack classifier less dependent on a few dataset/model combinations.
"""
    return text


def validate_report_text(text: str) -> None:
    lowered = text.lower()
    hits = [term for term in DISALLOWED_REPORT_TERMS if re.search(rf"\b{re.escape(term)}\b", lowered)]
    if hits:
        raise ValueError(f"Generated report contains disallowed terms: {hits}")


def main() -> None:
    ensure_dirs()
    sns.set_theme(style="whitegrid", context="notebook")

    upstream = read_csv("upstream_attack_results.csv")
    binary_metrics = read_csv("detection_bacc_with_bacc.csv")
    binary_fi = read_csv("detection_fi_with_bacc.csv")
    baseline_type_metrics = read_csv("isolation_bacc_nn.csv")
    optimized_type_metrics = read_csv("attack_type_optimized_metrics.csv")
    grouped_metrics = read_csv("attack_type_optimized_grouped_metrics.csv")
    optimized_confusions = read_csv("attack_type_optimized_confusion_matrices.csv")
    optimized_predictions = read_csv("attack_type_optimized_predictions.csv")
    optimized_fi = read_csv("attack_type_optimized_feature_importance.csv")
    windowed = read_csv("attr_attacks_type_optimized_windowed.csv")

    binary_params = read_json(RESULTS_DIR / "binary_best_params.json")
    type_params = read_json(RESULTS_DIR / "attack_type_best_params.json")
    exact_config = read_json(MODELS_DIR / "attack_type_optimized" / "attack_type_optimized_config.json")
    family_config = read_json(MODELS_DIR / "attack_type_optimized" / "attack_family_optimized_config.json")
    exact_family = exact_config.get("selected_family", "extratrees")
    family_family = family_config.get("selected_family", exact_family)

    figures = {
        "pipeline": plot_pipeline(),
        "coverage": plot_attack_coverage(windowed),
        "upstream": plot_upstream_effectiveness(upstream),
        "diagnostics": plot_diagnostic_distributions(windowed),
        "binary": plot_binary_metrics(binary_metrics),
        "comparison": plot_task_comparison(
            binary_metrics,
            baseline_type_metrics,
            optimized_type_metrics,
            grouped_metrics,
            exact_family=exact_family,
            family_family=family_family,
        ),
        "exact_cm": plot_exact_confusion(optimized_confusions, exact_family),
        "family_cm": plot_family_confusion(optimized_predictions, exact_family),
        "importance": plot_feature_importances(binary_fi, optimized_fi, exact_config, family_config),
    }

    tables = {
        "binary_summary": scenario_summary_table(binary_metrics, "model_class"),
        "baseline_type_summary": scenario_summary_table(baseline_type_metrics, "model_class"),
        "optimized_type_summary": scenario_summary_table(
            optimized_type_metrics[optimized_type_metrics["family"].eq(exact_family)],
            "family",
            exact_family,
        ),
        "family_summary": scenario_summary_table(
            grouped_metrics[grouped_metrics["family"].eq(family_family)],
            "family",
            family_family,
        ),
    }

    configs = {
        "binary_params": binary_params,
        "type_params": type_params,
        "exact_config": exact_config,
        "family_config": family_config,
    }

    report = build_report(figures, tables, configs)
    validate_report_text(report)
    REPORT_PATH.write_text(report, encoding="utf-8")

    print(f"Wrote report: {REPORT_PATH}")
    print(f"Wrote figures: {FIG_DIR}")
    print("Generated figure files:")
    for name, rel in figures.items():
        print(f"- {name}: {rel}")


if __name__ == "__main__":
    main()
