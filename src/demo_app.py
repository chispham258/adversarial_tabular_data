from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import balanced_accuracy_score

try:
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover - handled in UI.
    go = None

from rf_xgb_detection import (
    NeighborhoodDiagnostics,
    _aggregate,
    build_clean_org_dataframe,
    find_repo_root,
    load_clean_split,
    predict_upstream_model,
)


SUPPORTED_TYPE_LABELS = ["bim", "fgm", "hsj", "org", "pgd", "zoo"]
DATASETS = ["banknote", "diabetes", "wilt"]
MODELS = ["nn", "lin", "svm", "xgb"]
DEFAULT_BATCH_SIZE = 20
LIVE_ART_MAX_BATCH = 10

ATTACK_PROFILES: Dict[str, Dict[str, str]] = {
    "fgm": {
        "name": "Fast Gradient Method (FGM)",
        "type": "White-box / gradient-based",
        "signature": "Thường thay đổi nhiều cột một lượng nhỏ, tạo nhiễu mịn và khó thấy bằng mắt.",
        "note": "Nhanh, phù hợp để demo live với mô hình NN.",
    },
    "bim": {
        "name": "Basic Iterative Method (BIM)",
        "type": "White-box / iterative gradient",
        "signature": "Tương tự FGM nhưng cập nhật nhiều bước, nhiễu có xu hướng rõ hơn.",
        "note": "Ổn cho demo live với batch nhỏ.",
    },
    "pgd": {
        "name": "Projected Gradient Descent (PGD)",
        "type": "White-box / iterative projected gradient",
        "signature": "Tấn công mạnh hơn FGM/BIM, bóp dữ liệu theo nhiều bước trong một biên epsilon.",
        "note": "Nên dùng batch nhỏ khi chạy live.",
    },
    "zoo": {
        "name": "Zeroth Order Optimization (ZOO)",
        "type": "Black-box / query-based",
        "signature": "Ước lượng hướng tấn công bằng cách hỏi mô hình nhiều lần; thường chậm.",
        "note": "Demo nên dùng replay để tránh treo giao diện.",
    },
    "hsj": {
        "name": "HopSkipJump (HSJ)",
        "type": "Black-box / decision-boundary search",
        "signature": "Có thể thay đổi rất ít cột nhưng vẫn đẩy mẫu qua biên quyết định.",
        "note": "Rất hay để trình diễn delta highlight; live mode có thể chậm.",
    },
    "lpf": {
        "name": "LowProFool (LPF)",
        "type": "Black-box / tabular-oriented",
        "signature": "Tập trung thay đổi các feature có độ ưu tiên cao.",
        "note": "Có file replay nhưng không thuộc các nhãn attack-type cuối cùng.",
    },
}


def repo_root() -> Path:
    return find_repo_root()


@st.cache_data(show_spinner=False)
def processed_oos(dataset: str, root_str: str) -> Tuple[pd.DataFrame, pd.Series]:
    root = Path(root_str)
    x_path = root / "data" / "processed" / dataset / "X_test.csv"
    y_path = root / "data" / "processed" / dataset / "y_test.csv"
    X = pd.read_csv(x_path)
    y = pd.read_csv(y_path).iloc[:, 0]
    return X, y


@st.cache_data(show_spinner=False)
def available_attack_table(root_str: str) -> pd.DataFrame:
    root = Path(root_str)
    rows = []
    for path in sorted((root / "data" / "adversarial_upstream").glob("*/*.csv")):
        parts = path.stem.split("_")
        if len(parts) < 2:
            continue
        model = parts[0]
        attack = parts[1]
        epsilon = None
        if "eps" in parts:
            epsilon = parts[-1]
        rows.append(
            {
                "dataset": path.parent.name,
                "model": model,
                "attack": attack,
                "epsilon": epsilon,
                "path": str(path),
            }
        )
    return pd.DataFrame(rows)


@st.cache_resource(show_spinner=False)
def clean_context(dataset: str, model_name: str, root_str: str):
    root = Path(root_str)
    split = load_clean_split(root, dataset)
    clean_df = build_clean_org_dataframe(root, dataset, model_name, split)
    context = NeighborhoodDiagnostics(clean_df, split.feature_names, split.n_classes)
    return split, clean_df, context


@st.cache_resource(show_spinner=False)
def load_detector_artifacts(root_str: str, detector_kind: str, model_family: str):
    root = Path(root_str)
    if detector_kind == "binary":
        artifact_dir = root / "models" / "binary_attack"
        prefix = f"binary_{model_family}"
        model_path = artifact_dir / f"{prefix}.joblib"
        encoder_path = artifact_dir / f"{prefix}_label_encoder.joblib"
        features_path = artifact_dir / f"{prefix}_features.json"
        config_path = artifact_dir / f"{prefix}_config.json"
    elif detector_kind == "type_optimized":
        artifact_dir = root / "models" / "attack_type_optimized"
        prefix = "attack_type_optimized"
        model_path = artifact_dir / f"{prefix}_best.joblib"
        encoder_path = artifact_dir / f"{prefix}_label_encoder.joblib"
        features_path = artifact_dir / f"{prefix}_features.json"
        config_path = artifact_dir / f"{prefix}_config.json"
    elif detector_kind == "family_optimized":
        artifact_dir = root / "models" / "attack_type_optimized"
        prefix = "attack_family_optimized"
        model_path = artifact_dir / f"{prefix}_best.joblib"
        encoder_path = artifact_dir / f"{prefix}_label_encoder.joblib"
        features_path = artifact_dir / f"{prefix}_features.json"
        config_path = artifact_dir / f"{prefix}_config.json"
    else:
        artifact_dir = root / "models" / "attack_type"
        prefix = f"attack_type_{model_family}"
        model_path = artifact_dir / f"{prefix}.joblib"
        encoder_path = artifact_dir / f"{prefix}_label_encoder.joblib"
        features_path = artifact_dir / f"{prefix}_features.json"
        config_path = artifact_dir / f"{prefix}_config.json"

    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    features = json.loads(features_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return model, encoder, features, config


def choose_attack_file(
    attacks: pd.DataFrame,
    dataset: str,
    model_name: str,
    attack: str,
) -> Optional[Path]:
    subset = attacks[
        (attacks["dataset"] == dataset)
        & (attacks["model"] == model_name)
        & (attacks["attack"] == attack)
    ].copy()
    if subset.empty:
        return None
    if "epsilon" in subset.columns and subset["epsilon"].notna().any():
        subset["epsilon_num"] = pd.to_numeric(subset["epsilon"], errors="coerce").fillna(-1)
        subset = subset.sort_values("epsilon_num", ascending=False)
    return Path(subset.iloc[0]["path"])


def sample_indices(total: int, batch_size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    size = min(batch_size, total)
    return np.sort(rng.choice(total, size=size, replace=False))


def load_processed_oos_batch(
    dataset: str,
    batch_size: int,
    seed: int,
    root: Path,
) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
    X, y = processed_oos(dataset, str(root))
    indices = sample_indices(len(X), batch_size, seed)
    return X.iloc[indices].reset_index(drop=True), y.iloc[indices].reset_index(drop=True), indices


def load_precomputed_attack(
    dataset: str,
    model_name: str,
    attack: str,
    indices: Sequence[int],
    root: Path,
) -> pd.DataFrame:
    attacks = available_attack_table(str(root))
    path = choose_attack_file(attacks, dataset, model_name, attack)
    if path is None:
        raise FileNotFoundError(f"No precomputed attack file for {dataset}/{model_name}/{attack}")
    adv = pd.read_csv(path)
    valid_indices = [idx for idx in indices if idx < len(adv)]
    return adv.iloc[valid_indices].reset_index(drop=True)


def build_detector_dataframe(
    X_adv: np.ndarray,
    dataset: str,
    model_name: str,
    indices: Sequence[int],
    root: Path,
) -> pd.DataFrame:
    split, _clean_df, _context = clean_context(dataset, model_name, str(root))
    y_true = split.y_test[np.asarray(indices)]
    y_pred, probs = predict_upstream_model(root, dataset, model_name, X_adv)
    df = pd.DataFrame(X_adv, columns=split.feature_names)
    df["name"] = list(indices)
    df["is_train"] = 0
    df["target"] = y_true
    df["prediction"] = y_pred
    for class_idx in range(probs.shape[1]):
        df[f"score_{class_idx}"] = probs[:, class_idx]
    return df


def generate_live_art_attack(
    dataset: str,
    model_name: str,
    attack: str,
    indices: Sequence[int],
    root: Path,
) -> pd.DataFrame:
    if model_name != "nn" or attack not in {"fgm", "bim", "pgd"}:
        raise NotImplementedError("Live ART mode is implemented only for nn + fgm/bim/pgd.")

    try:
        import torch
        import torch.nn as nn
        from art.attacks.evasion import BasicIterativeMethod, FastGradientMethod, ProjectedGradientDescent
        from art.estimators.classification import PyTorchClassifier
    except ImportError as exc:
        raise RuntimeError("ART/PyTorch is not installed. Use replay mode or install requirements.") from exc

    split, _clean_df, _context = clean_context(dataset, model_name, str(root))
    model_path = root / "models" / dataset / "upstream_nn.pt"
    checkpoint = torch.load(model_path, map_location="cpu")

    class TabDataModel2(nn.Module):
        def __init__(self, input_dim: int, num_classes: int):
            super().__init__()
            self.input = nn.Linear(input_dim, 32)
            self.relu = nn.ReLU()
            self.output = nn.Linear(32, num_classes)

        def forward(self, x):
            return self.output(self.relu(self.input(x)))

    model = TabDataModel2(checkpoint["input_dim"], checkpoint["num_classes"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    classifier = PyTorchClassifier(
        model=model,
        loss=loss_fn,
        optimizer=optimizer,
        input_shape=(split.X_train.shape[1],),
        nb_classes=split.n_classes,
    )

    eps = 0.2
    if attack == "fgm":
        attacker = FastGradientMethod(estimator=classifier, eps=eps)
    elif attack == "bim":
        attacker = BasicIterativeMethod(estimator=classifier, eps=eps, eps_step=0.05, max_iter=20)
    else:
        attacker = ProjectedGradientDescent(
            estimator=classifier,
            eps=eps,
            eps_step=0.05,
            max_iter=20,
            num_random_init=1,
        )

    idx = np.asarray(indices)
    X_clean = split.X_test[idx].astype(np.float32)
    X_adv = attacker.generate(x=X_clean)
    X_adv = np.clip(np.nan_to_num(X_adv, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    return build_detector_dataframe(X_adv, dataset, model_name, idx, root)


def clean_detector_batch(dataset: str, model_name: str, indices: Sequence[int], root: Path) -> pd.DataFrame:
    _split, clean_df, _context = clean_context(dataset, model_name, str(root))
    valid_indices = [idx for idx in indices if idx < len(clean_df)]
    return clean_df.iloc[valid_indices].reset_index(drop=True)


def manual_detector_batch(
    dataset: str,
    model_name: str,
    manual_df: pd.DataFrame,
    seed: int,
    root: Path,
) -> pd.DataFrame:
    split, _clean_df, _context = clean_context(dataset, model_name, str(root))
    manual_values = manual_df.reindex(columns=split.feature_names)
    manual_values = manual_values.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if manual_values.empty:
        raise ValueError("Manual input table is empty.")

    row_count = len(manual_values)
    base_count = min(row_count, len(split.y_test))
    indices = sample_indices(len(split.y_test), base_count, seed)
    if row_count > base_count:
        indices = np.resize(indices, row_count)

    X_manual = manual_values.to_numpy(dtype=np.float32)
    return build_detector_dataframe(X_manual, dataset, model_name, indices, root)


def detector_batch_from_attack_generator(payload: Dict[str, Any], root: Path) -> pd.DataFrame:
    if payload.get("adv_df") is not None:
        return payload["adv_df"]
    return load_precomputed_attack(
        payload["dataset"],
        payload["model"],
        payload["attack"],
        payload["indices"],
        root,
    )


def safe_bacc(df: pd.DataFrame) -> float:
    try:
        return float(balanced_accuracy_score(df["target"], df["prediction"]))
    except Exception:
        return float("nan")


def aggregate_for_detector(
    batch_df: pd.DataFrame,
    dataset: str,
    model_name: str,
    attack: str,
    root: Path,
    task: str,
) -> pd.DataFrame:
    split, _clean_df, context = clean_context(dataset, model_name, str(root))
    diag = context.diagnose(batch_df, dataset, model_name, attack, safe_bacc(batch_df))
    if task == "binary":
        diag["attack_binary"] = 0 if attack == "org" else 1
        return _aggregate(diag, ["dataset", "model", "attack", "n_test", "n_classes", "attack_binary"])
    return _aggregate(diag, ["dataset", "model", "attack", "bacc_test", "n_test", "n_classes"])


def align_features(row: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    out = row.copy()
    for col in features:
        if col not in out.columns:
            out[col] = 0.0
    return out[features]


def predict_with_artifact(
    row: pd.DataFrame,
    detector_kind: str,
    model_family: str,
    root: Path,
) -> Dict[str, Any]:
    model, encoder, features, _config = load_detector_artifacts(str(root), detector_kind, model_family)
    X = align_features(row, features)
    pred_enc = model.predict(X)
    pred_label = encoder.inverse_transform(pred_enc.astype(int))[0]
    proba = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        classes = encoder.inverse_transform(np.arange(len(probs)))
        proba = {str(label): float(prob) for label, prob in zip(classes, probs)}
    return {"label": pred_label, "proba": proba}


def optimized_artifact_available(root: Path, prefix: str) -> bool:
    artifact_dir = root / "models" / "attack_type_optimized"
    return all(
        (artifact_dir / f"{prefix}{suffix}").exists()
        for suffix in ["_best.joblib", "_label_encoder.joblib", "_features.json", "_config.json"]
    )


def predict_batch_detector(
    batch_df: pd.DataFrame,
    dataset: str,
    model_name: str,
    attack: str,
    detector_family: str,
    root: Path,
) -> Dict[str, Any]:
    binary_row = aggregate_for_detector(batch_df, dataset, model_name, attack, root, task="binary")
    binary_result = predict_with_artifact(binary_row, "binary", detector_family, root)

    type_result = None
    family_result = None
    type_source = None
    family_source = None
    if attack in SUPPORTED_TYPE_LABELS:
        type_row = aggregate_for_detector(batch_df, dataset, model_name, attack, root, task="type")
        if optimized_artifact_available(root, "attack_type_optimized"):
            type_result = predict_with_artifact(type_row, "type_optimized", detector_family, root)
            type_source = "optimized"
        else:
            type_result = predict_with_artifact(type_row, "type", detector_family, root)
            type_source = detector_family.upper()

        if optimized_artifact_available(root, "attack_family_optimized"):
            family_result = predict_with_artifact(type_row, "family_optimized", detector_family, root)
            family_source = "optimized"

    return {
        "binary": binary_result,
        "attack_type": type_result,
        "attack_type_source": type_source,
        "attack_family": family_result,
        "attack_family_source": family_source,
        "binary_features": binary_row,
    }


def compute_delta_table(clean: pd.DataFrame, attacked: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in clean.columns if c in attacked.columns]
    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(clean[c])]
    return attacked[numeric].reset_index(drop=True) - clean[numeric].reset_index(drop=True)


def delta_styler(delta: pd.DataFrame):
    max_abs = float(np.nanmax(np.abs(delta.to_numpy()))) if not delta.empty else 0.0
    if max_abs <= 0:
        max_abs = 1.0

    def style_value(value):
        intensity = min(abs(float(value)) / max_abs, 1.0) if pd.notna(value) else 0.0
        alpha = 0.12 + 0.75 * intensity
        if abs(float(value)) < 1e-8:
            return "background-color: #f8fafc; color: #64748b"
        return f"background-color: rgba(220, 38, 38, {alpha:.3f}); color: #111827"

    return delta.style.format("{:.4f}").map(style_value)


def make_radar_chart(clean_row: pd.Series, attacked_row: pd.Series, title: str):
    if go is None:
        return None
    cols = [c for c in clean_row.index if c in attacked_row.index]
    values = pd.DataFrame({"clean": clean_row[cols], "attacked": attacked_row[cols]}).astype(float)
    min_v = values.min(axis=1)
    max_v = values.max(axis=1)
    denom = (max_v - min_v).replace(0, 1.0)
    clean_norm = ((values["clean"] - min_v) / denom).tolist()
    adv_norm = ((values["attacked"] - min_v) / denom).tolist()
    categories = cols + [cols[0]]
    clean_norm = clean_norm + [clean_norm[0]]
    adv_norm = adv_norm + [adv_norm[0]]
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(r=clean_norm, theta=categories, fill="toself", name="Clean", line_color="#2563eb")
    )
    fig.add_trace(
        go.Scatterpolar(r=adv_norm, theta=categories, fill="toself", name="Adversarial", line_color="#dc2626")
    )
    fig.update_layout(title=title, polar=dict(radialaxis=dict(visible=True, range=[0, 1])), height=420)
    return fig


def attack_options(dataset: str, model_name: str, root: Path, live_mode: bool) -> List[str]:
    if live_mode and model_name == "nn":
        return ["fgm", "bim", "pgd"]
    attacks = available_attack_table(str(root))
    opts = sorted(
        attacks[(attacks["dataset"] == dataset) & (attacks["model"] == model_name)]["attack"].unique()
    )
    return opts


def attack_profile(attack: str) -> None:
    profile = ATTACK_PROFILES.get(attack, {})
    st.info(
        "\n".join(
            [
                f"**Tấn công:** {profile.get('name', attack)}",
                f"**Loại hình:** {profile.get('type', 'N/A')}",
                f"**Chữ ký:** {profile.get('signature', 'N/A')}",
                f"**Ghi chú:** {profile.get('note', 'N/A')}",
            ]
        )
    )


def verdict_text(label: Any) -> str:
    label_str = str(label)
    if label_str in {"1", "attack", "True"}:
        return "Phát hiện tấn công"
    return "Bình thường"


def display_probability(proba: Optional[Dict[str, float]]) -> None:
    if not proba:
        return
    prob_df = pd.DataFrame({"label": list(proba.keys()), "probability": list(proba.values())})
    st.dataframe(prob_df, use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="Adversarial Tabular Demo", layout="wide")
    root = repo_root()

    st.title("Demo: Nhận diện tấn công trên dữ liệu dạng bảng")
    st.caption(
        "Mô hình detector trả kết luận ở mức batch/window. Bảng từng dòng bên dưới là phần giải thích trực quan, "
        "không phải dự đoán độc lập cho từng dòng."
    )

    with st.sidebar:
        st.header("Cấu hình")
        st.write(f"Repo: `{root}`")
        detector_family = st.selectbox("Detector model", ["rf", "xgb"], index=0, format_func=str.upper)
        st.warning("Demo dùng 3 datasets: banknote, diabetes, wilt. Các attack per/noise không có trong dữ liệu hiện tại.")

    tab_attack, tab_detector = st.tabs(["Mũi giáo: Attack Generator", "Tấm khiên: Test Detector"])

    with tab_attack:
        st.subheader("Tạo batch tấn công")
        mode = st.radio(
            "Chế độ tạo dữ liệu",
            ["Fast demo replay (Recommended)", "Live ART generation"],
            horizontal=True,
        )
        live_mode = mode == "Live ART generation"
        if live_mode:
            st.warning(
                f"Live ART chỉ hỗ trợ ổn định cho nn + fgm/bim/pgd và sẽ giới hạn batch tối đa {LIVE_ART_MAX_BATCH} dòng."
            )

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            dataset = st.selectbox("Dataset", DATASETS)
        with c2:
            model_name = st.selectbox("Mô hình bị tấn công", MODELS)
        opts = attack_options(dataset, model_name, root, live_mode)
        if not opts:
            st.error("Không có attack phù hợp cho lựa chọn này.")
            st.stop()
        with c3:
            attack = st.selectbox("Attack", opts)
        with c4:
            batch_size = st.selectbox("Batch size", [10, 20, 50, 100], index=1)

        seed = st.number_input("Random seed", min_value=0, max_value=999_999, value=42, step=1)
        effective_batch = min(int(batch_size), LIVE_ART_MAX_BATCH) if live_mode else int(batch_size)
        if live_mode and effective_batch < batch_size:
            st.caption(f"Live mode đang dùng {effective_batch} dòng để tránh chạy quá lâu.")

        attack_profile(attack)

        clean_processed, y_processed, indices = load_processed_oos_batch(dataset, effective_batch, int(seed), root)
        split, clean_detector_full, _context = clean_context(dataset, model_name, str(root))
        clean_detector = clean_detector_full.iloc[indices].reset_index(drop=True)

        adv_df: Optional[pd.DataFrame] = None
        source = "replay"
        if st.button("Generate attack batch", type="primary"):
            if live_mode:
                try:
                    with st.spinner("Đang chạy ART attack live..."):
                        adv_df = generate_live_art_attack(dataset, model_name, attack, indices, root)
                        source = "live_art"
                except Exception as exc:
                    st.warning(f"Live attack failed; using precomputed attack sample for demo stability. ({exc})")
                    adv_df = load_precomputed_attack(dataset, model_name, attack, indices, root)
                    source = "replay_fallback"
            else:
                adv_df = load_precomputed_attack(dataset, model_name, attack, indices, root)

            st.session_state["attack_payload"] = {
                "dataset": dataset,
                "model": model_name,
                "attack": attack,
                "indices": indices,
                "adv_df": adv_df,
                "source": source,
            }

        payload = st.session_state.get("attack_payload")
        if payload and payload["dataset"] == dataset and payload["model"] == model_name and payload["attack"] == attack:
            adv_df = detector_batch_from_attack_generator(payload, root)
            source = payload.get("source", "replay")

        st.markdown("### Dữ liệu sạch từ `data/processed`")
        display_clean = clean_processed.copy()
        display_clean["target"] = y_processed
        st.dataframe(display_clean, use_container_width=True)

        if adv_df is not None:
            st.success(f"Attack batch ready. Source: `{source}`")
            feature_cols = split.feature_names
            clean_compare = clean_detector[feature_cols].reset_index(drop=True)
            adv_compare = adv_df[feature_cols].reset_index(drop=True)
            delta = compute_delta_table(clean_compare, adv_compare)

            left, right = st.columns(2)
            with left:
                st.markdown("### Clean batch trong attack/detector space")
                st.dataframe(clean_compare, use_container_width=True)
            with right:
                st.markdown("### Adversarial batch")
                st.dataframe(adv_compare, use_container_width=True)

            st.markdown("### Delta highlight: adversarial - clean")
            st.dataframe(delta_styler(delta), use_container_width=True)

            row_choice = st.slider("Chọn dòng để xem radar chart", 0, len(clean_compare) - 1, 0)
            fig = make_radar_chart(
                clean_compare.iloc[row_choice],
                adv_compare.iloc[row_choice],
                f"{dataset}/{model_name}/{attack} - row {row_choice}",
            )
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Plotly chưa được cài, không thể hiển thị radar chart.")

            if st.button("Send this batch to Detector"):
                st.session_state["detector_payload"] = st.session_state["attack_payload"]
                st.success("Đã gửi batch sang tab Detector.")

    with tab_detector:
        st.subheader("Kiểm thử mô hình phòng thủ")
        source_options = ["Random clean OOS batch", "Manual input"]
        if "detector_payload" in st.session_state:
            source_options.insert(0, "Batch from Attack Generator")
        source_choice = st.radio("Nguồn batch", source_options, horizontal=True)

        if source_choice == "Batch from Attack Generator":
            payload = st.session_state["detector_payload"]
            det_dataset = payload["dataset"]
            det_model = payload["model"]
            det_attack = payload["attack"]
            det_indices = payload["indices"]
            batch_df = detector_batch_from_attack_generator(payload, root)
        elif source_choice == "Manual input":
            d1, d2, d3 = st.columns(3)
            with d1:
                det_dataset = st.selectbox("Dataset kiểm thử", DATASETS, key="manual_det_dataset")
            with d2:
                det_model = st.selectbox("Mô hình giám sát", MODELS, key="manual_det_model")
            with d3:
                det_batch_size = st.selectbox("Số dòng nhập", [1, 5, 10, 20], index=2, key="manual_det_bs")
            det_seed = st.number_input(
                "Seed tạo bảng mẫu",
                min_value=0,
                max_value=999_999,
                value=11,
                step=1,
                key="manual_det_seed",
            )

            split, clean_detector_full, _context = clean_context(det_dataset, det_model, str(root))
            template_indices = sample_indices(len(clean_detector_full), int(det_batch_size), int(det_seed))
            template = clean_detector_full.iloc[template_indices][split.feature_names].reset_index(drop=True)
            st.caption(
                "Nhập hoặc sửa trực tiếp các giá trị feature bên dưới. "
                "Các giá trị này được hiểu là feature space của mô hình giám sát/attack."
            )
            manual_values = st.data_editor(
                template,
                num_rows="dynamic",
                use_container_width=True,
                key=f"manual_values_{det_dataset}_{det_model}_{det_batch_size}_{det_seed}",
            )
            det_attack = "org"
            batch_df = manual_detector_batch(det_dataset, det_model, manual_values, int(det_seed), root)
        else:
            d1, d2, d3 = st.columns(3)
            with d1:
                det_dataset = st.selectbox("Dataset kiểm thử", DATASETS, key="det_dataset")
            with d2:
                det_model = st.selectbox("Mô hình giám sát", MODELS, key="det_model")
            with d3:
                det_batch_size = st.selectbox("Batch size kiểm thử", [10, 20, 50, 100], index=1, key="det_bs")
            det_seed = st.number_input("Seed kiểm thử", min_value=0, max_value=999_999, value=7, step=1)
            _X_display, _y_display, det_indices = load_processed_oos_batch(det_dataset, int(det_batch_size), int(det_seed), root)
            det_attack = "org"
            batch_df = clean_detector_batch(det_dataset, det_model, det_indices, root)

        st.caption(
            f"Batch đang kiểm thử: dataset=`{det_dataset}`, model=`{det_model}`, "
            f"attack/source=`{det_attack}`, rows={len(batch_df)}"
        )

        if det_attack == "lpf":
            st.warning("LPF có thể hiển thị trực quan, nhưng không thuộc nhãn attack-type cuối cùng.")

        if len(batch_df) < 30:
            st.warning("Batch nhỏ hơn 30 dòng: kết quả phù hợp cho demo/cảnh báo nhanh, không nên xem là quyết định chắc chắn.")

        if st.button("Run detector", type="primary"):
            with st.spinner("Đang tạo diagnostic vector và chạy detector..."):
                try:
                    result = predict_batch_detector(batch_df, det_dataset, det_model, det_attack, detector_family, root)
                    label = result["binary"]["label"]
                    text = verdict_text(label)
                    st.metric("Binary detector verdict", text)
                    display_probability(result["binary"]["proba"])

                    if result["attack_type"] is not None:
                        type_label = "Attack-type classifier"
                        if result.get("attack_type_source"):
                            type_label += f" ({result['attack_type_source']})"
                        st.metric(type_label, str(result["attack_type"]["label"]))
                        display_probability(result["attack_type"]["proba"])
                    else:
                        st.info("Attack-type detector không hỗ trợ nhãn này.")

                    if result.get("attack_family") is not None:
                        family_label = "Attack-family classifier"
                        if result.get("attack_family_source"):
                            family_label += f" ({result['attack_family_source']})"
                        st.metric(family_label, str(result["attack_family"]["label"]))
                        display_probability(result["attack_family"]["proba"])
                    else:
                        st.info("Attack-family detector chưa có artifact tối ưu.")

                    rows_view = batch_df.copy()
                    rows_view["batch_verdict"] = text
                    if result.get("attack_type") is not None:
                        rows_view["batch_attack_type"] = str(result["attack_type"]["label"])
                    if result.get("attack_family") is not None:
                        rows_view["batch_attack_family"] = str(result["attack_family"]["label"])
                    rows_view["row_note"] = "Member of inspected batch"
                    st.markdown("### Các dòng thuộc batch được kiểm thử")
                    st.dataframe(rows_view, use_container_width=True)

                    with st.expander("Diagnostic vector fed to detector"):
                        st.dataframe(result["binary_features"], use_container_width=True)
                except Exception as exc:
                    st.error(f"Detector failed: {exc}")


if __name__ == "__main__":
    main()
