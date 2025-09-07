import os
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime


# -------------------------
# Page / Cache Config
# -------------------------
st.set_page_config(page_title="LaminitisCare - Risk Assistant", layout="wide")


# Domain schema: expected types and ranges
FEATURE_SCHEMAS: Dict[str, Dict[str, object]] = {
    "Respiratoryrate": {"type": "int", "min": 8, "max": 32, "step": 1},
    "Rectaltemperature": {"type": "float", "min": 36.1, "max": 38.0, "step": 0.1},
    "Digitalpulses": {"type": "int", "min": 0, "max": 6, "step": 1},
    "LLRF": {"type": "int", "min": 0, "max": 1, "step": 1},
    "LLLH": {"type": "int", "min": 0, "max": 1, "step": 1},
    "HTLH": {"type": "int", "min": 0, "max": 1, "step": 1},
    "LERF": {"type": "int", "min": 0, "max": 3, "step": 1},
    "LELF": {"type": "int", "min": 0, "max": 3, "step": 1},
    "LERH": {"type": "int", "min": 0, "max": 3, "step": 1},
    "LELH": {"type": "int", "min": 0, "max": 3, "step": 1},
}


@st.cache_resource(show_spinner=False)
def load_pickle(p: Path):
    with open(p, "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner=False)
def load_artifacts() -> Dict[str, object]:
    """Load model, scaler, and feature order from local files.

    Tries relative paths based on this file location first; falls back to the
    absolute workspace paths if needed.
    """
    here = Path(__file__).resolve()
    base_dir = here.parent.parent  # project root containing the pkl files

    model_path = base_dir / "svm_model.pkl"
    scaler_path = base_dir / "svm_scaler.pkl"
    features_path = base_dir / "svm_features.pkl"

    # Fallback to absolute paths if necessary
    if not model_path.exists():
        model_path = Path("/Users/mac/Downloads/Laminitis/svm_model.pkl")
    if not scaler_path.exists():
        scaler_path = Path("/Users/mac/Downloads/Laminitis/svm_scaler.pkl")
    if not features_path.exists():
        features_path = Path("/Users/mac/Downloads/Laminitis/svm_features.pkl")

    model = load_pickle(model_path)
    scaler = load_pickle(scaler_path)
    feature_order = load_pickle(features_path)

    if isinstance(feature_order, (pd.Index, np.ndarray)):
        feature_order = list(feature_order)

    return {
        "model": model,
        "scaler": scaler,
        "feature_order": feature_order,
        "base_dir": base_dir,
    }


def align_features(df: pd.DataFrame, feature_order: List[str]) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Align columns to training feature order.

    - Unknown columns are dropped and returned as `extra_cols`.
    - Missing columns are added with NaN and returned as `missing_cols`.
    """
    df_cols = list(df.columns)
    extra_cols = [c for c in df_cols if c not in feature_order]
    missing_cols = [c for c in feature_order if c not in df_cols]

    # Keep only known columns
    df_known = df[[c for c in df_cols if c in feature_order]].copy()

    # Add missing columns as NaN
    for c in missing_cols:
        df_known[c] = np.nan

    # Reorder
    df_known = df_known[feature_order]
    return df_known, missing_cols, extra_cols


def fill_missing_with_training_stats(df: pd.DataFrame, scaler, feature_order: List[str]) -> pd.DataFrame:
    """Fill NaNs using training statistics from the scaler when available.

    Prefers aligning with scaler.feature_names_in_ if present to ensure correct mapping
    of statistics to columns. Falls back to provided feature_order.
    """
    filled = df.copy()
    names_for_stats: Optional[List[str]] = None
    if hasattr(scaler, "feature_names_in_"):
        try:
            names_for_stats = list(getattr(scaler, "feature_names_in_"))
        except Exception:
            names_for_stats = None
    if names_for_stats is None:
        names_for_stats = feature_order

    if hasattr(scaler, "mean_") and len(getattr(scaler, "mean_")) == len(names_for_stats):
        means = list(getattr(scaler, "mean_"))
        for i, feat in enumerate(names_for_stats):
            if feat in filled.columns:
                filled[feat] = pd.to_numeric(filled[feat], errors="coerce").astype(float).fillna(float(means[i]))
        # Any remaining NaNs: median
        filled = filled.fillna(filled.median(numeric_only=True))
    else:
        filled = filled.fillna(filled.median(numeric_only=True))
    return filled


def coerce_and_clip_to_schema(df: pd.DataFrame, schemas: Dict[str, Dict[str, object]]) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Coerce numeric types and clip to configured ranges.

    Returns adjusted DataFrame and per-column adjustment counts (values that were clipped or coerced).
    """
    adjusted = df.copy()
    adjustments: Dict[str, int] = {}
    for col, cfg in schemas.items():
        if col not in adjusted.columns:
            continue
        col_series = pd.to_numeric(adjusted[col], errors="coerce")
        before = col_series.copy()
        # Round for integer features before clipping
        if cfg.get("type") == "int":
            col_series = col_series.round().astype("Int64")
        # Clip to min/max
        min_v = cfg.get("min")
        max_v = cfg.get("max")
        if min_v is not None:
            col_series = col_series.where(col_series >= min_v, other=min_v)
        if max_v is not None:
            col_series = col_series.where(col_series <= max_v, other=max_v)
        # Count adjustments (including rounding and clipping)
        adj_count = int((before != col_series).sum(skipna=True))
        if adj_count > 0:
            adjustments[col] = adj_count
        adjusted[col] = col_series
    return adjusted, adjustments


def fill_missing_with_schema_means(df: pd.DataFrame, schemas: Dict[str, Dict[str, object]], feature_order: List[str]) -> pd.DataFrame:
    filled = df.copy()
    for feat in feature_order:
        if feat in filled.columns and filled[feat].isna().any():
            cfg = schemas.get(feat)
            if cfg and cfg.get("min") is not None and cfg.get("max") is not None:
                mid = (float(cfg["min"]) + float(cfg["max"])) / 2.0
                filled[feat] = pd.to_numeric(filled[feat], errors="coerce").fillna(mid)
            else:
                filled[feat] = pd.to_numeric(filled[feat], errors="coerce").fillna(filled[feat].median(skipna=True))
    return filled


def domain_scale(df: pd.DataFrame, schemas: Dict[str, Dict[str, object]], feature_order: List[str]) -> np.ndarray:
    scaled = pd.DataFrame(index=df.index)
    for feat in feature_order:
        series = pd.to_numeric(df[feat], errors="coerce")
        cfg = schemas.get(feat)
        if cfg and cfg.get("min") is not None and cfg.get("max") is not None:
            min_v = float(cfg["min"])
            max_v = float(cfg["max"])
            denom = (max_v - min_v) if (max_v - min_v) != 0 else 1.0
            s = (series - min_v) / denom
            s = s.clip(lower=0.0, upper=1.0)
            scaled[feat] = s
        else:
            # If no schema, pass-through (best-effort)
            scaled[feat] = series
    return scaled.values


def preprocess(df: pd.DataFrame, scaler) -> np.ndarray:
    # Preserve DataFrame with column names to avoid sklearn feature-name warnings
    df_float = df.astype(float)
    try:
        return scaler.transform(df_float)
    except Exception:
        return scaler.transform(df_float.values)


def proba_or_confidence(model, X: np.ndarray) -> Tuple[np.ndarray, str]:
    """Return probability if available, otherwise decision_function through a sigmoid as confidence.

    Returns: (scores, method) where method in {"probability", "confidence", "none"}
    """
    if hasattr(model, "predict_proba"):
        try:
            p = model.predict_proba(X)[:, 1]
            return p, "probability"
        except Exception:
            pass
    if hasattr(model, "decision_function"):
        try:
            df_score = model.decision_function(X)
            p = 1.0 / (1.0 + np.exp(-df_score))  # uncalibrated confidence-like score
            return p, "confidence"
        except Exception:
            pass
    preds = model.predict(X)
    p = np.where(preds == 1, 1.0, 0.0)
    return p, "none"


def risk_label_from_score(score: float) -> str:
    return "High" if score >= 0.5 else "Low"


def guidance_band_from_score(score: float) -> str:
    # Conservative bands for guidance
    if score >= 0.7:
        return "High"
    if score >= 0.4:
        return "Moderate"
    return "Low"


def render_guidance(band: str):
    if band == "High":
        st.error(
            "Immediate veterinary evaluation recommended. Stall rest, restrict movement, "
            "consider NSAIDs strictly per veterinarian guidance, and monitor digital pulse "
            "and hoof temperature frequently."
        )
    elif band == "Moderate":
        st.warning(
            "Increase monitoring (e.g., twice daily). Adjust diet to reduce non-structural "
            "carbohydrates, ensure hydration, and track lameness, respiratory rate, and rectal temperature."
        )
    else:
        st.info(
            "Maintain routine care. Do weekly hoof checks, keep records, and educate caretakers "
            "on early warning signs (heat in hoof, stronger digital pulse, shifting weight)."
        )


def explain_prediction_plain(score: float, method: str) -> str:
    if method == "probability":
        return (
            "This score reflects the model's estimated probability of laminitis risk based on the "
            "entered measurements. Higher values indicate greater concern."
        )
    if method == "confidence":
        return (
            "This confidence score is derived from the model's decision boundary and is not a "
            "calibrated probability, but higher values still indicate higher concern."
        )
    return (
        "This result is based on the model's classification output. Because a probability wasn't "
        "available, please interpret the result alongside clinical signs and consult a veterinarian."
    )


def predict_dataframe(df_raw: pd.DataFrame, artifacts: Dict[str, object]) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    feature_order: List[str] = artifacts["feature_order"]
    model = artifacts["model"]
    scaler = artifacts["scaler"]

    aligned, missing, extra = align_features(df_raw, feature_order)

    # Use domain schema scaling to match training normalization, avoiding saved scaler
    filled = fill_missing_with_schema_means(aligned, FEATURE_SCHEMAS, feature_order)
    X_for_model = domain_scale(filled, FEATURE_SCHEMAS, feature_order)
    scores, method = proba_or_confidence(model, X_for_model)
    labels = np.array([risk_label_from_score(s) for s in scores])
    guidance_bands = np.array([guidance_band_from_score(s) for s in scores])

    result = aligned.copy()
    if method == "probability":
        result["probability"] = scores
    elif method == "confidence":
        result["confidence"] = scores
    else:
        result["score"] = scores
    result["risk_label"] = labels
    result["guidance_band"] = guidance_bands

    warnings = {"missing": missing, "extra": extra, "score_method": method}
    return result, warnings


def render_shap_image(base_dir: Path):
    shap_img = base_dir / "global_shap_summary_normalized.png"
    if shap_img.exists():
        st.caption("Model-wide feature impact from training (for context):")
        st.image(str(shap_img), width="stretch")


# -------------------------
# Local Agent Assistant
# -------------------------
def _ensure_agent_state():
    if "agent_history" not in st.session_state:
        st.session_state["agent_history"] = []  # list of {role, content}
    if "last_single" not in st.session_state:
        st.session_state["last_single"] = None
    if "last_batch" not in st.session_state:
        st.session_state["last_batch"] = None
    if "llm_api_key" not in st.session_state:
        # Load from env, else from local file Streamlit/openai_api_key.txt if present
        key = os.environ.get("OPENAI_API_KEY", "")
        try:
            if not key:
                key_file = Path(__file__).parent / "openai_api_key.txt"
                if key_file.exists():
                    key = key_file.read_text(encoding="utf-8").strip()
        except Exception:
            pass
        st.session_state["llm_api_key"] = key
    # Auto-enable LLM if a key is present; otherwise use local agent
    st.session_state["llm_enabled"] = bool(st.session_state.get("llm_api_key"))
    if "llm_model" not in st.session_state:
        st.session_state["llm_model"] = "gpt-4o-mini"


def _format_care_plan(band: str) -> List[str]:
    if band == "High":
        return [
            "Arrange immediate veterinary evaluation",
            "Stall rest and restrict movement",
            "Discuss NSAIDs strictly per veterinarian guidance",
            "Monitor digital pulse and hoof temperature frequently",
            "Pad/bed the stall for comfort; avoid hard ground",
        ]
    if band == "Moderate":
        return [
            "Increase monitoring to twice daily",
            "Reduce non-structural carbohydrates in diet",
            "Ensure ample hydration and track intake",
            "Track lameness, respiratory rate, and rectal temperature",
        ]
    return [
        "Maintain routine care and turnout as advised",
        "Conduct weekly hoof checks",
        "Keep care records and weigh/BCS periodically",
        "Review warning signs with caretakers",
    ]


def _agent_followup_questions(band: str) -> List[str]:
    common = [
        "Any recent change in diet or access to lush pasture?",
        "Have you noticed heat in the hooves or stronger digital pulses?",
        "Any weight shifting, reluctance to move, or abnormal stance?",
    ]
    if band == "High":
        return [
            "Can you restrict movement safely right now?",
            "Is a veterinarian available for urgent evaluation today?",
        ] + common
    if band == "Moderate":
        return [
            "Can you increase monitoring to twice daily?",
            "Can you reduce grain/non-structural carbohydrates immediately?",
        ] + common
    return common + [
        "Do you have a schedule for weekly hoof checks?",
    ]


def agent_reply(user_text: Optional[str]) -> str:
    last_single = st.session_state.get("last_single")
    last_batch = st.session_state.get("last_batch")

    if user_text:
        lower = user_text.lower()
        if any(k in lower for k in ["dose", "dosage", "prescribe", "prescription", "medication", "nsaid"]):
            return (
                "I can’t provide medication prescriptions or dosing. Please consult a veterinarian. "
                "I can help with monitoring plans and care checklists in the meantime."
            )

    # Prefer single prediction context when available
    if last_single is not None:
        band = last_single.get("guidance_band", "Low")
        score = last_single.get("score", None)
        label = last_single.get("risk_label", None)
        plan = _format_care_plan(band)
        score_str = f" at {score:.2f}" if isinstance(score, (int, float)) else ""
        intro = f"Based on the latest prediction, risk is {label}{score_str} with guidance band {band}."
        if user_text:
            return intro + "\n\nHere’s a practical plan you can start now:\n- " + "\n- ".join(plan)
        return intro + "\n\nWould you like a step-by-step plan or follow-up questions?"

    # If no single, but batch exists, provide batch summary guidance
    if last_batch is not None:
        num_high = last_batch.get("num_high", 0)
        num_low = last_batch.get("num_low", 0)
        avg = last_batch.get("avg_score", None)
        intro = (
            f"Batch summary: High={num_high}, Low={num_low}, Average score={avg:.2f}"
            if isinstance(avg, (int, float)) else f"Batch summary: High={num_high}, Low={num_low}"
        )
        return intro + "\nConsider prioritizing high-risk horses for immediate evaluation and monitoring."

    # No context yet
    return (
        "I don’t see a recent prediction yet. Run a single or batch prediction, or tell me the horse’s "
        "current signs and I’ll propose a monitoring plan."
    )


def llm_reply(user_text: str) -> str:
    """LLM-based assistant that avoids sending raw measurement data.

    Only uses high-level context (risk label, band, average score) and natural-language user text.
    """
    try:
        from openai import OpenAI
    except Exception:
        return (
            "LLM mode is enabled but the OpenAI SDK is not installed. Please install 'openai' and retry, "
            "or switch back to the local assistant."
        )

    api_key = st.session_state.get("llm_api_key", "")
    if not api_key:
        return "LLM mode is enabled, but no API key is set. Please add your key in the sidebar."

    client = OpenAI(api_key=api_key)

    last_single = st.session_state.get("last_single") or {}
    last_batch = st.session_state.get("last_batch") or {}

    # Build privacy-preserving context
    context_summary = {
        "single": {
            "risk_label": last_single.get("risk_label"),
            "guidance_band": last_single.get("guidance_band"),
            # Provide score rounded only, no raw features
            "score": round(float(last_single.get("score", 0.0)), 2) if last_single.get("score") is not None else None,
        },
        "batch": {
            "num_high": last_batch.get("num_high"),
            "num_low": last_batch.get("num_low"),
            "avg_score": round(float(last_batch.get("avg_score", 0.0)), 2) if last_batch.get("avg_score") is not None else None,
        }
    }

    system_prompt = (
        "You are LaminitisCare, an equine health assistant. Follow these rules:\n"
        "- Do not provide medical prescriptions or dosing; advise consulting a veterinarian.\n"
        "- Use only the provided high-level context (risk label, band, rounded scores).\n"
        "- Do not request or reproduce raw measurements.\n"
        "- Provide clear, actionable steps tailored to the risk band.\n"
        "High risk → immediate vet evaluation, stall rest, restricted movement, NSAIDs per vet guidance, frequent monitoring of digital pulse and hoof temperature.\n"
        "Moderate risk → increase monitoring (twice daily), reduce non-structural carbohydrates, ensure hydration, track lameness/respiratory rate/rectal temperature.\n"
        "Low risk → maintain routine care, weekly hoof checks, keep records, educate caretakers on warning signs."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": f"Context: {context_summary}"},
        {"role": "user", "content": user_text or "Provide a concise care plan."},
    ]

    try:
        resp = client.chat.completions.create(
            model=st.session_state.get("llm_model", "gpt-4o-mini"),
            messages=messages,
            temperature=0.4,
            max_tokens=400,
        )
        text = resp.choices[0].message.content.strip()
        return text
    except Exception as e:
        return f"LLM request failed: {e}"


def main():
    st.title("LaminitisCare: Laminitis Risk Assistant")
    st.caption(
        "Privacy-first: all predictions are computed locally. No data is sent outside this app."
    )

    # Load artifacts
    try:
        artifacts = load_artifacts()
    except Exception as e:
        st.error(f"Failed to load artifacts: {e}")
        st.stop()

    feature_order: List[str] = artifacts["feature_order"]

    # Sidebar Info
    with st.sidebar:
        st.header("Model Artifacts")
        st.write(f"Features: {len(feature_order)}")
        st.write("Classifier: SVM")
        st.write("Preprocessing: Scaler applied")
        st.divider()
        _ensure_agent_state()
        if st.session_state.get("llm_enabled"):
            st.caption("Agent: LLM mode is active (key detected).")
        else:
            st.caption("Agent: Local mode is active (no API key detected).")
        st.markdown("If you need medication dosing or prescriptions, please consult a veterinarian.")

    tab_single, tab_agent = st.tabs(["Single Prediction", "Agent Assistant"]) 

    # -------------------------
    # Single Prediction Tab
    # -------------------------
    with tab_single:
        st.subheader("Enter Measurements")
        st.caption("Unknown inputs are ignored. Missing features will be imputed from training averages.")

        with st.form("single_form"):
            cols = st.columns(2)
            single_input: Dict[str, float] = {}
            for i, feat in enumerate(feature_order):
                with cols[i % 2]:
                    schema = FEATURE_SCHEMAS.get(feat)
                    if schema and schema.get("type") == "int":
                        min_v = int(schema.get("min", 0))
                        max_v = int(schema.get("max", 10**6))
                        step_v = int(schema.get("step", 1))
                        default_v = min_v
                        single_input[feat] = st.number_input(
                            label=feat,
                            min_value=min_v,
                            max_value=max_v,
                            value=default_v,
                            step=step_v,
                            help=f"Integer range: {min_v} to {max_v}"
                        )
                    elif schema and schema.get("type") == "float":
                        min_v = float(schema.get("min", 0.0))
                        max_v = float(schema.get("max", 10**6))
                        step_v = float(schema.get("step", 0.1))
                        default_v = min_v
                        single_input[feat] = st.number_input(
                            label=feat,
                            min_value=min_v,
                            max_value=max_v,
                            value=default_v,
                            step=step_v,
                            format="%f",
                            help=f"Range: {min_v} to {max_v}"
                        )
                    else:
                        single_input[feat] = st.number_input(
                            label=feat,
                            value=0.0,
                            step=0.1,
                            format="%f",
                            help="Enter numeric value"
                        )
            submitted = st.form_submit_button("Predict Risk")

        if submitted:
            row_df = pd.DataFrame([single_input])
            # Coerce and clip to schema (defensive)
            row_df, adj = coerce_and_clip_to_schema(row_df, FEATURE_SCHEMAS)
            result, warnings = predict_dataframe(row_df, artifacts)

            missing, extra, method = warnings["missing"], warnings["extra"], warnings["score_method"]
            if extra:
                st.warning(
                    "Ignored unknown inputs: " + ", ".join(extra)
                )
            if missing:
                st.warning(
                    "Missing features imputed from training statistics: " + ", ".join(missing)
                )

            score_col = "probability" if method == "probability" else ("confidence" if method == "confidence" else "score")
            score_val = float(result.iloc[0][score_col])
            label_val = str(result.iloc[0]["risk_label"])
            guidance_band = str(result.iloc[0]["guidance_band"])

            st.markdown("---")
            c1, c2, c3 = st.columns([1, 1, 2])
            with c1:
                st.metric("Risk Label (primary)", label_val)
            with c2:
                st.metric(
                    "Score",
                    f"{score_val:.2f}",
                    help=(
                        "Probability" if method == "probability" else
                        ("Confidence (uncalibrated)" if method == "confidence" else "Model score")
                    ),
                )
            with c3:
                st.metric("Guidance Band", guidance_band)

            st.write(explain_prediction_plain(score_val, method))
            render_guidance(guidance_band)
            render_shap_image(artifacts["base_dir"])  # context only

            # Update agent context
            _ensure_agent_state()
            st.session_state["last_single"] = {
                "risk_label": label_val,
                "score": score_val,
                "guidance_band": guidance_band,
                "timestamp": datetime.utcnow().isoformat(),
            }
            st.session_state["agent_history"].append({
                "role": "system",
                "content": f"Single prediction updated: label={label_val}, score={score_val:.2f}, band={guidance_band}"
            })
            if adj:
                st.info("Inputs were validated and constrained to expected ranges/types.")

    # Batch Prediction tab removed per request

    # -------------------------
    # Agent Assistant Tab
    # -------------------------
    with tab_agent:
        _ensure_agent_state()
        st.subheader("Agent Assistant")
        st.caption("Local, privacy-first assistant that turns predictions into actionable steps.")

        # Show context snapshot
        with st.expander("Latest context", expanded=False):
            last_single = st.session_state.get("last_single")
            last_batch = st.session_state.get("last_batch")
            if last_single:
                st.write({k: last_single[k] for k in ["risk_label", "score", "guidance_band", "timestamp"] if k in last_single})
            if last_batch:
                st.write({k: last_batch[k] for k in ["num_high", "num_low", "avg_score", "timestamp"] if k in last_batch})
            if not last_single and not last_batch:
                st.write("No prediction context yet.")

        # Show conversation
        history = st.session_state.get("agent_history", [])
        for msg in history[-30:]:
            role = msg.get("role", "assistant")
            content = msg.get("content", "")
            st.chat_message(role).write(content)

        user_input = st.chat_input("Ask about care steps, monitoring, or next actions…")
        if user_input:
            st.session_state["agent_history"].append({"role": "user", "content": user_input})
            if st.session_state.get("llm_enabled", False):
                reply = llm_reply(user_input)
            else:
                reply = agent_reply(user_input)
            st.session_state["agent_history"].append({"role": "assistant", "content": reply})
            st.chat_message("assistant").write(reply)

            # Offer follow-up questions based on latest single-band
            last_single = st.session_state.get("last_single")
            if last_single:
                band = last_single.get("guidance_band", "Low")
                qs = _agent_followup_questions(band)
                st.markdown("\n".join(["- " + q for q in qs]))

    st.markdown("---")
    st.caption(
        "Limitations: Predictions assist but do not replace veterinary judgment. This tool does not "
        "provide medical prescriptions. For any treatment decisions, consult a veterinarian."
    )


if __name__ == "__main__":
    main()


