from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "gradient_boosting_cto_hf.joblib"
META_PATH = APP_DIR / "model_metadata.json"

FEATURE_COLUMNS = ["Hb", "Crea", "NT_ProBNP", "NYHA", "Vascular_recanalization"]

FEATURE_LABELS = {
    "Hb": "Hemoglobin, Hb (g/L)",
    "Crea": "Creatinine, Crea (umol/L)",
    "NT_ProBNP": "NT-proBNP (pg/mL)",
    "NYHA": "NYHA functional class",
    "Vascular_recanalization": "Vascular recanalization",
}

NYHA_OPTIONS = {
    "Class I": 1,
    "Class II": 2,
    "Class III": 3,
    "Class IV": 4,
}

RECANALIZATION_OPTIONS = {
    "No": 0,
    "Yes": 1,
}


st.set_page_config(
    page_title="CTO-HF In-hospital Mortality Risk Calculator",
    page_icon="CTO",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 2.5rem;
    }
    .stApp {
        background: #f7f8fa;
    }
    [data-testid="stHeader"] {
        background: rgba(247, 248, 250, 0.92);
    }
    .tool-title {
        font-family: Georgia, 'Times New Roman', serif;
        font-size: 2.1rem;
        line-height: 1.18;
        color: #152033;
        margin-bottom: 0.15rem;
    }
    .tool-subtitle {
        color: #5a6472;
        font-size: 0.98rem;
        margin-bottom: 1.25rem;
    }
    .panel {
        border: 1px solid #d9dde5;
        background: #ffffff;
        border-radius: 8px;
        padding: 1rem 1.1rem;
    }
    .risk-card {
        border: 1px solid #cfd6e2;
        border-left: 6px solid var(--risk-color);
        background: #ffffff;
        border-radius: 8px;
        padding: 1rem 1.1rem;
        margin-top: 0.25rem;
    }
    .risk-value {
        color: #152033;
        font-family: Georgia, 'Times New Roman', serif;
        font-size: 2.4rem;
        line-height: 1;
        margin: 0.15rem 0 0.45rem 0;
    }
    .risk-label {
        color: var(--risk-color);
        font-weight: 700;
        font-size: 1rem;
    }
    .small-muted {
        color: #667085;
        font-size: 0.88rem;
    }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #d9dde5;
        border-radius: 8px;
        padding: 0.85rem 0.9rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.45rem;
    }
    .stTabs [data-baseweb="tab"] {
        border: 1px solid #d9dde5;
        border-radius: 7px;
        padding: 0.55rem 0.9rem;
        background: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_metadata() -> dict:
    with META_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def coerce_input_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [feature for feature in FEATURE_COLUMNS if feature not in frame.columns]
    if missing:
        raise ValueError("Missing required predictors: " + ", ".join(missing))

    clean = frame.loc[:, FEATURE_COLUMNS].copy()
    for feature in FEATURE_COLUMNS:
        clean[feature] = pd.to_numeric(clean[feature], errors="coerce")
    if clean.isna().any().any():
        bad = clean.columns[clean.isna().any()].tolist()
        raise ValueError("Non-numeric or missing values found in: " + ", ".join(bad))

    clean["NYHA"] = clean["NYHA"].clip(1, 4).round().astype(int)
    clean["Vascular_recanalization"] = clean["Vascular_recanalization"].clip(0, 1).round().astype(int)
    return clean


def predict_risk(model, frame: pd.DataFrame) -> np.ndarray:
    clean = coerce_input_frame(frame)
    return model.predict_proba(clean)[:, 1]


def risk_category(risk: float, threshold: float) -> tuple[str, str]:
    if risk >= threshold:
        return "Model-positive high-risk", "#b42318"
    if risk >= threshold * 0.5:
        return "Intermediate risk", "#b54708"
    return "Lower risk", "#067647"


def read_uploaded_table(uploaded_file) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(uploaded_file)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(uploaded_file)
    raise ValueError("Only CSV and Excel files are supported.")


def sample_input(metadata: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Hb": metadata["feature_stats"]["Hb"]["median"],
                "Crea": metadata["feature_stats"]["Crea"]["median"],
                "NT_ProBNP": metadata["feature_stats"]["NT_ProBNP"]["median"],
                "NYHA": 3,
                "Vascular_recanalization": 1,
            }
        ]
    )


model = load_model()
metadata = load_metadata()
threshold = float(metadata["threshold"])

st.markdown('<div class="tool-title">CTO-HF In-hospital Mortality Risk Calculator</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="tool-subtitle">Gradient boosting model for patients with chronic total occlusion and heart failure.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Model")
    st.write("Outcome: in-hospital death")
    st.write("Algorithm: Gradient boosting")
    st.write(f"Fixed threshold: {threshold:.3f}")
    st.markdown("### Predictors")
    for feature in FEATURE_COLUMNS:
        st.write(FEATURE_LABELS[feature])
    st.markdown("### Validation")
    performance = pd.DataFrame(metadata["performance"])
    st.dataframe(performance, width="stretch", hide_index=True)

metric_1, metric_2, metric_3 = st.columns(3)
metric_1.metric("Training set", f'{metadata["sample_size"]["training"]:,}')
metric_2.metric("Internal validation set", f'{metadata["sample_size"]["internal_validation"]:,}')
metric_3.metric("External validation set", f'{metadata["sample_size"]["external_validation"]:,}')

tab_single, tab_batch, tab_about = st.tabs(["Single patient", "Batch prediction", "Model details"])

with tab_single:
    left, right = st.columns([1.05, 0.95], gap="large")
    stats = metadata["feature_stats"]
    with left:
        st.markdown("#### Patient profile")
        with st.form("single_prediction"):
            hb = st.number_input(
                FEATURE_LABELS["Hb"],
                min_value=40.0,
                max_value=220.0,
                value=float(stats["Hb"]["median"]),
                step=1.0,
                format="%.1f",
            )
            crea = st.number_input(
                FEATURE_LABELS["Crea"],
                min_value=10.0,
                max_value=2000.0,
                value=float(stats["Crea"]["median"]),
                step=1.0,
                format="%.1f",
            )
            nt_probnp = st.number_input(
                FEATURE_LABELS["NT_ProBNP"],
                min_value=0.0,
                max_value=50000.0,
                value=float(stats["NT_ProBNP"]["median"]),
                step=50.0,
                format="%.0f",
            )
            nyha_label = st.selectbox(FEATURE_LABELS["NYHA"], list(NYHA_OPTIONS.keys()), index=2)
            rec_label = st.selectbox(FEATURE_LABELS["Vascular_recanalization"], list(RECANALIZATION_OPTIONS.keys()), index=1)
            submitted = st.form_submit_button("Calculate risk", width="stretch")

    input_df = pd.DataFrame(
        [
            {
                "Hb": hb,
                "Crea": crea,
                "NT_ProBNP": nt_probnp,
                "NYHA": NYHA_OPTIONS[nyha_label],
                "Vascular_recanalization": RECANALIZATION_OPTIONS[rec_label],
            }
        ]
    )
    risk = float(predict_risk(model, input_df)[0])
    category, color = risk_category(risk, threshold)
    decision = "Positive" if risk >= threshold else "Negative"

    with right:
        st.markdown("#### Prediction")
        st.markdown(
            f"""
            <div class="risk-card" style="--risk-color:{color};">
                <div class="small-muted">Predicted probability of in-hospital death</div>
                <div class="risk-value">{risk * 100:.2f}%</div>
                <div class="risk-label">{category}</div>
                <div class="small-muted">Model classification at threshold {threshold:.3f}: {decision}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(min(max(risk / max(threshold * 2, 0.001), 0.0), 1.0))
        result = input_df.copy()
        result["PredictedRisk"] = risk
        result["RiskPercent"] = risk * 100
        result["ModelClassification"] = decision
        st.dataframe(result, width="stretch", hide_index=True)
        csv = result.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "Download this prediction",
            data=csv,
            file_name="cto_hf_single_prediction.csv",
            mime="text/csv",
            width="stretch",
        )

with tab_batch:
    st.markdown("#### Batch prediction")
    sample = sample_input(metadata)
    st.download_button(
        "Download input template",
        data=sample.to_csv(index=False).encode("utf-8-sig"),
        file_name="cto_hf_prediction_template.csv",
        mime="text/csv",
    )
    uploaded = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])
    if uploaded is not None:
        try:
            batch = read_uploaded_table(uploaded)
            clean_batch = coerce_input_frame(batch)
            risks = predict_risk(model, clean_batch)
            output = batch.copy()
            output["PredictedRisk"] = risks
            output["RiskPercent"] = risks * 100
            output["ModelClassification"] = np.where(risks >= threshold, "Positive", "Negative")
            output["RiskCategory"] = [risk_category(float(r), threshold)[0] for r in risks]
            st.success(f"Prediction completed for {len(output):,} rows.")
            st.dataframe(output, width="stretch", hide_index=True)
            st.download_button(
                "Download batch predictions",
                data=output.to_csv(index=False).encode("utf-8-sig"),
                file_name="cto_hf_batch_predictions.csv",
                mime="text/csv",
                width="stretch",
            )
        except Exception as exc:
            st.error(str(exc))

with tab_about:
    st.markdown("#### Model specification")
    st.write("Final predictors were selected from the training set and externally validated using an independent cohort.")
    st.dataframe(pd.DataFrame(metadata["performance"]), width="stretch", hide_index=True)
    st.markdown("#### Required input columns")
    st.dataframe(pd.DataFrame({"Column": FEATURE_COLUMNS, "Display name": [FEATURE_LABELS[x] for x in FEATURE_COLUMNS]}), hide_index=True)
    st.info(
        "This calculator is intended for research presentation of the published model. "
        "It is not a standalone clinical decision system and should not replace clinician judgement."
    )
