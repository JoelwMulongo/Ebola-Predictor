"""Ebola-Predictor: Streamlit screening demo (Kenya).

Run:  streamlit run app.py
Educational demo trained on SIMULATED data. Not a medical device or diagnosis.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import pandas as pd
import sklearn
import streamlit as st

from common import COUNTY_POPULATION, RAW_INPUT_COLUMNS, screening_rule, validate_input

MODEL_PATH = Path(__file__).with_name("ebola_detector.joblib")

st.set_page_config(page_title="Ebola-Predictor (Kenya)", page_icon="🩺", layout="centered")


@st.cache_resource
def load_artifact(path: str):
    return joblib.load(path)


def tier_for(prob: float, thr: float, high: float, rule_flag: bool) -> str:
    tier = "High" if prob >= high else "Elevated" if prob >= thr else "Low"
    if rule_flag and tier == "Low":
        tier = "Elevated"  # safety net overrides a low model score
    return tier


st.title("🩺 Ebola-Predictor (Kenya)")
st.error(
    "**Educational demo only.** This model was trained on *simulated* data and has not been "
    "clinically validated. It cannot diagnose Ebola. If you have fever plus a possible exposure, "
    "isolate from others, call the Ministry of Health toll-free line 719 and go to the nearest "
    "health facility (tell staff about your exposure before you arrive)."
)

if not MODEL_PATH.exists():
    st.warning("Model file not found. Run `python generate_data.py` then `python train.py` first.")
    st.stop()

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    art = load_artifact(str(MODEL_PATH))

if art.get("sklearn_version") != sklearn.__version__:
    st.warning(f"Model was trained with scikit-learn {art.get('sklearn_version')} but you have "
               f"{sklearn.__version__}. Retrain with `python train.py` if you see errors.")

with st.form("patient"):
    st.subheader("Patient details")
    c1, c2, c3 = st.columns(3)
    age = c1.number_input("Age (years)", 0, 120, 30)
    sex = c2.selectbox("Sex", ["Female", "Male"])
    county = c3.selectbox("County", sorted(COUNTY_POPULATION), index=sorted(COUNTY_POPULATION).index("Nairobi"))

    c4, c5 = st.columns(2)
    temp_known = c4.checkbox("Temperature measured", value=True)
    temp = c4.number_input("Temperature (°C)", 32.0, 45.0, 36.8, 0.1, disabled=not temp_known)
    days = c5.number_input("Days since symptoms began (0 = none)", 0, 60, 0)

    st.subheader("Symptoms")
    labels = {
        "fever": "Fever", "headache": "Severe headache", "fatigue": "Extreme fatigue",
        "muscle_pain": "Muscle pain", "vomiting": "Vomiting", "diarrhea": "Diarrhoea",
        "abdominal_pain": "Abdominal pain", "sore_throat": "Sore throat", "rash": "Rash",
        "red_eyes": "Red eyes", "hiccups": "Hiccups", "unexplained_bleeding": "Unexplained bleeding/bruising",
    }
    cols = st.columns(3)
    sym = {k: int(cols[i % 3].checkbox(v)) for i, (k, v) in enumerate(labels.items())}

    st.subheader("Exposures (last 21 days)")
    exp_labels = {
        "contact_with_case": "Contact with a confirmed/probable case",
        "funeral_attendance": "Attended a funeral / handled a body",
        "travel_outbreak_area": "Travelled to an outbreak-affected area",
        "bushmeat_exposure": "Handled/ate bushmeat or found dead animals",
        "healthcare_worker": "Works in a healthcare setting",
    }
    exp = {k: int(st.checkbox(v)) for k, v in exp_labels.items()}
    submitted = st.form_submit_button("Assess risk", type="primary")

if submitted:
    row = {"age": age, "sex": sex, "county": county,
           "temperature_c": temp if temp_known else float("nan"),
           "days_since_onset": days, **sym, **exp}
    frame = pd.DataFrame([row])[RAW_INPUT_COLUMNS]
    problems = validate_input(frame)
    if problems:
        st.error("Please fix: " + "; ".join(problems))
        st.stop()

    prob = float(art["model"].predict_proba(frame)[0, 1])
    rule_flag = screening_rule(frame.iloc[0])
    tier = tier_for(prob, art["threshold"], art["high_risk_cutoff"], rule_flag)

    st.divider()
    m1, m2 = st.columns(2)
    m1.metric("Estimated risk score", f"{prob:.1%}")
    m2.metric("Screening tier", tier)

    if tier == "High":
        st.error("High risk. Isolate, avoid contact with others, and contact a health facility or "
                 "county public health office immediately. Do not travel by public transport.")
    elif tier == "Elevated":
        st.warning("Elevated risk. Seek assessment at a health facility soon, limit contact with "
                   "others, and tell staff about your exposures before you arrive.")
    else:
        st.success("Low risk on this screen. If symptoms appear or worsen, or you develop a new "
                   "exposure, seek care and re-screen.")
    if rule_flag:
        st.info("Safety rule triggered: fever or bleeding together with a high-risk exposure is "
                "always treated as at least Elevated, regardless of the model score.")
    if not any(sym.values()) and not any(exp.values()):
        st.caption("No symptoms or exposures selected.")

with st.expander("About this model"):
    t = art["metrics"]["test"]
    st.write(
        f"**Algorithm:** {art['model_name'].replace('_', ' ')} (probability-calibrated) · "
        f"**Trained:** {art['trained_at']}  \n"
        f"**Held-out test (simulated data):** recall {t['recall']:.1%}, specificity {t['specificity']:.1%}, "
        f"precision {t['precision']:.1%}, ROC-AUC {t['roc_auc']:.3f}.  \n"
        "Precision is modest because true cases are rare. Most flagged people will not have Ebola, "
        "which is why a flag means *get assessed*, not *you have Ebola*. "
        "These figures describe simulated data and say nothing about real-world accuracy."
    )
    st.bar_chart(pd.DataFrame(art["metrics"]["top_features"]).set_index("feature")["importance"])
