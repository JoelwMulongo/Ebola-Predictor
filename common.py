"""Shared schema, constants and feature engineering for Ebola-Predictor.

Everything that must stay identical between data generation, training and the
Streamlit app lives here, so the three can never drift apart.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "ebola_case"

# Approximate 2019 census population by county, in millions. Used only to
# weight how often each county appears in the simulated data.
COUNTY_POPULATION = {
    "Mombasa": 1.21, "Kwale": 0.87, "Kilifi": 1.45, "Tana River": 0.32,
    "Lamu": 0.14, "Taita Taveta": 0.34, "Garissa": 0.84, "Wajir": 0.78,
    "Mandera": 0.87, "Marsabit": 0.46, "Isiolo": 0.27, "Meru": 1.55,
    "Tharaka-Nithi": 0.39, "Embu": 0.61, "Kitui": 1.14, "Machakos": 1.42,
    "Makueni": 0.99, "Nyandarua": 0.64, "Nyeri": 0.76, "Kirinyaga": 0.61,
    "Murang'a": 1.06, "Kiambu": 2.42, "Turkana": 0.93, "West Pokot": 0.62,
    "Samburu": 0.31, "Trans Nzoia": 0.99, "Uasin Gishu": 1.16,
    "Elgeyo-Marakwet": 0.45, "Nandi": 0.88, "Baringo": 0.67, "Laikipia": 0.52,
    "Nakuru": 2.16, "Narok": 1.16, "Kajiado": 1.12, "Kericho": 0.90,
    "Bomet": 0.88, "Kakamega": 1.87, "Vihiga": 0.59, "Bungoma": 1.68,
    "Busia": 0.89, "Siaya": 0.99, "Kisumu": 1.16, "Homa Bay": 1.13,
    "Migori": 1.12, "Kisii": 1.27, "Nyamira": 0.60, "Nairobi": 4.40,
}

# Counties that share a land border with Uganda, Tanzania, South Sudan,
# Ethiopia or Somalia (higher cross-border exposure risk).
BORDER_COUNTIES = {
    "Busia", "Bungoma", "Trans Nzoia", "West Pokot", "Turkana", "Marsabit",
    "Mandera", "Wajir", "Garissa", "Lamu", "Kwale", "Taita Taveta",
    "Kajiado", "Narok", "Migori",
}

SYMPTOMS = [
    "fever", "headache", "fatigue", "muscle_pain", "vomiting", "diarrhea",
    "abdominal_pain", "sore_throat", "rash", "red_eyes", "hiccups",
    "unexplained_bleeding",
]
EXPOSURES = [
    "contact_with_case",      # direct contact with a confirmed/probable case
    "funeral_attendance",     # attended a funeral / handled a body
    "travel_outbreak_area",   # recent travel to an outbreak-affected area
    "bushmeat_exposure",      # handled/ate bushmeat or found dead animals
    "healthcare_worker",      # works in a healthcare setting
]
NUMERIC_RAW = ["age", "temperature_c", "days_since_onset"]
CATEGORICAL = ["sex"]

# Columns a caller must supply (what the saved pipeline accepts).
RAW_INPUT_COLUMNS = ["age", "sex", "county", "temperature_c", "days_since_onset"] + SYMPTOMS + EXPOSURES

# Columns that exist after feature engineering and feed the model.
ENGINEERED = ["border_county", "n_symptoms", "high_risk_exposure"]
MODEL_NUMERIC = NUMERIC_RAW + ["n_symptoms"]
MODEL_BINARY = SYMPTOMS + EXPOSURES + ["border_county", "high_risk_exposure"]

# Plausible ranges used for input validation.
VALID_RANGES = {"age": (0, 120), "temperature_c": (32.0, 45.0), "days_since_onset": (0, 60)}


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features. Tolerates missing values (NaN) in any input."""
    missing = [c for c in RAW_INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    out = df.copy()
    # Force a consistent float dtype. Training data (from CSV, with blanks) is float, while a
    # single row typed into the app is int; the imputers fail if the two differ.
    for col in NUMERIC_RAW + SYMPTOMS + EXPOSURES:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    out["border_county"] = out["county"].isin(BORDER_COUNTIES).astype("float64")
    out["n_symptoms"] = out[SYMPTOMS].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
    risky = ["contact_with_case", "funeral_attendance", "travel_outbreak_area"]
    out["high_risk_exposure"] = (
        out[risky].apply(pd.to_numeric, errors="coerce").fillna(0).max(axis=1).astype("float64")
    )
    return out


def validate_input(df: pd.DataFrame) -> list[str]:
    """Return a list of human-readable problems (empty list = valid)."""
    problems = [f"Missing column: {c}" for c in RAW_INPUT_COLUMNS if c not in df.columns]
    if problems:
        return problems
    for col, (lo, hi) in VALID_RANGES.items():
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if ((vals < lo) | (vals > hi)).any():
            problems.append(f"{col} must be between {lo} and {hi}")
    for col in SYMPTOMS + EXPOSURES:
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if (~vals.isin([0, 1])).any():
            problems.append(f"{col} must be 0 or 1")
    bad_county = set(df["county"].dropna()) - set(COUNTY_POPULATION)
    if bad_county:
        problems.append(f"Unknown county: {sorted(bad_county)}")
    return problems


def screening_rule(row: pd.Series) -> bool:
    """Conservative safety net, independent of the ML model.

    Flags a person for screening when they have fever AND a high-risk exposure,
    or unexplained bleeding with any exposure. Inspired by the logic of standard
    suspected-case screening; it is NOT the official case definition.
    """
    g = lambda k: float(np.nan_to_num(pd.to_numeric(row.get(k, 0), errors="coerce"), nan=0.0))
    high_risk = max(g("contact_with_case"), g("funeral_attendance"), g("travel_outbreak_area"))
    any_exposure = max(high_risk, g("bushmeat_exposure"))
    return bool((g("fever") and high_risk) or (g("unexplained_bleeding") and any_exposure))
