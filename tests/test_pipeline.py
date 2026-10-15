import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import (COUNTY_POPULATION, RAW_INPUT_COLUMNS, SYMPTOMS, TARGET,  # noqa: E402
                    add_features, screening_rule, validate_input)
from generate_data import generate  # noqa: E402
from train import train  # noqa: E402


@pytest.fixture(scope="module")
def df():
    return generate(n=6000, prevalence=0.05, seed=1)


def test_county_list_is_complete():
    assert len(COUNTY_POPULATION) == 47


def test_generated_schema_and_prevalence(df):
    assert set(RAW_INPUT_COLUMNS + [TARGET]).issubset(df.columns)
    assert 0.03 < df[TARGET].mean() < 0.07
    assert df["temperature_c"].isna().any()  # missingness exercised
    assert validate_input(df.dropna(subset=["age"])) == []


def test_generation_is_reproducible():
    a, b = generate(500, seed=7), generate(500, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_add_features_handles_nan(df):
    out = add_features(df.head(50))
    assert {"border_county", "n_symptoms", "high_risk_exposure"}.issubset(out.columns)
    assert not out[["border_county", "n_symptoms", "high_risk_exposure"]].isna().any().any()


def test_add_features_requires_columns(df):
    with pytest.raises(ValueError):
        add_features(df.drop(columns=["fever"]))


def test_validation_rejects_bad_values(df):
    bad = df.head(3).copy()
    bad.loc[bad.index[0], "age"] = 500
    bad.loc[bad.index[1], "fever"] = 2
    bad.loc[bad.index[2], "county"] = "Atlantis"
    assert len(validate_input(bad)) == 3


def test_screening_rule():
    base = {k: 0 for k in RAW_INPUT_COLUMNS}
    assert not screening_rule(pd.Series(base))
    assert screening_rule(pd.Series({**base, "fever": 1, "contact_with_case": 1}))
    assert screening_rule(pd.Series({**base, "unexplained_bleeding": 1, "bushmeat_exposure": 1}))
    assert not screening_rule(pd.Series({**base, "fever": 1}))


def test_train_save_and_predict(df, tmp_path):
    out = tmp_path / "model.joblib"
    art = train(df, out=str(out), metrics_out=None,
                candidates={"logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced")},
                verbose=False)
    loaded = joblib.load(out)
    sample = df[RAW_INPUT_COLUMNS].head(20)
    p = loaded["model"].predict_proba(sample)[:, 1]
    assert p.shape == (20,) and np.all((p >= 0) & (p <= 1))
    assert 0 < art["threshold"] < 1
    assert art["metrics"]["test"]["recall"] > 0.7


def test_model_handles_missing_inputs(df, tmp_path):
    art = train(df, out=str(tmp_path / "m.joblib"), metrics_out=None,
                candidates={"logistic_regression": LogisticRegression(max_iter=2000)}, verbose=False)
    row = df[RAW_INPUT_COLUMNS].head(1).copy()
    row[["age", "temperature_c", "contact_with_case"]] = np.nan
    assert 0 <= art["model"].predict_proba(row)[0, 1] <= 1


def test_train_rejects_too_few_positives(df, tmp_path):
    tiny = df[df[TARGET] == 0].head(500)
    with pytest.raises(ValueError):
        train(tiny, out=str(tmp_path / "x.joblib"), metrics_out=None, verbose=False)


def test_all_symptoms_are_binary(df):
    assert set(np.unique(df[SYMPTOMS].values)) <= {0, 1}
