# Ebola-Predictor (Kenya)

A symptom- and exposure-based **Ebola screening tool** for Kenya, built as a Streamlit app on top of a
calibrated scikit-learn pipeline. It is the companion to [COVID-19-PREDICTOR](https://github.com/mulongocheloti/COVID-19-PREDICTOR),
rebuilt with a more rigorous evaluation setup for a rare-outcome problem.

> **Genesis.** Kenya confirmed its first-ever Ebola case on 6 October 2026: a single *imported*
> case (Bundibugyo virus, travelled from DRC via Uganda; the patient died). Kenya has still not had a
> domestic outbreak, so there is no real case-level Kenyan dataset. `ebola_kenya.csv` is therefore
> **simulated**: real Kenyan geography (47 counties, population-weighted, 15 border counties flagged),
> symptom frequencies loosely based on published clinical descriptions of Ebola disease (mostly from
> Zaire-ebolavirus outbreaks, not Bundibugyo-specific), overlap from other febrile illnesses (malaria,
> typhoid, flu), milder presentations, and missing values. The 4% prevalence is deliberately enriched.
> **Model metrics describe the simulation, not real-world accuracy, and this is not a medical device or
> diagnostic tool.** For real concerns call the Ministry of Health toll-free line 719 or go to a health facility.

## Project layout

| File | Purpose |
|---|---|
| `common.py` | Single source of truth: schema, counties, feature engineering, input validation, safety rule |
| `generate_data.py` | Builds `ebola_kenya.csv` (seeded and reproducible) |
| `train.py` | Compares models, calibrates, picks a threshold, evaluates, saves `ebola_detector.joblib` + `metrics.json` |
| `app.py` | Streamlit UI |
| `tests/test_pipeline.py` | Tests for data, validation, training, missing-value handling |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python train.py --target-recall 0.95
pytest -q
streamlit run app.py
```


## Example results (simulated data, seed 42)

Logistic regression was selected (CV PR-AUC ≈ 0.85 vs a 0.04 baseline; it edges out the tree models because the
simulated risk is roughly additive). On the held-out test set at the chosen threshold: recall ≈ 95%
(95% CI 92-97%), specificity ≈ 94%, precision ≈ 40%, ROC-AUC ≈ 0.98. Top drivers: contact with a case,
diarrhoea, funeral attendance, vomiting, travel to an outbreak area.
Precision is modest because cases are rare, so most flags are not Ebola. A flag means *get assessed*.

## Using real data

Replace `ebola_kenya.csv` with a real line list that has the same columns (see `RAW_INPUT_COLUMNS` and
`TARGET` in `common.py`), then run `python train.py`. Candidate sources are WHO and Ministry of Health
outbreak line lists (for example from the West Africa or DRC/Uganda outbreaks), but check their licences
and consent terms. Before any real use you would need external validation, ethics review and
clinician input.

## Limitations

- Simulated data: the model largely rediscovers the assumptions in `generate_data.py`.
- Symptoms of Ebola overlap heavily with common illnesses, so symptom checkers have inherently low precision.
- No temporal or outbreak-size dynamics; county is used only through the border-county flag.
- For anything real, follow Kenya Ministry of Health and WHO case definitions and reporting procedures.
