"""Generate the simulated Kenya Ebola screening dataset (ebola_kenya.csv).

IMPORTANT: Kenya's only confirmed case (6 Oct 2026) was a single imported
Bundibugyo-virus case, not an outbreak, so no real Kenyan case-level dataset exists. This script simulates a *screening cohort*
(people assessed at health facilities and border points) using:
  * real Kenyan geography (47 counties, population-weighted, border counties
    flagged for higher cross-border exposure), and
  * symptom frequencies loosely based on published clinical descriptions of
    Ebola virus disease, with realistic overlap from malaria/typhoid-like
    febrile illness, plus missing values and milder presentations.
The prevalence is deliberately enriched (default 4%) so a model can learn; it is
NOT a real-world infection rate. Replace the CSV with real line-list data in the
same schema when available.

Usage:  python generate_data.py --n 40000 --prevalence 0.04 --seed 42
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common import BORDER_COUNTIES, COUNTY_POPULATION, EXPOSURES, SYMPTOMS, TARGET

# P(symptom | group). Order matches common.SYMPTOMS.
#                 fever headache fatigue muscle vomit diarr  abdo  sore  rash  eyes  hiccup bleed
P_CASE = np.array([.87,  .53,     .76,    .39,   .68,  .66,  .44,  .20,  .08,  .10,  .15,   .18])
P_FEBRILE = np.array([.88, .60,   .50,    .50,   .25,  .20,  .20,  .25,  .05,  .04,  .01,   .005])
P_WELL = np.array([.03,  .10,     .10,    .08,   .03,  .04,  .04,  .08,  .02,  .03,  .005,  .002])


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _solve_intercept(lin: np.ndarray, target: float) -> float:
    """Find the intercept so that mean(sigmoid(a + lin)) == target."""
    lo, hi = -20.0, 5.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if _sigmoid(mid + lin).mean() < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def generate(n: int = 40000, prevalence: float = 0.04, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    counties = list(COUNTY_POPULATION)
    w = np.array([COUNTY_POPULATION[c] for c in counties])
    county = rng.choice(counties, size=n, p=w / w.sum())
    border = np.isin(county, list(BORDER_COUNTIES))

    age = np.clip(rng.gamma(2.2, 14.0, n), 1, 95).round().astype(int)
    sex = rng.choice(["Female", "Male"], size=n)

    # Exposures (enriched vs. the general population: this is a screening cohort).
    def bern(p_border, p_other):
        return rng.random(n) < np.where(border, p_border, p_other)

    contact = bern(0.18, 0.10)
    funeral = bern(0.14, 0.08)
    travel = bern(0.25, 0.12)
    bushmeat = bern(0.16, 0.10)
    hcw = rng.random(n) < 0.06

    lin = (3.2 * contact + 2.2 * funeral + 2.0 * travel + 1.4 * bushmeat
           + 0.9 * hcw + 0.6 * border)
    a = _solve_intercept(lin, prevalence)
    case = rng.random(n) < _sigmoid(a + lin)

    febrile_other = (~case) & (rng.random(n) < 0.22)  # malaria, typhoid, flu...
    mild = case & (rng.random(n) < 0.12)               # milder / early presentation

    base = np.where(case[:, None], P_CASE, np.where(febrile_other[:, None], P_FEBRILE, P_WELL))
    base = np.where(mild[:, None], base * 0.6, base)
    sym = (rng.random((n, len(SYMPTOMS))) < base).astype(int)
    fever = sym[:, 0].astype(bool)

    temp = np.where(
        fever,
        np.clip(rng.normal(38.4, 0.7, n) + 0.2 * case, 37.6, 41.5),
        np.clip(rng.normal(36.8, 0.4, n), 35.5, 37.5),
    ).round(1)

    any_sym = sym.sum(axis=1) > 0
    # Same distribution for cases and non-cases, so it carries no label information.
    days = np.where(any_sym, rng.integers(1, 11, n), 0)

    df = pd.DataFrame({
        "patient_id": [f"KE-{i:06d}" for i in range(1, n + 1)],
        "age": age, "sex": sex, "county": county,
        "temperature_c": temp, "days_since_onset": days,
    })
    for j, name in enumerate(SYMPTOMS):
        df[name] = sym[:, j]
    for name, arr in zip(EXPOSURES, [contact, funeral, travel, bushmeat, hcw]):
        df[name] = arr.astype(int)
    df[TARGET] = case.astype(int)

    # Realistic missingness (kept off the target and the id).
    def punch(col, frac):
        df.loc[rng.random(n) < frac, col] = np.nan

    df["age"] = df["age"].astype(float)
    punch("temperature_c", 0.03)
    punch("age", 0.01)
    for col in EXPOSURES:
        df[col] = df[col].astype("Int8")
        punch(col, 0.02)
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=40000)
    p.add_argument("--prevalence", type=float, default=0.04)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="ebola_kenya.csv")
    args = p.parse_args()
    if not 0.005 <= args.prevalence <= 0.5:
        p.error("--prevalence must be between 0.005 and 0.5")
    df = generate(args.n, args.prevalence, args.seed)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} rows to {args.out} | positives: {df[TARGET].sum():,} "
          f"({df[TARGET].mean():.1%})")


if __name__ == "__main__":
    main()
