"""Train, calibrate and evaluate the Ebola screening model.

Pipeline (saved as ONE object, so the app feeds it raw columns):
  add_features -> impute/scale/encode -> classifier -> probability calibration

Design choices for a rare-outcome screening tool:
  * Stratified splits and CV (positives are rare).
  * Models compared on PR-AUC (average precision) and recall, not accuracy or MSE.
  * Class weighting during fitting, then Platt calibration so probabilities are
    meaningful risk estimates.
  * Decision threshold chosen from out-of-fold predictions on the TRAINING set to
    reach a target recall (default 95%), then reported honestly on a held-out test set.

Usage:  python train.py --data ebola_kenya.csv --target-recall 0.95
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, precision_recall_curve, roc_auc_score)
from sklearn.model_selection import (StratifiedKFold, cross_val_predict,
                                     cross_validate, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from common import (CATEGORICAL, MODEL_BINARY, MODEL_NUMERIC, RAW_INPUT_COLUMNS,
                    TARGET, add_features, validate_input)

HIGH_RISK_CUTOFF = 0.50  # calibrated probability at/above which risk is "High"


def build_pipeline(clf) -> Pipeline:
    prep = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), MODEL_NUMERIC),
        ("bin", SimpleImputer(strategy="constant", fill_value=0), MODEL_BINARY),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
    ])
    return Pipeline([("features", FunctionTransformer(add_features)), ("prep", prep), ("clf", clf)])


def default_candidates(seed: int) -> dict:
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                                class_weight="balanced_subsample",
                                                n_jobs=-1, random_state=seed),
        "hist_gradient_boosting": HistGradientBoostingClassifier(class_weight="balanced",
                                                                 max_depth=4, learning_rate=0.08,
                                                                 random_state=seed),
    }


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(float(c - h), 4), round(float(c + h), 4)]


def pick_threshold(y, p, target_recall: float) -> float:
    """Highest threshold that still achieves target recall (max precision at that recall)."""
    _, recall, thr = precision_recall_curve(y, p)
    ok = recall[:-1] >= target_recall
    return float(thr[ok].max()) if ok.any() else float(thr.min())


def evaluate(y, p, thr: float) -> dict:
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    div = lambda a, b: round(float(a / b), 4) if b else None
    return {
        "threshold": round(thr, 4),
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "recall": div(tp, tp + fn), "recall_95ci": wilson(int(tp), int(tp + fn)),
        "specificity": div(tn, tn + fp), "specificity_95ci": wilson(int(tn), int(tn + fp)),
        "precision": div(tp, tp + fp), "npv": div(tn, tn + fn),
        "prevalence": round(float(np.mean(y)), 4),
    }


def calibration_table(y, p, bins: int = 8) -> list[dict]:
    d = pd.DataFrame({"y": np.asarray(y), "p": p})
    d["bin"] = pd.qcut(d["p"], q=bins, duplicates="drop")
    g = d.groupby("bin", observed=True).agg(n=("y", "size"), predicted=("p", "mean"), observed=("y", "mean"))
    return [{"n": int(r.n), "predicted": round(float(r.predicted), 4), "observed": round(float(r.observed), 4)}
            for r in g.itertuples()]


def train(df: pd.DataFrame, target_recall: float = 0.95, seed: int = 42, out: str = "ebola_detector.joblib",
          metrics_out: str | None = "metrics.json", candidates: dict | None = None, verbose: bool = True) -> dict:
    log = print if verbose else (lambda *a, **k: None)

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found")
    df = df.dropna(subset=[TARGET]).copy()
    problems = validate_input(df)
    if problems:
        raise ValueError("Invalid data: " + "; ".join(problems))
    y = df[TARGET].astype(int)
    if y.nunique() < 2 or y.sum() < 50:
        raise ValueError("Need at least 50 positive cases and both classes to train reliably")
    X = df[RAW_INPUT_COLUMNS]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    log(f"Train {len(X_tr):,} / Test {len(X_te):,} | positives: train {y_tr.sum():,}, test {y_te.sum():,}")

    # 1) Model comparison with stratified CV on the training set only.
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    scoring = {"roc_auc": "roc_auc", "pr_auc": "average_precision", "recall": "recall", "precision": "precision"}
    cands = candidates or default_candidates(seed)
    cv_results = {}
    for name, clf in cands.items():
        res = cross_validate(build_pipeline(clf), X_tr, y_tr, cv=cv, scoring=scoring, n_jobs=1)
        cv_results[name] = {m: round(float(res[f"test_{m}"].mean()), 4) for m in scoring}
        cv_results[name]["pr_auc_std"] = round(float(res["test_pr_auc"].std()), 4)
        log(f"  CV {name:<24} PR-AUC {cv_results[name]['pr_auc']:.3f} "
            f"ROC-AUC {cv_results[name]['roc_auc']:.3f} recall {cv_results[name]['recall']:.3f}")
    dummy = cross_validate(build_pipeline(DummyClassifier(strategy="prior")), X_tr, y_tr, cv=cv,
                           scoring={"pr_auc": "average_precision"})
    baseline_pr = round(float(dummy["test_pr_auc"].mean()), 4)

    best = max(cv_results, key=lambda k: cv_results[k]["pr_auc"])
    log(f"Selected: {best} (baseline PR-AUC = prevalence = {baseline_pr})")

    # 2) Calibrate, then choose the threshold from out-of-fold predictions.
    model = CalibratedClassifierCV(build_pipeline(cands[best]), method="sigmoid", cv=5)
    oof = cross_val_predict(model, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    thr = pick_threshold(y_tr, oof, target_recall)
    model.fit(X_tr, y_tr)

    # 3) Honest evaluation on the untouched test set.
    p_te = model.predict_proba(X_te)[:, 1]
    test = evaluate(y_te.values, p_te, thr)
    calib = calibration_table(y_te.values, p_te)
    log(f"Threshold {thr:.3f} -> test recall {test['recall']} {test['recall_95ci']}, "
        f"specificity {test['specificity']}, precision {test['precision']}, ROC-AUC {test['roc_auc']}")

    # 4) Which inputs matter (permutation importance on a test subsample).
    sub = X_te.sample(min(4000, len(X_te)), random_state=seed)
    imp = permutation_importance(model, sub, y_te.loc[sub.index], scoring="average_precision",
                                 n_repeats=5, random_state=seed, n_jobs=1)
    importance = sorted(({"feature": c, "importance": round(float(m), 4)}
                         for c, m in zip(RAW_INPUT_COLUMNS, imp.importances_mean)),
                        key=lambda d: -d["importance"])[:12]

    metrics = {
        "model": best, "cv": cv_results, "baseline_pr_auc": baseline_pr, "test": test,
        "calibration": calib, "top_features": importance, "target_recall": target_recall,
        "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
    }
    artifact = {
        "model": model, "threshold": thr, "high_risk_cutoff": HIGH_RISK_CUTOFF,
        "raw_columns": RAW_INPUT_COLUMNS, "model_name": best, "metrics": metrics,
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_note": "Trained on SIMULATED data. Not validated for clinical use.",
    }
    joblib.dump(artifact, out, compress=3)
    if metrics_out:
        with open(metrics_out, "w") as f:
            json.dump(metrics, f, indent=2)
    log(f"Saved {out}" + (f" and {metrics_out}" if metrics_out else ""))
    return artifact


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="ebola_kenya.csv")
    p.add_argument("--out", default="ebola_detector.joblib")
    p.add_argument("--metrics", default="metrics.json")
    p.add_argument("--target-recall", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    if not 0.5 <= a.target_recall < 1.0:
        p.error("--target-recall must be in [0.5, 1.0)")
    train(pd.read_csv(a.data), a.target_recall, a.seed, a.out, a.metrics)


if __name__ == "__main__":
    main()
