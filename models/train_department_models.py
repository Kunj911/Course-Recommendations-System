"""
Department-Specific Ensemble Model Trainer
==========================================
Trains **20 models** (4 per department × 5 departments) + 5 scalers:

  • GradientBoostingClassifier  — pass/fail (captures non-linear interactions)
  • RandomForestClassifier      — pass/fail (stable base + explainability)
  • GradientBoostingRegressor   — grade prediction (non-linear patterns)
  • SVR (RBF kernel)            — grade prediction (outlier robustness)

Run:
    python -m models.train_department_models

Requires: scikit-learn >= 1.3.0
"""

import os
import sqlite3
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score,
    mean_absolute_error, r2_score,
)

from models.feature_engineering import build_features, DEPARTMENTS
from models.ensemble_config import MODEL_PARAMS, MODEL_FILES

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "database", "course_recommendation.db")
SAVED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved")


def _load_data() -> pd.DataFrame:
    """Join enrollments with students and courses into a single DataFrame."""
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT
            e.grade, e.passed,
            s.cgpa, s.hardworking_level,
            s.cs_proficiency, s.math_proficiency,
            s.stat_proficiency, s.is_proficiency, s.se_proficiency,
            s.credits_completed, s.year,
            c.course_code, c.department,
            c.difficulty_level, c.avg_workload_hours
        FROM enrollments e
        JOIN students s ON e.student_id = s.student_id
        JOIN courses  c ON e.course_id  = c.course_id
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def train_all_models(verbose: bool = True):
    """Train and save all ensemble models for every department.

    For each department:
      1. Filter to that department's enrolment rows.
      2. Build engineered features.
      3. Split 80/20, scale.
      4. Train 4 models: GB-classifier, RF-classifier, GB-regressor, SVR.
      5. Print evaluation metrics.
      6. Persist all models & scaler to ``models/saved/``.
    """
    os.makedirs(SAVED_DIR, exist_ok=True)
    df = _load_data()

    summary: list[dict] = []

    for dept in DEPARTMENTS:
        dept_df = df[df["department"] == dept].copy()
        if dept_df.empty:
            print(f"[WARN] No data for {dept}, skipping.")
            continue

        X = build_features(dept_df, dept)
        y_pass = dept_df["passed"].astype(int).values
        y_grade = dept_df["grade"].astype(float).values

        X_train, X_test, yp_train, yp_test, yg_train, yg_test = train_test_split(
            X, y_pass, y_grade, test_size=0.2, random_state=42
        )

        # ── Scaling (shared across all 4 models) ──
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        # ═══════════════════════════════════════════════════════════════
        # 1. GradientBoostingClassifier (pass/fail)
        # ═══════════════════════════════════════════════════════════════
        gb_clf = GradientBoostingClassifier(**MODEL_PARAMS["gb_classifier"])
        gb_clf.fit(X_train_s, yp_train)
        yp_pred_gb = gb_clf.predict(X_test_s)
        gb_acc = accuracy_score(yp_test, yp_pred_gb)
        gb_f1  = f1_score(yp_test, yp_pred_gb, zero_division=0)

        # ═══════════════════════════════════════════════════════════════
        # 2. RandomForestClassifier (pass/fail — explainability model)
        # ═══════════════════════════════════════════════════════════════
        rf_clf = RandomForestClassifier(**MODEL_PARAMS["rf_classifier"])
        rf_clf.fit(X_train_s, yp_train)
        yp_pred_rf = rf_clf.predict(X_test_s)
        rf_acc = accuracy_score(yp_test, yp_pred_rf)
        rf_f1  = f1_score(yp_test, yp_pred_rf, zero_division=0)

        # ═══════════════════════════════════════════════════════════════
        # 3. GradientBoostingRegressor (grade prediction)
        # ═══════════════════════════════════════════════════════════════
        gb_reg = GradientBoostingRegressor(**MODEL_PARAMS["gb_regressor"])
        gb_reg.fit(X_train_s, yg_train)
        yg_pred_gb = gb_reg.predict(X_test_s)
        gb_mae = mean_absolute_error(yg_test, yg_pred_gb)
        gb_r2  = r2_score(yg_test, yg_pred_gb)

        # ═══════════════════════════════════════════════════════════════
        # 4. SVR (grade prediction — outlier robustness)
        # ═══════════════════════════════════════════════════════════════
        svr_reg = SVR(**MODEL_PARAMS["svr_regressor"])
        svr_reg.fit(X_train_s, yg_train)
        yg_pred_svr = svr_reg.predict(X_test_s)
        svr_mae = mean_absolute_error(yg_test, yg_pred_svr)
        svr_r2  = r2_score(yg_test, yg_pred_svr)

        # ── Persist all models ──
        joblib.dump(gb_clf,  os.path.join(SAVED_DIR, MODEL_FILES["gb_classifier"].format(dept=dept)))
        joblib.dump(rf_clf,  os.path.join(SAVED_DIR, MODEL_FILES["rf_classifier"].format(dept=dept)))
        joblib.dump(gb_reg,  os.path.join(SAVED_DIR, MODEL_FILES["gb_regressor"].format(dept=dept)))
        joblib.dump(svr_reg, os.path.join(SAVED_DIR, MODEL_FILES["svr_regressor"].format(dept=dept)))
        joblib.dump(scaler,  os.path.join(SAVED_DIR, MODEL_FILES["scaler"].format(dept=dept)))

        metrics = {
            "department": dept,
            "samples": len(dept_df),
            "gb_clf_acc": round(gb_acc, 4),
            "gb_clf_f1":  round(gb_f1, 4),
            "rf_clf_acc": round(rf_acc, 4),
            "rf_clf_f1":  round(rf_f1, 4),
            "gb_reg_mae": round(gb_mae, 4),
            "gb_reg_r2":  round(gb_r2, 4),
            "svr_mae":    round(svr_mae, 4),
            "svr_r2":     round(svr_r2, 4),
        }
        summary.append(metrics)

        if verbose:
            print(f"\n{'='*65}")
            print(f"  Department: {dept}  ({len(dept_df)} samples)")
            print(f"{'='*65}")
            print(f"  GB Classifier  — Acc: {gb_acc:.3f}  F1: {gb_f1:.3f}")
            print(f"  RF Classifier  — Acc: {rf_acc:.3f}  F1: {rf_f1:.3f}")
            print(f"  GB Regressor   — MAE: {gb_mae:.3f}  R²: {gb_r2:.3f}")
            print(f"  SVR Regressor  — MAE: {svr_mae:.3f}  R²: {svr_r2:.3f}")

    if verbose:
        print(f"\n{'='*65}")
        print(f"  TRAINING COMPLETE — {len(summary)} departments")
        print(f"  Total models saved: {len(summary) * 4} models + {len(summary)} scalers")
        print(f"  Output directory: {SAVED_DIR}")
        print(f"{'='*65}")

        if summary:
            summary_df = pd.DataFrame(summary)
            print("\n-- Summary Table --")
            print(summary_df.to_string(index=False))

    return summary


if __name__ == "__main__":
    train_all_models()
