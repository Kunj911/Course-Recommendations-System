"""
models/train_department_models.py
-----------------------------------
Trains 10 models (Logistic + Linear per department) using enrollment history.

Each department receives its own specialized model trained only on that
department's course data, ensuring each model captures domain-specific
performance patterns rather than blending signals across departments.
"""

import os
import sys
import sqlite3
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.feature_engineering import build_features_for_department, get_feature_names

DB_PATH   = os.path.join(os.path.dirname(os.path.dirname(__file__)), "database", "course_recommendation.db")
SAVED_DIR = os.path.join(os.path.dirname(__file__), "saved")
DEPARTMENTS = ["CS", "MATH", "STAT", "IS", "SE"]


def load_training_data(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Join enrollments with student profiles and course attributes.
    Returns a flat DataFrame ready for feature engineering.
    """
    query = """
        SELECT
            e.student_id, e.course_id, e.grade, e.passed, e.hours_spent,
            s.cgpa, s.hardworking_level,
            s.cs_proficiency, s.math_proficiency, s.stat_proficiency,
            s.is_proficiency, s.se_proficiency,
            s.credits_completed, s.year,
            c.department, c.difficulty_level, c.avg_workload_hours,
            c.is_advanced, c.course_code
        FROM enrollments e
        JOIN students  s ON e.student_id = s.student_id
        JOIN courses   c ON e.course_id  = c.course_id
        WHERE e.grade IS NOT NULL
    """
    df = pd.read_sql_query(query, conn)
    return df


def train_department(
    dept: str,
    df_dept: pd.DataFrame,
) -> dict:
    """
    Train Logistic (pass/fail) and Linear (grade prediction) models for one department.

    We use StandardScaler fitted on training data only, then apply it to test
    to prevent data leakage.

    Returns:
        dict with logistic_model, linear_model, scaler, and evaluation metrics.
    """
    print(f"\n  📊 Training {dept} models on {len(df_dept)} samples...")

    X = build_features_for_department(df_dept, dept)
    y_class = df_dept["passed"].astype(int)
    y_reg   = df_dept["grade"].astype(float)

    # Stratified split for classification balance
    X_train, X_test, yc_train, yc_test, yr_train, yr_test = train_test_split(
        X, y_class, y_reg, test_size=0.2, random_state=42, stratify=y_class
    )

    # Scale features — department-specific scaler avoids cross-domain normalization issues
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Logistic Regression: Predict pass (1) / fail (0)
    logistic = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",  # handles potential class imbalance
        C=1.0,
        random_state=42,
    )
    logistic.fit(X_train_scaled, yc_train)
    yc_pred = logistic.predict(X_test_scaled)
    acc = accuracy_score(yc_test, yc_pred)
    f1  = f1_score(yc_test, yc_pred, zero_division=0)

    # Linear Regression: Predict expected grade (0-10)
    linear = LinearRegression()
    linear.fit(X_train_scaled, yr_train)
    yr_pred = linear.predict(X_test_scaled)
    mae = mean_absolute_error(yr_test, yr_pred)
    r2  = r2_score(yr_test, yr_pred)

    print(f"    Logistic → Accuracy: {acc:.3f} | F1: {f1:.3f}")
    print(f"    Linear   → MAE: {mae:.3f} | R²: {r2:.3f}")

    return {
        "logistic": logistic,
        "linear":   linear,
        "scaler":   scaler,
        "metrics": {
            "dept": dept,
            "n_samples": len(df_dept),
            "logistic_accuracy": round(acc, 4),
            "logistic_f1": round(f1, 4),
            "linear_mae": round(mae, 4),
            "linear_r2": round(r2, 4),
        }
    }


def save_models(dept: str, models_dict: dict) -> None:
    """Persist trained models and scaler to disk."""
    os.makedirs(SAVED_DIR, exist_ok=True)
    joblib.dump(models_dict["logistic"], os.path.join(SAVED_DIR, f"{dept}_logistic.pkl"))
    joblib.dump(models_dict["linear"],   os.path.join(SAVED_DIR, f"{dept}_linear.pkl"))
    joblib.dump(models_dict["scaler"],   os.path.join(SAVED_DIR, f"scaler_{dept}.pkl"))


def train_all_models() -> None:
    """Main entry point: trains all 10 models and reports performance."""
    conn = sqlite3.connect(DB_PATH)

    print("📦 Loading training data from database...")
    df = load_training_data(conn)
    conn.close()

    print(f"   Total enrollment records: {len(df)}")
    all_metrics = []

    for dept in DEPARTMENTS:
        df_dept = df[df["department"] == dept].copy()
        if len(df_dept) < 50:
            print(f"  ⚠️  Skipping {dept} — insufficient data ({len(df_dept)} samples)")
            continue

        result = train_department(dept, df_dept)
        save_models(dept, result)
        all_metrics.append(result["metrics"])
        print(f"    ✅ {dept} models saved.")

    print("\n" + "="*55)
    print("📈 TRAINING SUMMARY")
    print("="*55)
    metrics_df = pd.DataFrame(all_metrics)
    print(metrics_df.to_string(index=False))
    print("="*55)
    print("\n🏁 All models trained and saved to models/saved/")


if __name__ == "__main__":
    train_all_models()
