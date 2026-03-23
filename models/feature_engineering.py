"""
Feature Engineering Module
==========================
Creates department-specific feature vectors used by the ML models.

Each department model receives the same *shape* of features but the
"primary proficiency" column is swapped to match the target department.
"""

import pandas as pd
import numpy as np

# The order matters – it matches the column order the models expect
DEPARTMENTS = ["CS", "MATH", "STAT", "IS", "SE"]

DEPT_PROFICIENCY_COL = {
    "CS":   "cs_proficiency",
    "MATH": "math_proficiency",
    "STAT": "stat_proficiency",
    "IS":   "is_proficiency",
    "SE":   "se_proficiency",
}

# Canonical feature names (after engineering)
FEATURE_NAMES = [
    "department_proficiency",
    "hardworking_level",
    "cgpa",
    "credits_completed",
    "year",
    "difficulty_level",
    "avg_workload_hours",
    "proficiency_difficulty_gap",
    "workload_capacity",
    "hardwork_proficiency_product",
    "experience_factor",
]


def build_features(df: pd.DataFrame, department: str) -> pd.DataFrame:
    """Return a DataFrame with engineered features for *one* department.

    Parameters
    ----------
    df : pd.DataFrame
        Merged enrollment + student + course data.  Must contain at
        minimum the raw columns referenced below.
    department : str
        One of CS, MATH, STAT, IS, SE.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns matching ``FEATURE_NAMES``.
    """
    prof_col = DEPT_PROFICIENCY_COL[department]

    features = pd.DataFrame()
    features["department_proficiency"] = df[prof_col].astype(float)
    features["hardworking_level"]     = df["hardworking_level"].astype(float)
    features["cgpa"]                  = df["cgpa"].astype(float)
    features["credits_completed"]     = df["credits_completed"].astype(float)
    features["year"]                  = df["year"].astype(float)
    features["difficulty_level"]      = df["difficulty_level"].astype(float)
    features["avg_workload_hours"]    = df["avg_workload_hours"].astype(float)

    # Derived features
    features["proficiency_difficulty_gap"] = (
        features["department_proficiency"] - features["difficulty_level"]
    )
    features["workload_capacity"] = (
        features["credits_completed"] / (features["year"] + 1)
    )
    features["hardwork_proficiency_product"] = (
        features["hardworking_level"] * features["department_proficiency"]
    )
    features["experience_factor"] = (
        features["year"] * features["credits_completed"] / 100.0
    )

    return features


def build_features_for_student(student_profile: dict,
                               course_row: dict,
                               department: str) -> np.ndarray:
    """Build a single feature vector for a new (custom) student.

    Parameters
    ----------
    student_profile : dict
        Keys: cgpa, hardworking_level, cs_proficiency, math_proficiency,
              stat_proficiency, is_proficiency, se_proficiency,
              credits_completed, year.
    course_row : dict
        Keys: difficulty_level, avg_workload_hours.
    department : str
        The department this course belongs to.

    Returns
    -------
    np.ndarray  shape (1, len(FEATURE_NAMES))
    """
    prof_key = DEPT_PROFICIENCY_COL[department]
    dept_prof = float(student_profile[prof_key])
    hw = float(student_profile["hardworking_level"])
    cgpa = float(student_profile["cgpa"])
    cc = float(student_profile.get("credits_completed", 60))
    yr = float(student_profile.get("year", 2))
    diff = float(course_row["difficulty_level"])
    wl = float(course_row["avg_workload_hours"])

    vec = [
        dept_prof,
        hw,
        cgpa,
        cc,
        yr,
        diff,
        wl,
        dept_prof - diff,                  # proficiency_difficulty_gap
        cc / (yr + 1),                     # workload_capacity
        hw * dept_prof,                    # hardwork_proficiency_product
        yr * cc / 100.0,                   # experience_factor
    ]
    return np.array(vec).reshape(1, -1)
