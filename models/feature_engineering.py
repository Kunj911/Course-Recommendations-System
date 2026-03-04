"""
models/feature_engineering.py
-------------------------------
Department-specific feature construction for ML models.

Design principle: Each department model receives features emphasizing that
department's primary proficiency as the dominant signal, with shared features
(hardworking, cgpa, etc.) providing supporting context.
"""

import numpy as np
import pandas as pd
from typing import Dict, List

# Maps department codes to their corresponding proficiency column
DEPT_PROFICIENCY_MAP: Dict[str, str] = {
    "CS":   "cs_proficiency",
    "MATH": "math_proficiency",
    "STAT": "stat_proficiency",
    "IS":   "is_proficiency",
    "SE":   "se_proficiency",
}

# Feature columns expected for model training
BASE_STUDENT_COLS = [
    "cgpa", "hardworking_level",
    "cs_proficiency", "math_proficiency", "stat_proficiency",
    "is_proficiency", "se_proficiency",
    "credits_completed", "year",
]

BASE_COURSE_COLS = [
    "difficulty_level", "avg_workload_hours",
]


def build_features_for_department(
    df: pd.DataFrame,
    department: str,
) -> pd.DataFrame:
    """
    Construct department-specific feature matrix from a merged student-course DataFrame.

    The PRIMARY feature is the department-specific proficiency. Derived interaction
    features amplify the signal: a student strong in CS attempting a hard CS course
    gets a clearly positive proficiency_difficulty_gap.

    Args:
        df: DataFrame with student + course columns merged together.
        department: One of CS, MATH, STAT, IS, SE.

    Returns:
        Feature DataFrame ready for scikit-learn.
    """
    dept_prof_col = DEPT_PROFICIENCY_MAP[department]
    dept_prof = df[dept_prof_col]

    features = pd.DataFrame({
        # Primary signal
        "dept_proficiency":      dept_prof,
        # Student effort and background
        "hardworking_level":     df["hardworking_level"],
        "cgpa":                  df["cgpa"],
        "credits_completed":     df["credits_completed"],
        "year":                  df["year"],
        # Course characteristics
        "difficulty_level":      df["difficulty_level"],
        "avg_workload_hours":    df["avg_workload_hours"],
        # Derived: gap between student capability and course demand
        "proficiency_difficulty_gap": dept_prof - df["difficulty_level"],
        # Derived: how much work capacity student has relative to progress
        "workload_capacity":     df["credits_completed"] / (df["year"] + 1),
        # Derived: multiplicative synergy between effort and subject skill
        "hardwork_proficiency_product": df["hardworking_level"] * dept_prof,
        # Derived: experience factor captures study momentum
        "experience_factor":     df["year"] * df["credits_completed"] / 100.0,
    })

    return features.fillna(0.0)


def build_single_student_features(
    student_profile: Dict,
    course_row: Dict,
    department: str,
) -> pd.DataFrame:
    """
    Build feature vector for a single (student, course) pair — used during inference.

    Args:
        student_profile: Dict with student attributes (cgpa, proficiencies, etc.)
        course_row: Dict with course attributes (difficulty_level, avg_workload_hours, etc.)
        department: Department code.

    Returns:
        Single-row feature DataFrame matching training schema.
    """
    dept_prof_col = DEPT_PROFICIENCY_MAP[department]
    dept_prof = student_profile.get(dept_prof_col, 5.0)

    row = {
        "dept_proficiency":            dept_prof,
        "hardworking_level":           student_profile.get("hardworking_level", 5),
        "cgpa":                        student_profile.get("cgpa", 5.0),
        "credits_completed":           student_profile.get("credits_completed", 0),
        "year":                        student_profile.get("year", 1),
        "difficulty_level":            course_row.get("difficulty_level", 5.0),
        "avg_workload_hours":          course_row.get("avg_workload_hours", 10.0),
        "proficiency_difficulty_gap":  dept_prof - course_row.get("difficulty_level", 5.0),
        "workload_capacity":           student_profile.get("credits_completed", 0) / (
                                           student_profile.get("year", 1) + 1
                                       ),
        "hardwork_proficiency_product": student_profile.get("hardworking_level", 5) * dept_prof,
        "experience_factor":            student_profile.get("year", 1) * student_profile.get("credits_completed", 0) / 100.0,
    }
    return pd.DataFrame([row])


def get_feature_names() -> List[str]:
    """Return ordered list of feature names (must match build_features_for_department output)."""
    return [
        "dept_proficiency",
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
