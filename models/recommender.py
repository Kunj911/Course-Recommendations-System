"""
models/recommender.py
----------------------
Multi-model recommendation engine.

Core design: Each course is evaluated using its department-specific model,
ensuring that CS predictions draw on CS-specialized learning rather than
a generic model that might average across departments.

Recommendation scoring formula:
  score = (success_prob^0.6) × (expected_grade/10) × (proficiency_match^0.4) × 10
"""

import os
import sys
import sqlite3
import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.feature_engineering import build_single_student_features, DEPT_PROFICIENCY_MAP
from models.prerequisite_checker import PrerequisiteChecker

SAVED_DIR = os.path.join(os.path.dirname(__file__), "saved")
DB_PATH   = os.path.join(os.path.dirname(os.path.dirname(__file__)), "database", "course_recommendation.db")
DEPARTMENTS = ["CS", "MATH", "STAT", "IS", "SE"]

# Readiness thresholds
READINESS_HIGH   = 0.70
READINESS_MEDIUM = 0.40

DEPT_COLORS = {
    "CS":   "#4FC3F7",  # Blue
    "MATH": "#81C784",  # Green
    "STAT": "#FFB74D",  # Orange
    "IS":   "#CE93D8",  # Purple
    "SE":   "#F48FB1",  # Pink
}


class CourseRecommender:
    """
    Loads pre-trained department models and generates ranked course recommendations
    for a custom student profile without persisting anything to the database.
    """

    def __init__(self):
        self.models: Dict[str, dict] = {}
        self.checker = PrerequisiteChecker()
        self._courses_cache: Optional[List[Dict]] = None
        self._load_models()

    def _load_models(self) -> None:
        """Load all 10 trained models and 5 scalers from disk."""
        for dept in DEPARTMENTS:
            logistic_path = os.path.join(SAVED_DIR, f"{dept}_logistic.pkl")
            linear_path   = os.path.join(SAVED_DIR, f"{dept}_linear.pkl")
            scaler_path   = os.path.join(SAVED_DIR, f"scaler_{dept}.pkl")

            if not all(os.path.exists(p) for p in [logistic_path, linear_path, scaler_path]):
                print(f"⚠️  Models for {dept} not found. Run train_department_models.py first.")
                continue

            self.models[dept] = {
                "logistic": joblib.load(logistic_path),
                "linear":   joblib.load(linear_path),
                "scaler":   joblib.load(scaler_path),
            }

    def _load_all_courses(self) -> List[Dict]:
        """Fetch all courses from database (cached after first call)."""
        if self._courses_cache is not None:
            return self._courses_cache

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM courses ORDER BY department, difficulty_level")
        cols = [d[0] for d in cursor.description]
        rows = cursor.fetchall()
        conn.close()

        self._courses_cache = [dict(zip(cols, row)) for row in rows]
        return self._courses_cache

    def _predict_for_course(
        self,
        student: Dict,
        course: Dict,
    ) -> Dict:
        """
        Run department-specific model prediction for one (student, course) pair.

        Returns dict with success_probability, expected_grade, and metadata.
        """
        dept = course["department"]
        if dept not in self.models:
            return {"success_probability": 0.5, "expected_grade": 5.0, "model_used": "fallback"}

        model_bundle = self.models[dept]
        features_df  = build_single_student_features(student, course, dept)

        features_scaled = model_bundle["scaler"].transform(features_df)

        # Probability of passing
        prob_pass = model_bundle["logistic"].predict_proba(features_scaled)[0][1]
        # Expected numerical grade
        raw_grade = model_bundle["linear"].predict(features_scaled)[0]
        expected_grade = float(np.clip(raw_grade, 0.0, 10.0))

        return {
            "success_probability": round(float(prob_pass), 4),
            "expected_grade":      round(expected_grade, 2),
            "model_used":          f"{dept}-specialized model",
        }

    def _compute_proficiency_match(self, student: Dict, dept: str) -> float:
        """
        Normalize the student's department proficiency to [0,1] as a match indicator.
        High proficiency → strong alignment between student strengths and course domain.
        """
        prof_col = DEPT_PROFICIENCY_MAP[dept]
        prof = student.get(prof_col, 5.0)
        return float(np.clip(prof / 10.0, 0.0, 1.0))

    def _compute_recommendation_score(
        self,
        success_prob: float,
        expected_grade: float,
        proficiency_match: float,
    ) -> float:
        """
        Composite score balancing probability of success, grade quality, and domain fit.
        Formula: (success_prob^0.6) × (expected_grade/10) × (proficiency_match^0.4) × 10
        """
        score = (
            (success_prob ** 0.6) *
            (expected_grade / 10.0) *
            (proficiency_match ** 0.4) *
            10.0
        )
        return round(float(np.clip(score, 0.0, 10.0)), 3)

    def _readiness_label(self, readiness_score: float, prereq_met: bool) -> str:
        """Map numeric readiness to human-readable label with color emoji."""
        if not prereq_met:
            return "🔴 Missing Prereq"
        if readiness_score >= READINESS_HIGH:
            return "🟢 High"
        elif readiness_score >= READINESS_MEDIUM:
            return "🟡 Medium"
        else:
            return "🔴 Low"

    def recommend(
        self,
        student: Dict,
        top_n: int = 10,
        completed_course_ids: Optional[List[int]] = None,
        prior_grades: Optional[Dict[int, float]] = None,
    ) -> pd.DataFrame:
        """
        Generate ranked course recommendations for a student profile.

        The student dict is NOT saved to the database — it's used transiently
        for inference only.

        Args:
            student: Dict with keys: cgpa, hardworking_level, cs/math/stat/is/se_proficiency,
                     credits_completed, year.
            top_n: Number of recommendations to return.
            completed_course_ids: Courses the student has already passed.
            prior_grades: Grade history for readiness score calculation.

        Returns:
            DataFrame of ranked recommendations with all metadata.
        """
        if completed_course_ids is None:
            completed_course_ids = []
        if prior_grades is None:
            prior_grades = {}

        courses = self._load_all_courses()
        results = []

        for course in courses:
            course_id = course["course_id"]

            # Skip already completed courses
            if course_id in completed_course_ids:
                continue

            # Prerequisite check
            prereq_met, warning_msg = self.checker.check_prerequisites(
                course_id, completed_course_ids
            )
            readiness_score = self.checker.compute_readiness_score(
                course_id, completed_course_ids, prior_grades
            )

            # Model prediction
            prediction = self._predict_for_course(student, course)

            dept = course["department"]
            proficiency_match = self._compute_proficiency_match(student, dept)

            rec_score = self._compute_recommendation_score(
                prediction["success_probability"],
                prediction["expected_grade"],
                proficiency_match,
            )

            results.append({
                "course_id":           course_id,
                "course_code":         course["course_code"],
                "course_name":         course["course_name"],
                "department":          dept,
                "credits":             course["credits"],
                "difficulty":          course["difficulty_level"],
                "avg_workload_hours":  course["avg_workload_hours"],
                "is_advanced":         bool(course["is_advanced"]),
                "success_probability": prediction["success_probability"],
                "expected_grade":      prediction["expected_grade"],
                "proficiency_match":   round(proficiency_match, 3),
                "recommendation_score": rec_score,
                "prereq_met":          prereq_met,
                "prereq_warning":      warning_msg,
                "readiness_score":     readiness_score,
                "readiness_label":     self._readiness_label(readiness_score, prereq_met),
                "model_used":          prediction["model_used"],
                "dept_color":          DEPT_COLORS.get(dept, "#FFFFFF"),
            })

        df = pd.DataFrame(results)
        if df.empty:
            return df

        df = df.sort_values("recommendation_score", ascending=False).reset_index(drop=True)
        df.index += 1
        df.index.name = "rank"

        return df.head(top_n)

    def get_student_explanation(self, student: Dict, course_row: pd.Series) -> str:
        """
        Generate a natural-language explanation for why a course was recommended.
        """
        dept = course_row["department"]
        prof_col = DEPT_PROFICIENCY_MAP[dept]
        prof = student.get(prof_col, 5.0)

        lines = [
            f"**Model**: {course_row['model_used']}",
            f"**Your {dept} proficiency**: {prof:.1f}/10 "
            f"({'Strong ✅' if prof >= 7 else 'Moderate 🟡' if prof >= 5 else 'Needs work 🔴'})",
            f"**Predicted success**: {course_row['success_probability']*100:.0f}%",
            f"**Expected grade**: {course_row['expected_grade']:.1f}/10",
        ]
        if not course_row["prereq_met"]:
            lines.append(f"**⚠️ Prerequisite missing** — complete it first for better outcomes")

        return "\n".join(lines)
