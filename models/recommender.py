"""
Multi-Model Ensemble Recommendation Engine
============================================
Generates ranked course recommendations using an ensemble of 4 models
per department:

  Classification (pass/fail):  GB(50%) + RF(50%)
  Regression     (grade):      GB(60%) + SVR(40%)

Confidence = 1.0 − std(gb_prob, rf_prob)

Requires: scikit-learn >= 1.3.0
"""

import os
import numpy as np
import joblib

from models.feature_engineering import (
    build_features_for_student,
    DEPARTMENTS,
    DEPT_PROFICIENCY_COL,
    FEATURE_NAMES,
)
from models.prerequisite_checker import PrerequisiteChecker
from models.ensemble_config import ENSEMBLE_WEIGHTS, MODEL_FILES

SAVED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved")


class CourseRecommender:
    """Load department ensemble models and produce recommendations."""

    def __init__(self):
        self._models: dict[str, dict] = {}
        self._load_models()
        self.checker = PrerequisiteChecker()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_models(self):
        """Load all persisted ensemble models and scalers."""
        for dept in DEPARTMENTS:
            paths = {
                key: os.path.join(SAVED_DIR, template.format(dept=dept))
                for key, template in MODEL_FILES.items()
            }

            if not all(os.path.exists(p) for p in paths.values()):
                print(f"[WARN] Ensemble models for {dept} not found - skipping.")
                continue

            self._models[dept] = {
                key: joblib.load(path) for key, path in paths.items()
            }

    # ------------------------------------------------------------------
    # Core prediction (ensemble)
    # ------------------------------------------------------------------

    def _predict_for_course(self, student: dict, course: dict,
                            department: str) -> dict:
        """Run the ensemble for a single (student, course) pair.

        Returns success_probability, expected_grade, and confidence_score.
        """
        models = self._models.get(department)
        if models is None:
            return {
                "success_probability": 0.5,
                "expected_grade": 5.0,
                "confidence_score": 0.0,
            }

        X = build_features_for_student(student, course, department)
        X_scaled = models["scaler"].transform(X)

        # ── Classification ensemble: GB + RF ──
        w_cls = ENSEMBLE_WEIGHTS["classification"]

        gb_prob_arr = models["gb_classifier"].predict_proba(X_scaled)[0]
        gb_prob = float(gb_prob_arr[1]) if len(gb_prob_arr) > 1 else float(gb_prob_arr[0])

        rf_prob_arr = models["rf_classifier"].predict_proba(X_scaled)[0]
        rf_prob = float(rf_prob_arr[1]) if len(rf_prob_arr) > 1 else float(rf_prob_arr[0])

        final_prob = gb_prob * w_cls["gb"] + rf_prob * w_cls["rf"]

        # ── Regression ensemble: GB + SVR ──
        w_reg = ENSEMBLE_WEIGHTS["regression"]

        gb_grade = float(models["gb_regressor"].predict(X_scaled)[0])
        svr_grade = float(models["svr_regressor"].predict(X_scaled)[0])

        final_grade = gb_grade * w_reg["gb"] + svr_grade * w_reg["svr"]
        final_grade = round(np.clip(final_grade, 0.0, 10.0), 1)

        # ── Confidence score: 1 - std of classifier probabilities ──
        confidence = 1.0 - float(np.std([gb_prob, rf_prob]))

        return {
            "success_probability": round(final_prob, 4),
            "expected_grade": final_grade,
            "confidence_score": round(confidence, 4),
        }

    # ------------------------------------------------------------------
    # Feature importances (from RandomForest)
    # ------------------------------------------------------------------

    def get_feature_importances(self, department: str,
                                top_n: int = 5) -> list[dict]:
        """Return the top-N most important features for a department.

        Uses the RandomForest classifier's ``.feature_importances_``.

        Returns
        -------
        list[dict]  – each dict has keys ``feature`` and ``importance``.
        """
        models = self._models.get(department)
        if models is None:
            return []

        importances = models["rf_classifier"].feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]

        return [
            {
                "feature": FEATURE_NAMES[i],
                "importance": round(float(importances[i]), 4),
            }
            for i in indices
        ]

    # ------------------------------------------------------------------
    # Recommendation scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _recommendation_score(success_prob: float, expected_grade: float,
                              proficiency_match: float) -> float:
        """Composite score per the spec:
        score = (success_prob^0.6) × (grade/10) × (prof_match^0.4) × 10
        """
        score = (
            (success_prob ** 0.6)
            * (expected_grade / 10.0)
            * (proficiency_match ** 0.4)
            * 10
        )
        return round(score, 3)

    @staticmethod
    def _readiness_label(score: float) -> str:
        if score >= 0.75:
            return "High"
        elif score >= 0.5:
            return "Medium"
        return "Low"

    @staticmethod
    def _confidence_label(confidence: float) -> str:
        """Map ensemble confidence score to a human-readable label."""
        if confidence >= 0.80:
            return "🟢 High confidence"
        elif confidence >= 0.60:
            return "🟡 Medium confidence"
        return "🔴 Low confidence"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recommend(self, student_profile: dict,
                  top_n: int = 10) -> list[dict]:
        """Generate top-N course recommendations for a custom student.

        Parameters
        ----------
        student_profile : dict
            Must contain: name, cgpa, hardworking_level,
            cs_proficiency, math_proficiency, stat_proficiency,
            is_proficiency, se_proficiency.
            Optional: credits_completed, year.
        top_n : int
            Number of recommendations to return.

        Returns
        -------
        list[dict]  – sorted by recommendation_score descending.
            Includes all original columns plus:
            - confidence_score  (float)
            - confidence_label  (str)
        """
        # Default optional fields
        student_profile.setdefault("credits_completed", 60)
        student_profile.setdefault("year", 2)

        all_courses = self.checker.get_all_courses()
        results: list[dict] = []

        for course in all_courses:
            dept = course["department"]
            cid = course["course_id"]

            # Department proficiency for this course
            prof_key = DEPT_PROFICIENCY_COL[dept]
            dept_proficiency = float(student_profile.get(prof_key, 5.0))

            # Prerequisite check (custom student has no history)
            prereq_result = self.checker.check(cid)

            # Predict with ensemble
            preds = self._predict_for_course(student_profile, course, dept)
            success_prob = preds["success_probability"]
            expected_grade = preds["expected_grade"]
            confidence_score = preds["confidence_score"]

            # Proficiency match: how well does the student's strength
            # align with the course difficulty?
            proficiency_match = min(1.0, dept_proficiency / 10.0)

            rec_score = self._recommendation_score(
                success_prob, expected_grade, proficiency_match
            )

            readiness = prereq_result.readiness_score
            readiness_label = self._readiness_label(readiness)
            conf_label = self._confidence_label(confidence_score)

            results.append({
                "course_id": cid,
                "course_code": course["course_code"],
                "course_name": course["course_name"],
                "department": dept,
                "difficulty": course["difficulty_level"],
                "credits": course.get("credits", 3),
                "success_probability": success_prob,
                "expected_grade": expected_grade,
                "recommendation_score": rec_score,
                "readiness_score": round(readiness, 2),
                "readiness_label": readiness_label,
                "confidence_score": confidence_score,
                "confidence_label": conf_label,
                "confidence": conf_label,          # backward compat alias
                "prerequisite_met": prereq_result.prerequisite_met,
                "prerequisite_code": prereq_result.prerequisite_code,
                "prerequisite_name": prereq_result.prerequisite_name,
                "prerequisite_warning": prereq_result.warning_message,
                "model_used": "Ensemble: GB(50%) + RF(50%) | Grade: GB(60%) + SVR(40%)",
            })

        # Sort: higher score = better recommendation
        results.sort(key=lambda r: r["recommendation_score"], reverse=True)
        return results[:top_n]
