"""
models/prerequisite_checker.py
--------------------------------
Validates whether a student meets course prerequisites and computes a
"readiness score" that reflects both completion status and prior performance.
"""

import sqlite3
import os
from typing import Dict, List, Optional, Tuple

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "database", "course_recommendation.db")


class PrerequisiteChecker:
    """
    Checks prerequisite chains for courses and returns structured warnings.

    For custom (non-database) students, we assume no prior courses are completed
    unless the user explicitly indicates year/credits, which we use as a proxy.
    """

    def __init__(self):
        self._course_cache: Optional[Dict] = None

    def _load_courses(self) -> Dict:
        """Lazy-load course data including prerequisite relationships."""
        if self._course_cache is not None:
            return self._course_cache

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT c.course_id, c.course_code, c.course_name, c.department,
                   c.difficulty_level, c.prerequisite_course_id, c.is_advanced,
                   p.course_code AS prereq_code, p.course_name AS prereq_name
            FROM courses c
            LEFT JOIN courses p ON c.prerequisite_course_id = p.course_id
        """)
        cols = [d[0] for d in cursor.description]
        rows = cursor.fetchall()
        conn.close()

        courses = {}
        for row in rows:
            d = dict(zip(cols, row))
            courses[d["course_id"]] = d

        self._course_cache = courses
        return courses

    def check_prerequisites(
        self,
        course_id: int,
        completed_course_ids: List[int],
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if a student has met the prerequisite for a given course.

        Args:
            course_id: Target course to check.
            completed_course_ids: List of course IDs the student has passed.

        Returns:
            (met: bool, warning_message: Optional[str])
        """
        courses = self._load_courses()
        course = courses.get(course_id)
        if not course:
            return True, None  # Unknown course — assume no prereq

        prereq_id = course.get("prerequisite_course_id")
        if not prereq_id:
            return True, None  # No prerequisite

        prereq_met = prereq_id in completed_course_ids
        if prereq_met:
            return True, None

        # Build descriptive warning
        prereq_code = course.get("prereq_code", "Unknown")
        prereq_name = course.get("prereq_name", "Unknown Course")
        target_code = course.get("course_code", "Unknown")
        target_name = course.get("course_name", "Unknown Course")

        warning = (
            f"⚠️ Warning: {prereq_code} ({prereq_name}) is a prerequisite for "
            f"{target_code} ({target_name}). You have not completed {prereq_code}."
        )
        return False, warning

    def compute_readiness_score(
        self,
        course_id: int,
        completed_course_ids: List[int],
        prior_grades: Dict[int, float],
    ) -> float:
        """
        Compute a readiness score [0-1] based on prerequisite completion and grade quality.

        A score of 1.0 means all prerequisites met with excellent grades.
        A score of 0.0 means prerequisite not completed at all.

        Args:
            course_id: Target course.
            completed_course_ids: Courses the student has passed.
            prior_grades: Mapping of course_id -> grade for completed courses.
        """
        courses = self._load_courses()
        course = courses.get(course_id)
        if not course:
            return 1.0

        prereq_id = course.get("prerequisite_course_id")
        if not prereq_id:
            return 1.0  # No prerequisite → full readiness

        if prereq_id not in completed_course_ids:
            return 0.0  # Missing prerequisite → zero readiness

        # Readiness scales with prerequisite grade quality
        # A grade of 5 (pass threshold) → 0.5, grade of 10 → 1.0
        prereq_grade = prior_grades.get(prereq_id, 5.0)
        readiness = prereq_grade / 10.0

        return round(min(1.0, max(0.0, readiness)), 3)

    def get_all_course_prerequisites(self) -> Dict[int, Optional[int]]:
        """Return mapping of course_id -> prerequisite_course_id (None if no prereq)."""
        courses = self._load_courses()
        return {cid: c.get("prerequisite_course_id") for cid, c in courses.items()}

    def get_all_courses(self) -> Dict[int, Dict]:
        """Expose full course dictionary for use in recommender."""
        return self._load_courses()
