"""
Prerequisite Checker
====================
Validates whether a student has completed the prerequisite courses
required for a given target course and computes a readiness score.
"""

import sqlite3
import os
from dataclasses import dataclass, field

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "database", "course_recommendation.db")


@dataclass
class PrerequisiteResult:
    """Container for prerequisite check output."""
    course_code: str
    course_name: str
    prerequisite_met: bool = True
    prerequisite_code: str | None = None
    prerequisite_name: str | None = None
    prerequisite_grade: float | None = None
    readiness_score: float = 1.0  # 0-1 scale
    warning_message: str = ""
    warnings: list[str] = field(default_factory=list)


class PrerequisiteChecker:
    """Check and report prerequisite status for courses."""

    def __init__(self, db_path: str = DB_PATH):
        self._db_path = db_path
        self._course_cache: dict = {}
        self._load_course_graph()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_course_graph(self):
        """Cache the full course table in memory for fast lookups."""
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT course_id, course_code, course_name, department, "
            "difficulty_level, avg_workload_hours, prerequisite_course_id, "
            "is_advanced FROM courses"
        ).fetchall()
        conn.close()

        for r in rows:
            self._course_cache[r["course_id"]] = dict(r)
        # Also index by code for convenience
        self._code_to_id = {
            v["course_code"]: k for k, v in self._course_cache.items()
        }

    def _get_prerequisite_id(self, course_id: int) -> int | None:
        course = self._course_cache.get(course_id)
        if course:
            return course["prerequisite_course_id"]
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, course_id: int,
              completed_course_ids: set[int] | None = None,
              completed_grades: dict[int, float] | None = None) -> PrerequisiteResult:
        """Check prerequisite status for one course.

        Parameters
        ----------
        course_id : int
            The target course to check.
        completed_course_ids : set[int] | None
            Course IDs the student has already completed.
            When ``None`` (custom student entry), the student is assumed
            to have completed *nothing*, so all prereqs will trigger warnings.
        completed_grades : dict[int, float] | None
            Mapping of course_id → grade for completed courses.

        Returns
        -------
        PrerequisiteResult
        """
        course = self._course_cache.get(course_id)
        if course is None:
            return PrerequisiteResult(
                course_code="?", course_name="Unknown",
                prerequisite_met=True,
            )

        prereq_id = course["prerequisite_course_id"]
        result = PrerequisiteResult(
            course_code=course["course_code"],
            course_name=course["course_name"],
        )

        # No prerequisite → always met
        if prereq_id is None:
            result.prerequisite_met = True
            result.readiness_score = 1.0
            return result

        prereq = self._course_cache.get(prereq_id, {})
        result.prerequisite_code = prereq.get("course_code", "?")
        result.prerequisite_name = prereq.get("course_name", "Unknown")

        completed = completed_course_ids or set()
        grades = completed_grades or {}

        if prereq_id in completed:
            result.prerequisite_met = True
            grade = grades.get(prereq_id)
            if grade is not None:
                result.prerequisite_grade = grade
                # Readiness is proportional to how well the prereq was mastered
                result.readiness_score = min(1.0, grade / 10.0)
            else:
                result.readiness_score = 0.7  # completed but unknown grade
        else:
            result.prerequisite_met = False
            result.readiness_score = 0.3  # penalise missing prereqs
            warning = (
                f"⚠️ Warning: {result.prerequisite_code} "
                f"({result.prerequisite_name}) is a prerequisite for "
                f"{result.course_code} ({result.course_name}). "
                f"You have not completed {result.prerequisite_code}."
            )
            result.warning_message = warning
            result.warnings.append(warning)

        return result

    def check_all(self, course_ids: list[int],
                  completed_course_ids: set[int] | None = None,
                  completed_grades: dict[int, float] | None = None
                  ) -> dict[int, PrerequisiteResult]:
        """Batch-check prerequisites for multiple courses."""
        return {
            cid: self.check(cid, completed_course_ids, completed_grades)
            for cid in course_ids
        }

    def get_course_info(self, course_id: int) -> dict | None:
        return self._course_cache.get(course_id)

    def get_all_courses(self) -> list[dict]:
        return list(self._course_cache.values())

    def get_course_id_by_code(self, code: str) -> int | None:
        return self._code_to_id.get(code)
