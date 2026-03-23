"""
Input Validators
================
Centralised validation helpers for student profile inputs.
Used by both the Streamlit UI and any programmatic callers.
"""

from typing import Any

PROFICIENCY_FIELDS = [
    ("cgpa", "CGPA"),
    ("hardworking_level", "Hardworking Level"),
    ("cs_proficiency", "CS Proficiency"),
    ("math_proficiency", "Math Proficiency"),
    ("stat_proficiency", "Statistics Proficiency"),
    ("is_proficiency", "IS Proficiency"),
    ("se_proficiency", "SE Proficiency"),
]

MIN_VAL = 0.0
MAX_VAL = 10.0
MIN_HARDWORKING = 1
MAX_HARDWORKING = 10


def validate_range(value: Any, field_name: str,
                   low: float = MIN_VAL, high: float = MAX_VAL) -> tuple[bool, str]:
    """Return (is_valid, error_message). Empty string if valid."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False, f"{field_name} must be a number."
    if v < low or v > high:
        return False, f"{field_name} must be between {low} and {high}."
    return True, ""


def validate_student_profile(profile: dict) -> list[str]:
    """Validate all fields in a student profile dict.

    Returns a list of error strings (empty list = valid).
    """
    errors: list[str] = []

    if not profile.get("name", "").strip():
        errors.append("Name is required.")

    for key, label in PROFICIENCY_FIELDS:
        if key == "hardworking_level":
            ok, msg = validate_range(profile.get(key), label,
                                     MIN_HARDWORKING, MAX_HARDWORKING)
        else:
            ok, msg = validate_range(profile.get(key), label)
        if not ok:
            errors.append(msg)

    return errors
