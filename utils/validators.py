"""
utils/validators.py
--------------------
Input validation for student profile entries.
Keeps validation logic separate from UI so it can be tested independently.
"""

from typing import Dict, List, Tuple, Optional


PROFICIENCY_FIELDS = [
    "cs_proficiency", "math_proficiency", "stat_proficiency",
    "is_proficiency", "se_proficiency",
]

SCORE_RANGE = (0.0, 10.0)
CGPA_RANGE  = (0.0, 10.0)
HARD_RANGE  = (1, 10)
YEAR_RANGE  = (1, 4)


def validate_student_profile(profile: Dict) -> Tuple[bool, List[str]]:
    """
    Validate a student profile dictionary before sending to recommender.

    Args:
        profile: Dict with all student fields.

    Returns:
        (is_valid: bool, errors: List[str])
    """
    errors = []

    # Name
    name = profile.get("name", "").strip()
    if not name:
        errors.append("Name cannot be empty.")
    elif len(name) > 100:
        errors.append("Name must be 100 characters or fewer.")

    # CGPA
    cgpa = profile.get("cgpa")
    if cgpa is None:
        errors.append("CGPA is required.")
    elif not (CGPA_RANGE[0] <= float(cgpa) <= CGPA_RANGE[1]):
        errors.append(f"CGPA must be between {CGPA_RANGE[0]} and {CGPA_RANGE[1]}.")

    # Hardworking level
    hw = profile.get("hardworking_level")
    if hw is None:
        errors.append("Hardworking level is required.")
    elif not (HARD_RANGE[0] <= int(hw) <= HARD_RANGE[1]):
        errors.append(f"Hardworking level must be between {HARD_RANGE[0]} and {HARD_RANGE[1]}.")

    # Proficiency scores
    for field in PROFICIENCY_FIELDS:
        val = profile.get(field)
        if val is None:
            errors.append(f"{field.replace('_', ' ').title()} is required.")
        elif not (SCORE_RANGE[0] <= float(val) <= SCORE_RANGE[1]):
            errors.append(
                f"{field.replace('_', ' ').title()} must be between "
                f"{SCORE_RANGE[0]} and {SCORE_RANGE[1]}."
            )

    return len(errors) == 0, errors


def sanitize_float(value: Optional[float], default: float = 5.0, min_val: float = 0.0, max_val: float = 10.0) -> float:
    """Clamp a float to a safe range, returning default if None."""
    if value is None:
        return default
    return float(max(min_val, min(max_val, value)))


def build_student_dict(
    name: str,
    cgpa: float,
    hardworking_level: int,
    cs_proficiency: float,
    math_proficiency: float,
    stat_proficiency: float,
    is_proficiency: float,
    se_proficiency: float,
    year: int = 2,
    credits_completed: int = 60,
) -> Dict:
    """
    Construct a normalized student profile dict for inference.
    Uses sanitize_float to prevent out-of-range values from crashing models.
    """
    return {
        "name":               name.strip(),
        "cgpa":               sanitize_float(cgpa),
        "hardworking_level":  max(1, min(10, int(hardworking_level))),
        "cs_proficiency":     sanitize_float(cs_proficiency),
        "math_proficiency":   sanitize_float(math_proficiency),
        "stat_proficiency":   sanitize_float(stat_proficiency),
        "is_proficiency":     sanitize_float(is_proficiency),
        "se_proficiency":     sanitize_float(se_proficiency),
        "year":               max(1, min(4, int(year))),
        "credits_completed":  max(0, int(credits_completed)),
    }
