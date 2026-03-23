"""
Synthetic Data Generator
========================
Generates realistic student, course, and enrollment data for the
Course Recommendation System.

Design decisions:
- Student proficiencies are drawn from correlated but distinct
  distributions so that a student can be strong in CS but weak in Math.
- Hardworking level acts as a compensatory factor that partially offsets
  low proficiency when computing grades.
- ~20 % of courses have a prerequisite chain.
"""

import sqlite3
import os
import random
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "database", "course_recommendation.db")

NUM_STUDENTS = 500
PASS_THRESHOLD = 5.0
SEMESTERS = ["Fall 2023", "Spring 2024", "Fall 2024", "Spring 2025"]

# Reproducible results
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Course catalogue – 60 courses across 5 departments
# ---------------------------------------------------------------------------
COURSE_CATALOGUE: list[dict] = [
    # ── CS (15) ── 3 basic, 8 intermediate, 4 advanced ──
    {"code": "CS101", "name": "Introduction to Programming",     "dept": "CS", "diff": 2.5, "wl": 4,  "prereq": None,    "adv": False},
    {"code": "CS102", "name": "Digital Logic Design",            "dept": "CS", "diff": 3.0, "wl": 4,  "prereq": None,    "adv": False},
    {"code": "CS103", "name": "Computer Fundamentals",           "dept": "CS", "diff": 2.0, "wl": 3,  "prereq": None,    "adv": False},
    {"code": "CS201", "name": "Data Structures",                 "dept": "CS", "diff": 5.0, "wl": 7,  "prereq": "CS101", "adv": False},
    {"code": "CS202", "name": "Object Oriented Programming",     "dept": "CS", "diff": 4.5, "wl": 6,  "prereq": "CS101", "adv": False},
    {"code": "CS203", "name": "Computer Architecture",           "dept": "CS", "diff": 5.5, "wl": 6,  "prereq": "CS102", "adv": False},
    {"code": "CS204", "name": "Database Systems",                "dept": "CS", "diff": 5.0, "wl": 6,  "prereq": None,    "adv": False},
    {"code": "CS205", "name": "Operating Systems",               "dept": "CS", "diff": 6.0, "wl": 7,  "prereq": "CS201", "adv": False},
    {"code": "CS206", "name": "Computer Networks",               "dept": "CS", "diff": 5.5, "wl": 6,  "prereq": None,    "adv": False},
    {"code": "CS301", "name": "Algorithms",                      "dept": "CS", "diff": 7.0, "wl": 8,  "prereq": "CS201", "adv": False},
    {"code": "CS350", "name": "Web Development",                 "dept": "CS", "diff": 4.0, "wl": 6,  "prereq": None,    "adv": False},
    {"code": "CS401", "name": "Machine Learning",                "dept": "CS", "diff": 8.0, "wl": 10, "prereq": "CS301", "adv": True},
    {"code": "CS402", "name": "Artificial Intelligence",         "dept": "CS", "diff": 8.5, "wl": 10, "prereq": "CS301", "adv": True},
    {"code": "CS403", "name": "Advanced Algorithms",             "dept": "CS", "diff": 9.0, "wl": 10, "prereq": "CS301", "adv": True},
    {"code": "CS404", "name": "Computer Vision",                 "dept": "CS", "diff": 8.5, "wl": 9,  "prereq": "CS401", "adv": True},

    # ── MATH (12) ── 4 basic, 6 intermediate, 2 advanced ──
    {"code": "MATH101", "name": "Calculus I",                    "dept": "MATH", "diff": 3.0, "wl": 5,  "prereq": None,      "adv": False},
    {"code": "MATH102", "name": "Calculus II",                   "dept": "MATH", "diff": 4.0, "wl": 5,  "prereq": "MATH101", "adv": False},
    {"code": "MATH103", "name": "Linear Algebra",                "dept": "MATH", "diff": 3.5, "wl": 5,  "prereq": None,      "adv": False},
    {"code": "MATH104", "name": "Discrete Mathematics",          "dept": "MATH", "diff": 3.0, "wl": 4,  "prereq": None,      "adv": False},
    {"code": "MATH201", "name": "Multivariable Calculus",        "dept": "MATH", "diff": 5.5, "wl": 6,  "prereq": "MATH102", "adv": False},
    {"code": "MATH202", "name": "Differential Equations",        "dept": "MATH", "diff": 6.0, "wl": 7,  "prereq": "MATH102", "adv": False},
    {"code": "MATH203", "name": "Abstract Algebra",              "dept": "MATH", "diff": 6.5, "wl": 7,  "prereq": "MATH103", "adv": False},
    {"code": "MATH204", "name": "Number Theory",                 "dept": "MATH", "diff": 5.5, "wl": 6,  "prereq": "MATH104", "adv": False},
    {"code": "MATH205", "name": "Numerical Methods",             "dept": "MATH", "diff": 5.0, "wl": 6,  "prereq": None,      "adv": False},
    {"code": "MATH206", "name": "Mathematical Modelling",        "dept": "MATH", "diff": 5.5, "wl": 6,  "prereq": None,      "adv": False},
    {"code": "MATH401", "name": "Real Analysis",                 "dept": "MATH", "diff": 8.5, "wl": 9,  "prereq": "MATH201", "adv": True},
    {"code": "MATH402", "name": "Complex Analysis",              "dept": "MATH", "diff": 8.0, "wl": 9,  "prereq": "MATH201", "adv": True},

    # ── STAT (10) ── 3 basic, 5 intermediate, 2 advanced ──
    {"code": "STAT101", "name": "Introduction to Statistics",    "dept": "STAT", "diff": 2.5, "wl": 4,  "prereq": None,      "adv": False},
    {"code": "STAT102", "name": "Probability Theory",            "dept": "STAT", "diff": 3.5, "wl": 5,  "prereq": None,      "adv": False},
    {"code": "STAT103", "name": "Descriptive Statistics",        "dept": "STAT", "diff": 2.0, "wl": 3,  "prereq": None,      "adv": False},
    {"code": "STAT201", "name": "Statistical Inference",         "dept": "STAT", "diff": 5.5, "wl": 6,  "prereq": "STAT101", "adv": False},
    {"code": "STAT202", "name": "Regression Analysis",           "dept": "STAT", "diff": 5.0, "wl": 6,  "prereq": "STAT101", "adv": False},
    {"code": "STAT203", "name": "Bayesian Statistics",           "dept": "STAT", "diff": 6.0, "wl": 7,  "prereq": "STAT102", "adv": False},
    {"code": "STAT204", "name": "Experimental Design",           "dept": "STAT", "diff": 5.5, "wl": 6,  "prereq": None,      "adv": False},
    {"code": "STAT310", "name": "Data Analysis",                 "dept": "STAT", "diff": 5.0, "wl": 6,  "prereq": None,      "adv": False},
    {"code": "STAT401", "name": "Advanced Statistical Learning", "dept": "STAT", "diff": 8.5, "wl": 9,  "prereq": "STAT201", "adv": True},
    {"code": "STAT402", "name": "Time Series Analysis",          "dept": "STAT", "diff": 7.5, "wl": 8,  "prereq": "STAT202", "adv": True},

    # ── IS (12) ── 4 basic, 6 intermediate, 2 advanced ──
    {"code": "IS101", "name": "Information Systems Fundamentals", "dept": "IS", "diff": 2.0, "wl": 3,  "prereq": None,    "adv": False},
    {"code": "IS102", "name": "Business Process Management",      "dept": "IS", "diff": 2.5, "wl": 4,  "prereq": None,    "adv": False},
    {"code": "IS103", "name": "IT Infrastructure",                "dept": "IS", "diff": 3.0, "wl": 4,  "prereq": None,    "adv": False},
    {"code": "IS104", "name": "Data Management Basics",           "dept": "IS", "diff": 2.5, "wl": 4,  "prereq": None,    "adv": False},
    {"code": "IS201", "name": "Enterprise Systems",               "dept": "IS", "diff": 5.0, "wl": 6,  "prereq": "IS101", "adv": False},
    {"code": "IS202", "name": "Systems Analysis & Design",        "dept": "IS", "diff": 5.5, "wl": 6,  "prereq": "IS101", "adv": False},
    {"code": "IS203", "name": "IT Project Management",            "dept": "IS", "diff": 4.5, "wl": 5,  "prereq": None,    "adv": False},
    {"code": "IS204", "name": "Cybersecurity Fundamentals",       "dept": "IS", "diff": 5.0, "wl": 6,  "prereq": None,    "adv": False},
    {"code": "IS320", "name": "System Design",                    "dept": "IS", "diff": 6.0, "wl": 7,  "prereq": "IS202", "adv": False},
    {"code": "IS205", "name": "Cloud Computing",                  "dept": "IS", "diff": 5.5, "wl": 6,  "prereq": None,    "adv": False},
    {"code": "IS401", "name": "Enterprise Architecture",          "dept": "IS", "diff": 8.0, "wl": 9,  "prereq": "IS201", "adv": True},
    {"code": "IS402", "name": "Digital Transformation Strategy",  "dept": "IS", "diff": 7.5, "wl": 8,  "prereq": "IS202", "adv": True},

    # ── SE (11) ── 3 basic, 6 intermediate, 2 advanced ──
    {"code": "SE101", "name": "Software Engineering Principles",  "dept": "SE", "diff": 2.5, "wl": 4,  "prereq": None,    "adv": False},
    {"code": "SE102", "name": "Version Control & Collaboration",  "dept": "SE", "diff": 2.0, "wl": 3,  "prereq": None,    "adv": False},
    {"code": "SE103", "name": "Requirements Engineering",         "dept": "SE", "diff": 3.0, "wl": 4,  "prereq": None,    "adv": False},
    {"code": "SE201", "name": "Software Design Patterns",         "dept": "SE", "diff": 5.0, "wl": 6,  "prereq": "SE101", "adv": False},
    {"code": "SE202", "name": "Software Testing & QA",            "dept": "SE", "diff": 4.5, "wl": 5,  "prereq": "SE101", "adv": False},
    {"code": "SE203", "name": "Agile Development",                "dept": "SE", "diff": 4.0, "wl": 5,  "prereq": None,    "adv": False},
    {"code": "SE204", "name": "DevOps & CI/CD",                   "dept": "SE", "diff": 5.5, "wl": 6,  "prereq": "SE102", "adv": False},
    {"code": "SE205", "name": "Mobile App Development",           "dept": "SE", "diff": 5.0, "wl": 6,  "prereq": None,    "adv": False},
    {"code": "SE206", "name": "API Design & Microservices",       "dept": "SE", "diff": 5.5, "wl": 7,  "prereq": None,    "adv": False},
    {"code": "SE401", "name": "Software Architecture",            "dept": "SE", "diff": 8.0, "wl": 9,  "prereq": "SE201", "adv": True},
    {"code": "SE402", "name": "Software Reliability Engineering", "dept": "SE", "diff": 7.5, "wl": 8,  "prereq": "SE202", "adv": True},
]

# Map department → proficiency column name for clarity
DEPT_PROFICIENCY_KEY = {
    "CS":   "cs_proficiency",
    "MATH": "math_proficiency",
    "STAT": "stat_proficiency",
    "IS":   "is_proficiency",
    "SE":   "se_proficiency",
}

FIRST_NAMES = [
    "Aarav", "Aditi", "Aisha", "Akira", "Alex", "Amara", "Ananya", "Arjun",
    "Beatriz", "Carlos", "Chen", "Daria", "Deepak", "Elena", "Fatima",
    "Gabriel", "Hana", "Ibrahim", "Ines", "Javier", "Kavya", "Kenji",
    "Liam", "Maria", "Mei", "Nadia", "Nikhil", "Olga", "Omar", "Priya",
    "Raj", "Riya", "Samuel", "Sara", "Tariq", "Uma", "Viktor", "Wei",
    "Yuki", "Zara", "Rohan", "Simran", "Tanvi", "Vivaan", "Yash", "Neha",
    "Ishaan", "Pooja", "Siddharth", "Divya",
]
LAST_NAMES = [
    "Patel", "Sharma", "Kumar", "Singh", "Wang", "Li", "Chen", "Tanaka",
    "Garcia", "Rodriguez", "Martinez", "Lopez", "Hernandez", "Kim", "Park",
    "Nguyen", "Johnson", "Williams", "Brown", "Davis", "Wilson", "Moore",
    "Taylor", "Anderson", "Thomas", "Gupta", "Reddy", "Shah", "Desai",
    "Mehta", "Iyer", "Rao", "Nair", "Joshi", "Mishra", "Verma",
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _random_name() -> str:
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"


def _generate_proficiencies() -> dict[str, float]:
    """Generate correlated but distinct proficiency scores.

    Students tend to be strong in a cluster of related fields while
    having genuine weaknesses in others.
    """
    # Pick a 'strength cluster' – one or two related fields get a boost
    base = np.random.normal(loc=5.5, scale=1.5, size=5)
    # Add a strong spike in 1-2 random departments
    strong_depts = random.sample(range(5), k=random.randint(1, 2))
    for idx in strong_depts:
        base[idx] += np.random.uniform(1.5, 3.0)
    # Optionally suppress 1 department
    if random.random() < 0.5:
        weak = random.choice([i for i in range(5) if i not in strong_depts])
        base[weak] -= np.random.uniform(1.0, 2.5)
    # Clip to valid range
    base = np.clip(base, 1.0, 10.0).round(1)
    keys = ["cs_proficiency", "math_proficiency", "stat_proficiency",
            "is_proficiency", "se_proficiency"]
    return dict(zip(keys, base))


def _compute_grade(proficiency: float, hardworking: int, cgpa: float,
                   difficulty: float) -> float:
    """Compute a realistic grade using the formula from the spec."""
    raw = (proficiency * 0.6
           + hardworking * 0.2
           + cgpa * 0.2
           - difficulty * 0.3
           + np.random.normal(0, 0.8))
    return round(float(np.clip(raw, 0.0, 10.0)), 1)


# ---------------------------------------------------------------------------
# Main generation functions
# ---------------------------------------------------------------------------

def generate_students(conn: sqlite3.Connection) -> list[dict]:
    """Insert 500 students with varied proficiency profiles."""
    students = []
    for sid in range(1, NUM_STUDENTS + 1):
        profs = _generate_proficiencies()
        student = {
            "student_id": sid,
            "name": _random_name(),
            "cgpa": round(np.random.normal(6.5, 1.5), 1),
            "hardworking_level": int(np.clip(np.random.normal(5.5, 2), 1, 10)),
            **profs,
            "credits_completed": random.choice(range(0, 121, 3)),
            "year": random.randint(1, 4),
        }
        # Keep CGPA realistic
        student["cgpa"] = float(np.clip(student["cgpa"], 2.0, 10.0))
        students.append(student)

    conn.executemany(
        """INSERT INTO students
           (student_id, name, cgpa, hardworking_level,
            cs_proficiency, math_proficiency, stat_proficiency,
            is_proficiency, se_proficiency, credits_completed, year)
           VALUES (:student_id, :name, :cgpa, :hardworking_level,
                   :cs_proficiency, :math_proficiency, :stat_proficiency,
                   :is_proficiency, :se_proficiency, :credits_completed, :year)""",
        students,
    )
    conn.commit()
    print(f"[OK] Inserted {len(students)} students")
    return students


def generate_courses(conn: sqlite3.Connection) -> dict[str, int]:
    """Insert all 60 courses. Returns a code ➜ course_id mapping."""
    # First pass: insert courses without prerequisites
    code_to_id: dict[str, int] = {}
    for idx, c in enumerate(COURSE_CATALOGUE, start=1):
        conn.execute(
            """INSERT INTO courses
               (course_id, course_code, course_name, department, credits,
                difficulty_level, avg_workload_hours, prerequisite_course_id, is_advanced)
               VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?)""",
            (idx, c["code"], c["name"], c["dept"], 3,
             c["diff"], c["wl"], int(c["adv"])),
        )
        code_to_id[c["code"]] = idx
    conn.commit()

    # Second pass: wire prerequisite foreign keys
    for c in COURSE_CATALOGUE:
        if c["prereq"] and c["prereq"] in code_to_id:
            conn.execute(
                "UPDATE courses SET prerequisite_course_id = ? WHERE course_code = ?",
                (code_to_id[c["prereq"]], c["code"]),
            )
    conn.commit()
    print(f"[OK] Inserted {len(COURSE_CATALOGUE)} courses")
    return code_to_id


def generate_enrollments(conn: sqlite3.Connection,
                         students: list[dict],
                         code_to_id: dict[str, int]) -> int:
    """Generate ≥ 15 000 realistic enrollment records."""
    # Build lookup: course_id → catalogue entry
    id_to_cat = {code_to_id[c["code"]]: c for c in COURSE_CATALOGUE}
    dept_courses = {}
    for c in COURSE_CATALOGUE:
        dept_courses.setdefault(c["dept"], []).append(code_to_id[c["code"]])

    enrollments: list[tuple] = []

    for student in students:
        # Each student takes 25-40 courses across semesters
        num_courses = random.randint(25, 40)
        available_ids = list(id_to_cat.keys())
        random.shuffle(available_ids)
        chosen = available_ids[:num_courses]

        for course_id in chosen:
            cat = id_to_cat[course_id]
            dept = cat["dept"]
            prof_key = DEPT_PROFICIENCY_KEY[dept]
            proficiency = student[prof_key]

            grade = _compute_grade(
                proficiency,
                student["hardworking_level"],
                student["cgpa"],
                cat["diff"],
            )
            passed = grade >= PASS_THRESHOLD
            hours = round(cat["wl"] + np.random.normal(0, 1.5), 1)
            hours = max(1.0, hours)
            semester = random.choice(SEMESTERS)

            enrollments.append((
                student["student_id"],
                course_id,
                semester,
                grade,
                int(passed),
                hours,
            ))

    # Bulk insert – ignore duplicates from random collisions
    conn.executemany(
        """INSERT OR IGNORE INTO enrollments
           (student_id, course_id, semester, grade, passed, hours_spent)
           VALUES (?, ?, ?, ?, ?, ?)""",
        enrollments,
    )
    conn.commit()
    count = conn.execute("SELECT COUNT(*) FROM enrollments").fetchone()[0]
    print(f"[OK] Inserted {count} enrollment records")
    return count


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(
            f"Database not found at {DB_PATH}. Run setup_database.py first."
        )

    conn = sqlite3.connect(DB_PATH)
    students = generate_students(conn)
    code_to_id = generate_courses(conn)
    generate_enrollments(conn, students, code_to_id)
    conn.close()
    print("\n[DONE] Data generation complete!")


if __name__ == "__main__":
    main()
