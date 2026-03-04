"""
data_generator.py
-----------------
Generates synthetic but realistic data for 500 students, 60 courses, and 15,000+ enrollments.

Key design decisions:
- Student proficiency profiles are correlated (CS students tend to have good Math too, but not always)
- Hardworking level acts as a genuine compensator for low proficiency
- Grade formula: (proficiency * 0.6) + (hardworking * 0.2) + (cgpa * 0.2) - (difficulty * 0.3) + noise
- Prerequisites are enforced: advanced courses only appear in enrollments after basic ones
"""

import sqlite3
import os
import random
import numpy as np
from typing import List, Tuple, Dict

DB_PATH = os.path.join(os.path.dirname(__file__), "database", "course_recommendation.db")

# Seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ─── Course Catalog ────────────────────────────────────────────────────────────
# Format: (course_code, course_name, department, credits, difficulty, avg_hours, prereq_code, is_advanced)
COURSE_CATALOG = [
    # CS – Basic (3)
    ("CS101", "Introduction to Programming",       "CS", 3, 3.0, 8,  None,    False),
    ("CS102", "Data Structures Fundamentals",      "CS", 3, 4.0, 10, None,    False),
    ("CS103", "Computer Organization",             "CS", 3, 3.5, 8,  None,    False),
    # CS – Intermediate (8)
    ("CS201", "Algorithms & Complexity",           "CS", 3, 6.0, 12, "CS102", False),
    ("CS202", "Object-Oriented Programming",       "CS", 3, 5.0, 10, "CS101", False),
    ("CS203", "Database Systems",                  "CS", 3, 5.5, 11, "CS101", False),
    ("CS204", "Operating Systems",                 "CS", 3, 6.5, 13, "CS103", False),
    ("CS205", "Computer Networks",                 "CS", 3, 6.0, 12, "CS103", False),
    ("CS301", "Advanced Algorithms",               "CS", 3, 7.5, 15, "CS201", False),
    ("CS302", "Web Development",                   "CS", 3, 5.0, 10, "CS202", False),
    ("CS303", "Software Engineering Principles",   "CS", 3, 5.5, 11, "CS202", False),
    # CS – Advanced (4)
    ("CS401", "Machine Learning",                  "CS", 4, 8.5, 18, "CS301", True),
    ("CS402", "Artificial Intelligence",           "CS", 4, 8.0, 17, "CS301", True),
    ("CS403", "Computer Vision",                   "CS", 4, 8.5, 18, "CS401", True),
    ("CS404", "Distributed Systems",               "CS", 4, 8.0, 17, "CS204", True),

    # MATH – Basic (4)
    ("MATH101", "Calculus I",                      "MATH", 3, 4.0, 10, None,      False),
    ("MATH102", "Calculus II",                     "MATH", 3, 5.0, 12, "MATH101", False),
    ("MATH103", "Linear Algebra",                  "MATH", 3, 5.5, 11, None,      False),
    ("MATH104", "Discrete Mathematics",            "MATH", 3, 4.5, 10, None,      False),
    # MATH – Intermediate (6)
    ("MATH201", "Probability Theory",              "MATH", 3, 6.0, 12, "MATH102", False),
    ("MATH202", "Multivariable Calculus",          "MATH", 3, 6.5, 13, "MATH102", False),
    ("MATH203", "Differential Equations",          "MATH", 3, 6.5, 13, "MATH102", False),
    ("MATH204", "Abstract Algebra",                "MATH", 3, 7.0, 14, "MATH103", False),
    ("MATH205", "Numerical Methods",               "MATH", 3, 6.0, 12, "MATH103", False),
    ("MATH301", "Optimization Theory",             "MATH", 3, 7.5, 15, "MATH201", False),
    # MATH – Advanced (2)
    ("MATH401", "Advanced Linear Algebra",         "MATH", 4, 8.5, 17, "MATH204", True),
    ("MATH402", "Mathematical Statistics",         "MATH", 4, 8.0, 16, "MATH201", True),

    # STAT – Basic (3)
    ("STAT101", "Introduction to Statistics",      "STAT", 3, 3.5, 8,  None,      False),
    ("STAT102", "Probability & Distributions",     "STAT", 3, 4.5, 10, "STAT101", False),
    ("STAT103", "Exploratory Data Analysis",       "STAT", 3, 4.0, 9,  None,      False),
    # STAT – Intermediate (5)
    ("STAT201", "Statistical Inference",           "STAT", 3, 6.0, 12, "STAT102", False),
    ("STAT202", "Regression Analysis",             "STAT", 3, 6.5, 13, "STAT102", False),
    ("STAT203", "Time Series Analysis",            "STAT", 3, 6.5, 13, "STAT201", False),
    ("STAT204", "Bayesian Statistics",             "STAT", 3, 7.0, 14, "STAT201", False),
    ("STAT301", "Data Analysis & Visualization",   "STAT", 3, 5.5, 11, "STAT202", False),
    # STAT – Advanced (2)
    ("STAT401", "Advanced Statistical Modeling",   "STAT", 4, 8.5, 17, "STAT204", True),
    ("STAT402", "Machine Learning Statistics",     "STAT", 4, 8.0, 16, "STAT202", True),

    # IS – Basic (4)
    ("IS101", "Introduction to Information Systems", "IS", 3, 3.0, 8, None,    False),
    ("IS102", "Business Process Management",         "IS", 3, 3.5, 8, None,    False),
    ("IS103", "IT Infrastructure",                   "IS", 3, 4.0, 9, None,    False),
    ("IS104", "Data Management Fundamentals",        "IS", 3, 4.0, 9, "IS101", False),
    # IS – Intermediate (6)
    ("IS201", "Systems Analysis & Design",           "IS", 3, 5.5, 11, "IS101", False),
    ("IS202", "Enterprise Systems",                  "IS", 3, 5.5, 11, "IS102", False),
    ("IS203", "Information Security",                "IS", 3, 6.0, 12, "IS103", False),
    ("IS204", "Cloud Computing",                     "IS", 3, 6.0, 12, "IS103", False),
    ("IS301", "System Design & Architecture",        "IS", 3, 7.0, 14, "IS201", False),
    ("IS302", "Data Warehousing",                    "IS", 3, 6.5, 13, "IS104", False),
    # IS – Advanced (2)
    ("IS401", "Advanced Enterprise Architecture",    "IS", 4, 8.0, 16, "IS301", True),
    ("IS402", "Big Data Analytics",                  "IS", 4, 8.5, 17, "IS302", True),

    # SE – Basic (3)
    ("SE101", "Software Development Lifecycle",  "SE", 3, 3.5, 8,  None,    False),
    ("SE102", "Version Control & DevOps",        "SE", 3, 4.0, 9,  None,    False),
    ("SE103", "Requirements Engineering",        "SE", 3, 3.5, 8,  None,    False),
    # SE – Intermediate (6)
    ("SE201", "Design Patterns",                 "SE", 3, 6.0, 12, "SE101", False),
    ("SE202", "Agile & Scrum Methodologies",     "SE", 3, 5.0, 10, "SE101", False),
    ("SE203", "Software Testing & QA",           "SE", 3, 5.5, 11, "SE103", False),
    ("SE204", "API Design & Microservices",      "SE", 3, 6.5, 13, "SE102", False),
    ("SE301", "Software Architecture Patterns",  "SE", 3, 7.0, 14, "SE201", False),
    ("SE302", "Performance Engineering",         "SE", 3, 6.5, 13, "SE203", False),
    # SE – Advanced (2)
    ("SE401", "Advanced Software Architecture", "SE", 4, 8.5, 17, "SE301", True),
    ("SE402", "Formal Methods in SE",           "SE", 4, 8.5, 17, "SE203", True),
]

# Student name pools
FIRST_NAMES = ["Aarav", "Priya", "Alex", "Meera", "Rohan", "Ananya", "Dev", "Nisha", "Arjun",
               "Kavya", "Nikhil", "Sneha", "Rahul", "Divya", "Kiran", "Pooja", "Vikram",
               "Shreya", "Aditya", "Riya", "Harsh", "Tanvi", "Yash", "Neha", "Siddharth",
               "Ishaan", "Zara", "Omar", "Leila", "Chen", "Wei", "Jun", "Sarah", "James",
               "Emma", "Liam", "Olivia", "Noah", "Ava", "Lucas", "Sofia", "Ethan"]
LAST_NAMES = ["Sharma", "Patel", "Kumar", "Singh", "Gupta", "Shah", "Mehta", "Verma",
              "Joshi", "Agarwal", "Reddy", "Nair", "Iyer", "Das", "Chatterjee", "Khan",
              "Ahmed", "Li", "Wang", "Zhang", "Johnson", "Williams", "Brown", "Davis",
              "Wilson", "Moore", "Taylor", "Anderson", "Thomas", "Jackson", "White"]


def generate_student_profiles(n: int = 500) -> List[Dict]:
    """
    Generate students with correlated proficiency profiles.
    Real students tend to cluster: CS-Math-Stat students vs IS-SE-focused students.
    We model 4 archetype clusters with noise to get realistic distributions.
    """
    archetypes = [
        # (cs, math, stat, is_, se) mean vectors
        {"cs": 8.0, "math": 7.5, "stat": 7.0, "is_": 5.0, "se": 6.0, "label": "theory_cs"},
        {"cs": 5.0, "math": 5.5, "stat": 4.5, "is_": 8.5, "se": 7.5, "label": "applied_is"},
        {"cs": 7.0, "math": 6.0, "stat": 7.5, "is_": 6.0, "se": 7.0, "label": "balanced"},
        {"cs": 5.5, "math": 4.5, "stat": 6.5, "is_": 6.5, "se": 8.5, "label": "se_focused"},
    ]
    weights = [0.25, 0.30, 0.25, 0.20]
    students = []

    used_names = set()
    for i in range(n):
        arch = random.choices(archetypes, weights=weights)[0]
        noise = lambda: np.random.normal(0, 1.5)

        def clamp(v): return round(float(np.clip(v, 1.0, 10.0)), 2)

        cs   = clamp(arch["cs"]   + noise())
        math = clamp(arch["math"] + noise())
        stat = clamp(arch["stat"] + noise())
        is_  = clamp(arch["is_"]  + noise())
        se   = clamp(arch["se"]   + noise())

        # CGPA correlates loosely with average proficiency + hardworking
        avg_prof = (cs + math + stat + is_ + se) / 5
        hardworking = random.randint(1, 10)
        cgpa = clamp(avg_prof * 0.7 + hardworking * 0.3 + np.random.normal(0, 0.8))

        year = random.randint(1, 4)
        # Credits increase with year, with some variance
        base_credits = (year - 1) * 30
        credits_completed = max(0, int(base_credits + np.random.normal(5, 8)))

        # Generate unique name
        attempts = 0
        while attempts < 100:
            name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
            if name not in used_names:
                used_names.add(name)
                break
            attempts += 1

        students.append({
            "student_id": i + 1,
            "name": name,
            "cgpa": cgpa,
            "hardworking_level": hardworking,
            "cs_proficiency": cs,
            "math_proficiency": math,
            "stat_proficiency": stat,
            "is_proficiency": is_,
            "se_proficiency": se,
            "credits_completed": credits_completed,
            "year": year,
        })

    return students


def insert_courses(conn: sqlite3.Connection) -> Dict[str, int]:
    """Insert course catalog and return mapping of course_code -> course_id."""
    cursor = conn.cursor()

    # First pass: insert courses without prerequisites (or store them)
    code_to_id: Dict[str, int] = {}
    prereq_map: Dict[str, str] = {}  # code -> prereq_code

    for idx, course in enumerate(COURSE_CATALOG, start=1):
        code, name, dept, credits, diff, hours, prereq_code, is_adv = course
        prereq_map[code] = prereq_code
        cursor.execute("""
            INSERT OR IGNORE INTO courses
              (course_id, course_code, course_name, department, credits,
               difficulty_level, avg_workload_hours, is_advanced)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (idx, code, name, dept, credits, diff, hours, int(is_adv)))
        code_to_id[code] = idx

    conn.commit()

    # Second pass: update prerequisite foreign keys
    for code, prereq_code in prereq_map.items():
        if prereq_code:
            prereq_id = code_to_id.get(prereq_code)
            if prereq_id:
                cursor.execute(
                    "UPDATE courses SET prerequisite_course_id = ? WHERE course_code = ?",
                    (prereq_id, code)
                )
    conn.commit()
    return code_to_id


def compute_grade(student: Dict, course: Dict, dept_proficiency: float) -> float:
    """
    Core grade formula incorporating proficiency, effort, and difficulty.
    Returns grade clamped to [1.0, 10.0].
    """
    base = (
        dept_proficiency          * 0.6 +
        student["hardworking_level"] * 0.2 +
        student["cgpa"]           * 0.2 -
        course["difficulty_level"] * 0.3 +
        np.random.normal(0, 0.8)    # realistic noise
    )
    return float(np.clip(round(base, 2), 1.0, 10.0))


def get_dept_proficiency(student: Dict, dept: str) -> float:
    """Return the relevant subject proficiency for a given department."""
    mapping = {
        "CS":   "cs_proficiency",
        "MATH": "math_proficiency",
        "STAT": "stat_proficiency",
        "IS":   "is_proficiency",
        "SE":   "se_proficiency",
    }
    return student[mapping[dept]]


def generate_enrollments(
    students: List[Dict],
    courses: List[Dict],
    code_to_id: Dict[str, int],
    target: int = 15000,
) -> List[Dict]:
    """
    Generate enrollments respecting prerequisite chains.
    Each student can only enroll in advanced courses if they've passed the prerequisite.
    """
    # Build prereq lookup: course_id -> prereq_course_id
    id_to_course = {c["course_id"]: c for c in courses}
    code_to_course = {c["course_code"]: c for c in courses}

    semesters = ["2021-Fall", "2022-Spring", "2022-Fall", "2023-Spring",
                 "2023-Fall", "2024-Spring", "2024-Fall", "2025-Spring"]

    enrollments = []
    enrollment_id = 1

    for student in students:
        # Track what courses this student has passed (for prereq checking)
        passed_course_ids: set = set()
        n_enrollments = random.randint(20, 45)  # Each student takes 20-45 courses
        sem_idx = 0

        # Categorize courses by prerequisite status for this student
        available_courses = []
        for course in courses:
            prereq_id = course.get("prerequisite_course_id")
            if prereq_id is None or prereq_id in passed_course_ids:
                available_courses.append(course)

        enrolled_this_student: set = set()
        sem_courses = []
        courses_per_sem = max(3, n_enrollments // len(semesters))

        for sem in semesters[:min(student["year"] * 2, len(semesters))]:
            # Recompute available based on what they've passed
            available = [
                c for c in courses
                if c["course_id"] not in enrolled_this_student
                and (
                    c.get("prerequisite_course_id") is None
                    or c.get("prerequisite_course_id") in passed_course_ids
                )
            ]
            if not available:
                break

            # Pick 3-6 courses per semester
            sem_count = min(random.randint(3, 6), len(available))
            chosen = random.sample(available, sem_count)

            for course in chosen:
                dept = course["department"]
                dept_prof = get_dept_proficiency(student, dept)
                grade = compute_grade(student, course, dept_prof)
                passed = grade >= 5.0

                hours = course["avg_workload_hours"] * (
                    1.0 + (student["hardworking_level"] - 5) * 0.05
                ) + np.random.normal(0, 1.5)
                hours = max(1.0, round(hours, 1))

                enrollments.append({
                    "enrollment_id": enrollment_id,
                    "student_id": student["student_id"],
                    "course_id": course["course_id"],
                    "semester": sem,
                    "grade": grade,
                    "passed": int(passed),
                    "hours_spent": hours,
                })
                enrollment_id += 1
                enrolled_this_student.add(course["course_id"])
                if passed:
                    passed_course_ids.add(course["course_id"])

    return enrollments


def run_data_generation() -> None:
    """Orchestrate full data generation pipeline."""
    from setup_database import setup_database
    setup_database()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    print("🎓 Generating student profiles...")
    students = generate_student_profiles(500)
    cursor.executemany("""
        INSERT OR REPLACE INTO students VALUES
        (:student_id, :name, :cgpa, :hardworking_level,
         :cs_proficiency, :math_proficiency, :stat_proficiency,
         :is_proficiency, :se_proficiency, :credits_completed, :year)
    """, students)
    conn.commit()
    print(f"  ✅ {len(students)} students inserted")

    print("📚 Inserting course catalog...")
    code_to_id = insert_courses(conn)
    print(f"  ✅ {len(code_to_id)} courses inserted")

    # Fetch courses back as dicts for enrollment generation
    cursor.execute("SELECT * FROM courses")
    cols = [d[0] for d in cursor.description]
    courses = [dict(zip(cols, row)) for row in cursor.fetchall()]

    print("📝 Generating enrollments (this may take a moment)...")
    enrollments = generate_enrollments(students, courses, code_to_id)
    cursor.executemany("""
        INSERT OR IGNORE INTO enrollments
        (enrollment_id, student_id, course_id, semester, grade, passed, hours_spent)
        VALUES (:enrollment_id, :student_id, :course_id, :semester, :grade, :passed, :hours_spent)
    """, enrollments)
    conn.commit()

    total = cursor.execute("SELECT COUNT(*) FROM enrollments").fetchone()[0]
    print(f"  ✅ {total} enrollments generated")
    conn.close()
    print("\n🏁 Data generation complete!")


if __name__ == "__main__":
    run_data_generation()
