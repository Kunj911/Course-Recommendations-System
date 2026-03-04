
-- Students: Core academic profile with subject-specific proficiencies
CREATE TABLE IF NOT EXISTS students (
    student_id          INTEGER PRIMARY KEY,
    name                TEXT NOT NULL,
    cgpa                REAL NOT NULL CHECK(cgpa BETWEEN 0 AND 10),
    hardworking_level   INTEGER NOT NULL CHECK(hardworking_level BETWEEN 1 AND 10),
    cs_proficiency      REAL NOT NULL CHECK(cs_proficiency BETWEEN 0 AND 10),
    math_proficiency    REAL NOT NULL CHECK(math_proficiency BETWEEN 0 AND 10),
    stat_proficiency    REAL NOT NULL CHECK(stat_proficiency BETWEEN 0 AND 10),
    is_proficiency      REAL NOT NULL CHECK(is_proficiency BETWEEN 0 AND 10),
    se_proficiency      REAL NOT NULL CHECK(se_proficiency BETWEEN 0 AND 10),
    credits_completed   INTEGER NOT NULL DEFAULT 0,
    year                INTEGER NOT NULL CHECK(year BETWEEN 1 AND 4)
);

-- Courses: Academic catalog with department classification and prerequisites
CREATE TABLE IF NOT EXISTS courses (
    course_id               INTEGER PRIMARY KEY,
    course_code             TEXT UNIQUE NOT NULL,
    course_name             TEXT NOT NULL,
    department              TEXT NOT NULL CHECK(department IN ('CS', 'MATH', 'STAT', 'IS', 'SE')),
    credits                 INTEGER NOT NULL,
    difficulty_level        REAL NOT NULL CHECK(difficulty_level BETWEEN 0 AND 10),
    avg_workload_hours      REAL NOT NULL,
    prerequisite_course_id  INTEGER REFERENCES courses(course_id),
    is_advanced             BOOLEAN NOT NULL DEFAULT FALSE
);

-- Enrollments: Student-course interaction history for model training
CREATE TABLE IF NOT EXISTS enrollments (
    enrollment_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id      INTEGER NOT NULL REFERENCES students(student_id),
    course_id       INTEGER NOT NULL REFERENCES courses(course_id),
    semester        TEXT NOT NULL,
    grade           REAL CHECK(grade BETWEEN 0 AND 10),
    passed          BOOLEAN,
    hours_spent     REAL,
    UNIQUE(student_id, course_id, semester)
);

CREATE INDEX IF NOT EXISTS idx_enrollments_student ON enrollments(student_id);
CREATE INDEX IF NOT EXISTS idx_enrollments_course ON enrollments(course_id);
CREATE INDEX IF NOT EXISTS idx_courses_department ON courses(department);
