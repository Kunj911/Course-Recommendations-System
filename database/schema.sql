-- Course Recommendation System Database Schema

CREATE TABLE IF NOT EXISTS students (
    student_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    cgpa REAL CHECK(cgpa >= 0 AND cgpa <= 10),
    hardworking_level INTEGER CHECK(hardworking_level >= 1 AND hardworking_level <= 10),
    cs_proficiency REAL CHECK(cs_proficiency >= 0 AND cs_proficiency <= 10),
    math_proficiency REAL CHECK(math_proficiency >= 0 AND math_proficiency <= 10),
    stat_proficiency REAL CHECK(stat_proficiency >= 0 AND stat_proficiency <= 10),
    is_proficiency REAL CHECK(is_proficiency >= 0 AND is_proficiency <= 10),
    se_proficiency REAL CHECK(se_proficiency >= 0 AND se_proficiency <= 10),
    credits_completed INTEGER DEFAULT 0,
    year INTEGER CHECK(year >= 1 AND year <= 4)
);

CREATE TABLE IF NOT EXISTS courses (
    course_id INTEGER PRIMARY KEY,
    course_code TEXT UNIQUE NOT NULL,
    course_name TEXT NOT NULL,
    department TEXT CHECK(department IN ('CS', 'MATH', 'STAT', 'IS', 'SE')),
    credits INTEGER DEFAULT 3,
    difficulty_level REAL CHECK(difficulty_level >= 0 AND difficulty_level <= 10),
    avg_workload_hours REAL,
    prerequisite_course_id INTEGER,
    is_advanced BOOLEAN DEFAULT 0,
    FOREIGN KEY (prerequisite_course_id) REFERENCES courses(course_id)
);

CREATE TABLE IF NOT EXISTS enrollments (
    enrollment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER NOT NULL,
    course_id INTEGER NOT NULL,
    semester TEXT NOT NULL,
    grade REAL CHECK(grade >= 0 AND grade <= 10),
    passed BOOLEAN,
    hours_spent REAL,
    FOREIGN KEY (student_id) REFERENCES students(student_id),
    FOREIGN KEY (course_id) REFERENCES courses(course_id),
    UNIQUE(student_id, course_id, semester)
);

-- Indexes for query performance
CREATE INDEX IF NOT EXISTS idx_enrollments_student ON enrollments(student_id);
CREATE INDEX IF NOT EXISTS idx_enrollments_course ON enrollments(course_id);
CREATE INDEX IF NOT EXISTS idx_courses_department ON courses(department);
CREATE INDEX IF NOT EXISTS idx_courses_prerequisite ON courses(prerequisite_course_id);
