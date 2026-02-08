# setup_database.py

import sqlite3
import os

# Create database directory if it doesn't exist
os.makedirs('database', exist_ok=True)

# Connect to database (creates it if it doesn't exist)
conn = sqlite3.connect('database/course_recommendation.db')
cursor = conn.cursor()

# Create students table (CGPA 0-10 scale)
cursor.execute('''
CREATE TABLE IF NOT EXISTS students (
    student_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    major TEXT NOT NULL,
    cgpa REAL NOT NULL CHECK(cgpa >= 0.0 AND cgpa <= 10.0),
    credits_completed INTEGER DEFAULT 0,
    year INTEGER CHECK(year IN (1, 2, 3, 4))
)
''')

# Create courses table
cursor.execute('''
CREATE TABLE IF NOT EXISTS courses (
    course_id INTEGER PRIMARY KEY,
    course_code TEXT UNIQUE NOT NULL,
    course_name TEXT NOT NULL,
    department TEXT NOT NULL,
    credits INTEGER DEFAULT 3,
    difficulty_level REAL CHECK(difficulty_level >= 1.0 AND difficulty_level <= 10.0),
    avg_workload_hours REAL,
    prerequisite_course_id INTEGER,
    FOREIGN KEY (prerequisite_course_id) REFERENCES courses(course_id)
)
''')

# Create enrollments table (grades on 0-10 scale)
cursor.execute('''
CREATE TABLE IF NOT EXISTS enrollments (
    enrollment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER NOT NULL,
    course_id INTEGER NOT NULL,
    semester TEXT NOT NULL,
    grade REAL CHECK(grade >= 0.0 AND grade <= 10.0),
    passed INTEGER CHECK(passed IN (0, 1)),
    hours_spent REAL,
    FOREIGN KEY (student_id) REFERENCES students(student_id),
    FOREIGN KEY (course_id) REFERENCES courses(course_id),
    UNIQUE(student_id, course_id, semester)
)
''')

conn.commit()
conn.close()

print("✓ Database created successfully at database/course_recommendation.db")
print("✓ Tables created: students, courses, enrollments")
print("✓ CGPA scale: 0-10")