# data_generator.py

import sqlite3
import numpy as np
import pandas as pd
from faker import Faker
import random

fake = Faker()
random.seed(42)
np.random.seed(42)

# Database connection
conn = sqlite3.connect('database/course_recommendation.db')
cursor = conn.cursor()

# Clear existing data (if re-running)
cursor.execute('DELETE FROM enrollments')
cursor.execute('DELETE FROM students')
cursor.execute('DELETE FROM courses')
conn.commit()

# --- Generate Students with Realistic CGPA Distribution (0-10 scale) ---
print("Generating students...")
majors = ['Computer Science', 'Data Science', 'Information Systems', 
          'Software Engineering', 'Cybersecurity']
students_data = []

for i in range(1, 501):  # 500 students
    # Create realistic CGPA distribution
    # 70% students: 6.0-8.5 (average to good)
    # 20% students: 8.5-9.5 (excellent)
    # 10% students: 4.0-6.0 (struggling)
    
    rand = random.random()
    if rand < 0.70:  # Average to good students
        cgpa = round(np.random.uniform(6.0, 8.5), 2)
    elif rand < 0.90:  # Excellent students
        cgpa = round(np.random.uniform(8.5, 9.5), 2)
    else:  # Struggling students
        cgpa = round(np.random.uniform(4.0, 6.0), 2)
    
    year = random.randint(1, 4)
    credits_completed = random.randint(
        year * 15 - 10,  # Minimum credits for year
        year * 35 + 10   # Maximum credits for year
    )
    
    students_data.append((
        i,
        fake.name(),
        random.choice(majors),
        cgpa,
        credits_completed,
        year
    ))

cursor.executemany('''
    INSERT INTO students (student_id, name, major, cgpa, credits_completed, year)
    VALUES (?, ?, ?, ?, ?, ?)
''', students_data)

# --- Generate Courses with Varied Difficulty (0-10 scale) ---
print("Generating courses...")
departments = ['CS', 'MATH', 'STAT', 'IS', 'SE']
course_templates = {
    'CS': [
        ('Introduction to Programming', 3.0, 8.0),
        ('Data Structures', 5.5, 12.0),
        ('Algorithms', 7.0, 14.0),
        ('Database Systems', 6.0, 10.0),
        ('Web Development', 4.5, 9.0),
        ('Machine Learning', 8.5, 16.0),
        ('Computer Networks', 6.5, 11.0),
        ('Operating Systems', 7.5, 13.0),
        ('Software Engineering', 5.0, 10.0),
        ('Artificial Intelligence', 8.0, 15.0),
    ],
    'MATH': [
        ('Calculus I', 6.0, 10.0),
        ('Calculus II', 7.0, 12.0),
        ('Linear Algebra', 6.5, 11.0),
        ('Discrete Mathematics', 7.5, 13.0),
        ('Probability Theory', 8.0, 14.0),
    ],
    'STAT': [
        ('Statistics Fundamentals', 4.0, 8.0),
        ('Statistical Inference', 7.0, 12.0),
        ('Data Analysis', 5.5, 10.0),
        ('Regression Analysis', 7.5, 13.0),
    ],
    'IS': [
        ('Information Systems', 4.0, 7.0),
        ('Business Analytics', 5.5, 9.0),
        ('System Design', 6.0, 10.0),
        ('IT Project Management', 5.0, 9.0),
    ],
    'SE': [
        ('Software Architecture', 7.0, 12.0),
        ('Agile Development', 5.0, 9.0),
        ('DevOps Practices', 6.5, 11.0),
        ('Testing & QA', 5.5, 10.0),
    ]
}

courses_data = []
course_id = 1

for dept, course_list in course_templates.items():
    for course_name, difficulty, workload in course_list:
        course_code = f"{dept}{random.randint(100, 499)}"
        
        courses_data.append((
            course_id,
            course_code,
            course_name,
            dept,
            random.choice([3, 4]),
            difficulty,
            workload,
            random.choice([None] + list(range(1, max(1, course_id - 3))))
        ))
        course_id += 1

# Add some more courses to reach 50
while len(courses_data) < 50:
    dept = random.choice(departments)
    courses_data.append((
        course_id,
        f"{dept}{random.randint(100, 499)}",
        f"Advanced {dept} Topics {random.randint(1, 10)}",
        dept,
        random.choice([3, 4]),
        round(random.uniform(4.0, 9.0), 1),
        round(random.uniform(8.0, 16.0), 1),
        random.choice([None] + list(range(1, max(1, course_id - 5))))
    ))
    course_id += 1

cursor.executemany('''
    INSERT INTO courses (course_id, course_code, course_name, department, 
                         credits, difficulty_level, avg_workload_hours, prerequisite_course_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
''', courses_data)

# --- Generate Enrollments with REALISTIC Grade Variations ---
print("Generating enrollments...")
semesters = ['Fall2023', 'Spring2024', 'Fall2024', 'Spring2025']
enrollments_data = []

for student_id in range(1, 501):
    student = cursor.execute('SELECT cgpa, year FROM students WHERE student_id = ?', 
                            (student_id,)).fetchone()
    base_cgpa = student[0]
    student_year = student[1]
    
    # Each student takes 16-32 courses total
    num_courses = random.randint(16, 32)
    
    taken_courses = set()
    for _ in range(num_courses):
        course_id = random.randint(1, 50)
        if course_id in taken_courses:
            continue
        taken_courses.add(course_id)
        
        course = cursor.execute(
            'SELECT difficulty_level, avg_workload_hours FROM courses WHERE course_id = ?',
            (course_id,)
        ).fetchone()
        
        difficulty = course[0]
        workload = course[1]
        
        # REALISTIC GRADE CALCULATION
        # Base grade influenced by student CGPA
        base_grade = base_cgpa
        
        # Strong penalty/bonus based on difficulty gap
        difficulty_gap = base_cgpa - difficulty
        
        if difficulty_gap > 3:  # Course much easier than student level
            grade_adjustment = random.uniform(0.5, 1.5)  # Easy A
        elif difficulty_gap > 1:  # Course slightly easier
            grade_adjustment = random.uniform(0.2, 0.8)
        elif difficulty_gap > -1:  # Matched difficulty
            grade_adjustment = random.uniform(-0.3, 0.5)
        elif difficulty_gap > -3:  # Course harder
            grade_adjustment = random.uniform(-1.5, -0.2)
        else:  # Course much harder
            grade_adjustment = random.uniform(-3.0, -0.8)  # Struggle
        
        # Workload impact (high workload = more variance)
        if workload > 12:
            workload_factor = random.uniform(-0.8, 0.3)
        else:
            workload_factor = random.uniform(-0.3, 0.5)
        
        # Year progression bonus (seniors do better)
        year_bonus = (student_year - 2) * 0.3
        
        # Random variance for realism
        random_variance = np.random.normal(0, 0.8)
        
        # Final grade calculation
        final_grade = base_grade + grade_adjustment + workload_factor + year_bonus + random_variance
        
        # Add occasional outliers (student had personal issues or excelled)
        if random.random() < 0.05:  # 5% chance of extreme outcome
            final_grade += random.choice([-2.5, 3.0])
        
        # Clip to valid range
        final_grade = np.clip(final_grade, 0.0, 10.0)
        final_grade = round(final_grade, 2)
        
        # Pass threshold: 5.0 (50%)
        passed = 1 if final_grade >= 5.0 else 0
        
        # Hours spent correlates with difficulty and grade outcome
        if passed:
            hours_spent = workload + np.random.normal(0, 3)
        else:
            # Students who failed either overworked or underworked
            hours_spent = workload * random.choice([0.4, 1.6]) + np.random.normal(0, 4)
        
        hours_spent = max(2.0, hours_spent)  # Minimum 2 hours
        
        enrollments_data.append((
            student_id,
            course_id,
            random.choice(semesters),
            final_grade,
            passed,
            round(hours_spent, 1)
        ))

cursor.executemany('''
    INSERT INTO enrollments (student_id, course_id, semester, grade, passed, hours_spent)
    VALUES (?, ?, ?, ?, ?, ?)
''', enrollments_data)

conn.commit()
print(f"✓ Generated {len(students_data)} students")
print(f"✓ Generated {len(courses_data)} courses")
print(f"✓ Generated {len(enrollments_data)} enrollments")

# Print sample statistics
avg_cgpa = cursor.execute('SELECT AVG(cgpa) FROM students').fetchone()[0]
avg_grade = cursor.execute('SELECT AVG(grade) FROM enrollments').fetchone()[0]
pass_rate = cursor.execute('SELECT AVG(passed) * 100 FROM enrollments').fetchone()[0]

print(f"\n📊 Dataset Statistics:")
print(f"   Average Student CGPA: {avg_cgpa:.2f}/10")
print(f"   Average Course Grade: {avg_grade:.2f}/10")
print(f"   Overall Pass Rate: {pass_rate:.1f}%")

conn.close()