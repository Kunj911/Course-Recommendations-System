# feature_engineering.py

import pandas as pd
import sqlite3

def create_features(db_path='database/course_recommendation.db'):
    """
    Creates feature matrix by joining students, courses, and enrollments.
    Returns: DataFrame with features and target variables (0-10 scale)
    """
    conn = sqlite3.connect(db_path)
    
    query = '''
    SELECT 
        e.enrollment_id,
        e.student_id,
        e.course_id,
        s.cgpa,
        s.credits_completed,
        s.year,
        s.major,
        c.difficulty_level,
        c.avg_workload_hours,
        c.credits,
        c.department,
        e.grade,
        e.passed,
        e.hours_spent
    FROM enrollments e
    JOIN students s ON e.student_id = s.student_id
    JOIN courses c ON e.course_id = c.course_id
    '''
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # --- Feature Engineering (adjusted for 0-10 scale) ---
    
    # 1. CGPA-Difficulty Gap (larger range now)
    df['cgpa_difficulty_gap'] = df['cgpa'] - df['difficulty_level']
    
    # 2. Workload Capacity
    df['workload_capacity'] = df['credits_completed'] / (df['year'] + 1)
    
    # 3. Normalized CGPA (0-1 scale for model stability)
    df['cgpa_normalized'] = df['cgpa'] / 10.0
    
    # 4. Difficulty Tier (categorical: easy/medium/hard)
    df['difficulty_tier'] = pd.cut(
        df['difficulty_level'], 
        bins=[0, 4, 7, 10], 
        labels=[0, 1, 2]  # 0=easy, 1=medium, 2=hard
    ).astype(int)
    
    # 5. Encode categorical features
    df['major_encoded'] = pd.Categorical(df['major']).codes
    df['department_encoded'] = pd.Categorical(df['department']).codes
    
    # 6. Interaction: CGPA * Course Credits
    df['cgpa_credits_interaction'] = df['cgpa'] * df['credits']
    
    # 7. Workload-to-CGPA ratio (students with high CGPA handle workload better)
    df['workload_cgpa_ratio'] = df['avg_workload_hours'] / (df['cgpa'] + 1)  # +1 to avoid division by zero
    
    # 8. Year-based experience factor
    df['experience_factor'] = df['year'] * df['credits_completed'] / 100.0
    
    return df

def get_feature_columns():
    """Returns list of feature column names for modeling"""
    return [
        'cgpa',
        'cgpa_normalized',
        'credits_completed',
        'year',
        'difficulty_level',
        'difficulty_tier',
        'avg_workload_hours',
        'credits',
        'cgpa_difficulty_gap',
        'workload_capacity',
        'major_encoded',
        'department_encoded',
        'cgpa_credits_interaction',
        'workload_cgpa_ratio',
        'experience_factor'
    ]