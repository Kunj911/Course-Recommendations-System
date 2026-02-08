# recommender.py

import pandas as pd
import sqlite3
import pickle
from models.feature_engineering import get_feature_columns

class CourseRecommender:
    def __init__(self, db_path='database/course_recommendation.db'):
        self.db_path = db_path
        
        # Load trained models
        with open('models/saved/logistic_model.pkl', 'rb') as f:
            self.log_model = pickle.load(f)
        
        with open('models/saved/linear_model.pkl', 'rb') as f:
            self.lin_model = pickle.load(f)
        
        with open('models/saved/scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        
        self.feature_cols = get_feature_columns()
    
    def recommend_courses(self, student_id, top_n=5):
        """
        Recommends top N courses for a given student.
        Returns: DataFrame with course details and predictions (0-10 scale)
        """
        conn = sqlite3.connect(self.db_path)
        
        # Get student info
        student = pd.read_sql_query(
            'SELECT * FROM students WHERE student_id = ?',
            conn, params=(student_id,)
        )
        
        if student.empty:
            conn.close()
            return None
        
        # Get courses NOT already taken by student
        already_taken = pd.read_sql_query(
            'SELECT course_id FROM enrollments WHERE student_id = ?',
            conn, params=(student_id,)
        )['course_id'].tolist()
        
        # Get all courses
        all_courses = pd.read_sql_query('SELECT * FROM courses', conn)
        conn.close()
        
        # Filter out already taken courses
        candidate_courses = all_courses[~all_courses['course_id'].isin(already_taken)]
        
        if candidate_courses.empty:
            return pd.DataFrame()
        
        # Build feature matrix for prediction
        predictions = []
        
        for _, course in candidate_courses.iterrows():
            cgpa = student['cgpa'].values[0]
            credits_comp = student['credits_completed'].values[0]
            year = student['year'].values[0]
            difficulty = course['difficulty_level']
            workload = course['avg_workload_hours']
            credits = course['credits']
            
            # Calculate difficulty tier
            if difficulty <= 4:
                difficulty_tier = 0
            elif difficulty <= 7:
                difficulty_tier = 1
            else:
                difficulty_tier = 2
            
            # Create feature vector
            features = {
                'cgpa': cgpa,
                'cgpa_normalized': cgpa / 10.0,
                'credits_completed': credits_comp,
                'year': year,
                'difficulty_level': difficulty,
                'difficulty_tier': difficulty_tier,
                'avg_workload_hours': workload,
                'credits': credits,
                'cgpa_difficulty_gap': cgpa - difficulty,
                'workload_capacity': credits_comp / (year + 1),
                'major_encoded': pd.Categorical([student['major'].values[0]]).codes[0],
                'department_encoded': pd.Categorical([course['department']]).codes[0],
                'cgpa_credits_interaction': cgpa * credits,
                'workload_cgpa_ratio': workload / (cgpa + 1),
                'experience_factor': year * credits_comp / 100.0
            }
            
            X = pd.DataFrame([features])[self.feature_cols]
            X_scaled = self.scaler.transform(X)
            
            # Predict success probability and expected grade
            success_prob = self.log_model.predict_proba(X_scaled)[0, 1]
            expected_grade = self.lin_model.predict(X_scaled)[0]
            expected_grade = max(0.0, min(10.0, expected_grade))  # Clip to 0-10
            
            # Recommendation score with better weighting
            # Prioritize high success AND high grade
            recommendation_score = (success_prob ** 0.7) * (expected_grade / 10.0) * 10
            
            predictions.append({
                'course_id': course['course_id'],
                'course_code': course['course_code'],
                'course_name': course['course_name'],
                'department': course['department'],
                'difficulty_level': difficulty,
                'success_probability': success_prob,
                'expected_grade': expected_grade,
                'recommendation_score': recommendation_score
            })
        
        # Convert to DataFrame and rank
        recommendations = pd.DataFrame(predictions)
        recommendations = recommendations.sort_values('recommendation_score', ascending=False)
        
        return recommendations.head(top_n)