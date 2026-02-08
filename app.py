# app.py

import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from models.recommender import CourseRecommender

import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Course Recommender", layout="wide", page_icon="🎓")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #555;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize recommender
@st.cache_resource
def load_recommender():
    return CourseRecommender()

recommender = load_recommender()

# --- Title ---
st.markdown('<div class="main-header">🎓 Course Recommendation System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Using Classical Machine Learning (Logistic + Linear Regression) | CGPA Scale: 0-10</div>', unsafe_allow_html=True)

# --- Sidebar: Student Selection ---
st.sidebar.header("📋 Student Selection")

conn = sqlite3.connect('database/course_recommendation.db')
students = pd.read_sql_query('SELECT student_id, name, major, cgpa FROM students ORDER BY name', conn)
conn.close()

student_options = {f"{row['name']} (ID: {row['student_id']}, CGPA: {row['cgpa']}/10)": row['student_id'] 
                   for _, row in students.iterrows()}

selected_student = st.sidebar.selectbox(
    "Choose a student:",
    options=list(student_options.keys())
)

student_id = student_options[selected_student]

# Number of recommendations
top_n = st.sidebar.slider("Number of recommendations:", 3, 10, 5)

# --- Display Student Profile ---
conn = sqlite3.connect('database/course_recommendation.db')
student_profile = pd.read_sql_query(
    'SELECT * FROM students WHERE student_id = ?',
    conn, params=(student_id,)
)

st.sidebar.markdown("---")
st.sidebar.subheader("👤 Student Profile")
st.sidebar.write(f"**Name:** {student_profile['name'].values[0]}")
st.sidebar.write(f"**Major:** {student_profile['major'].values[0]}")
st.sidebar.write(f"**CGPA:** {student_profile['cgpa'].values[0]}/10")
st.sidebar.write(f"**Year:** {student_profile['year'].values[0]}")
st.sidebar.write(f"**Credits:** {student_profile['credits_completed'].values[0]}")

conn.close()

# --- Main Content ---
if st.sidebar.button("🚀 Get Recommendations", type="primary", use_container_width=True):
    with st.spinner("🔄 Generating personalized recommendations..."):
        recommendations = recommender.recommend_courses(student_id, top_n)
    
    if recommendations is None or recommendations.empty:
        st.warning("⚠️ No courses available to recommend for this student.")
    else:
        st.success(f"✅ Top {len(recommendations)} Recommended Courses")
        
        # Display table with better formatting
        display_df = recommendations[[
            'course_code', 'course_name', 'department', 'difficulty_level',
            'success_probability', 'expected_grade', 'recommendation_score'
        ]].copy()
        
        display_df.columns = ['Code', 'Course Name', 'Dept', 'Difficulty (0-10)', 
                              'Success %', 'Expected Grade (0-10)', 'Score']
        display_df['Success %'] = display_df['Success %'].apply(lambda x: f"{x*100:.1f}%")
        display_df['Expected Grade (0-10)'] = display_df['Expected Grade (0-10)'].apply(lambda x: f"{x:.2f}")
        display_df['Score'] = display_df['Score'].apply(lambda x: f"{x:.2f}")
        display_df['Difficulty (0-10)'] = display_df['Difficulty (0-10)'].apply(lambda x: f"{x:.1f}")
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # --- Visualizations ---
        st.subheader("📊 Predicted Performance Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Success Probability Chart
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            courses = recommendations['course_code']
            probs = recommendations['success_probability'] * 100
            
            colors = ['#28a745' if p >= 75 else '#ffc107' if p >= 50 else '#dc3545' for p in probs]
            bars = ax1.barh(courses, probs, color=colors, edgecolor='black', linewidth=1.2)
            
            # Add value labels
            for i, (bar, prob) in enumerate(zip(bars, probs)):
                ax1.text(prob + 2, i, f'{prob:.1f}%', va='center', fontweight='bold', fontsize=10)
            
            ax1.set_xlabel('Success Probability (%)', fontsize=13, fontweight='bold')
            ax1.set_title('Predicted Success Probability', fontsize=15, fontweight='bold', pad=20)
            ax1.set_xlim(0, 105)
            ax1.grid(axis='x', alpha=0.3, linestyle='--')
            ax1.axvline(50, color='red', linestyle='--', linewidth=1, alpha=0.5, label='50% Threshold')
            ax1.legend()
            plt.tight_layout()
            st.pyplot(fig1)
        
        with col2:
            # Expected Grades Chart
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            grades = recommendations['expected_grade']
            
            grade_colors = ['#28a745' if g >= 7.5 else '#17a2b8' if g >= 6.0 else '#ffc107' if g >= 5.0 else '#dc3545' for g in grades]
            bars2 = ax2.barh(courses, grades, color=grade_colors, edgecolor='black', linewidth=1.2)
            
            # Add value labels
            for i, (bar, grade) in enumerate(zip(bars2, grades)):
                ax2.text(grade + 0.2, i, f'{grade:.2f}', va='center', fontweight='bold', fontsize=10)
            
            ax2.set_xlabel('Expected Grade (0-10 scale)', fontsize=13, fontweight='bold')
            ax2.set_title('Predicted Final Grade', fontsize=15, fontweight='bold', pad=20)
            ax2.set_xlim(0, 11)
            ax2.grid(axis='x', alpha=0.3, linestyle='--')
            ax2.axvline(5.0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Pass Threshold (5.0)')
            ax2.legend()
            plt.tight_layout()
            st.pyplot(fig2)
        
        # --- Scatter Plot ---
        st.subheader("🎯 Success vs Grade Prediction Landscape")
        fig3, ax3 = plt.subplots(figsize=(12, 7))
        
        scatter = ax3.scatter(
            recommendations['success_probability'] * 100,
            recommendations['expected_grade'],
            s=400,
            c=recommendations['recommendation_score'],
            cmap='viridis',
            alpha=0.8,
            edgecolors='black',
            linewidth=2
        )
        
        for i, row in recommendations.iterrows():
            ax3.annotate(
                row['course_code'],
                (row['success_probability'] * 100, row['expected_grade']),
                fontsize=11,
                ha='center',
                fontweight='bold',
                color='white'
            )
        
        ax3.set_xlabel('Success Probability (%)', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Expected Grade (0-10)', fontsize=14, fontweight='bold')
        ax3.set_title('Course Recommendations Map: Higher & Righter = Better', 
                     fontsize=16, fontweight='bold', pad=20)
        ax3.grid(alpha=0.3, linestyle='--')
        ax3.axhline(5.0, color='red', linestyle='--', linewidth=1.5, alpha=0.6, label='Pass Threshold')
        ax3.axvline(50, color='orange', linestyle='--', linewidth=1.5, alpha=0.6, label='50% Success')
        
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('Recommendation Score', fontsize=12, fontweight='bold')
        
        ax3.legend(loc='lower left', fontsize=11)
        plt.tight_layout()
        st.pyplot(fig3)
        
        # --- Difficulty Analysis ---
        st.subheader("📈 Difficulty vs Expected Performance")
        fig4, ax4 = plt.subplots(figsize=(12, 6))
        
        x = recommendations['difficulty_level']
        y1 = recommendations['expected_grade']
        y2 = recommendations['success_probability'] * 10  # Scale to 0-10 for comparison
        
        ax4.scatter(x, y1, s=200, alpha=0.7, c='#2ecc71', edgecolor='black', 
                   linewidth=1.5, label='Expected Grade', marker='o')
        ax4.scatter(x, y2, s=200, alpha=0.7, c='#3498db', edgecolor='black', 
                   linewidth=1.5, label='Success Probability (scaled)', marker='s')
        
        ax4.set_xlabel('Course Difficulty (0-10)', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Score (0-10 scale)', fontsize=14, fontweight='bold')
        ax4.set_title('How Difficulty Impacts Your Performance', fontsize=16, fontweight='bold', pad=20)
        ax4.legend(fontsize=12)
        ax4.grid(alpha=0.3, linestyle='--')
        plt.tight_layout()
        st.pyplot(fig4)

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.info(
    """
    **📖 How it works:**
    - **Logistic Regression** predicts pass/fail probability
    - **Linear Regression** predicts expected grade (0-10)
    - Courses ranked by combined score
    - **CGPA Scale:** 0-10 (Indian standard)
    - **Pass Threshold:** 5.0/10
    """
)

st.sidebar.markdown("---")
st.sidebar.caption("Built with ❤️ using Streamlit & Scikit-learn")