# CourseIQ — Intelligent Course Recommendation System

A production-ready ML system that recommends university courses by predicting
student performance through **5 department-specialized model pairs** (10 models total),
respecting prerequisite chains, and providing rich visual analytics.

---

##  Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the app (auto-generates data & trains models on first run)
streamlit run app.py
```

The app handles first-time setup automatically:
- Generates a SQLite database with 500 students, 60 courses, 15,000+ enrollments
- Trains 10 ML models (Logistic + Linear per department)
- Saves everything for fast subsequent launches

---

##  Architecture

```
course-recommendation-advanced/
│
├── app.py                          # Streamlit dashboard
├── setup_database.py               # Database schema creation
├── data_generator.py               # Synthetic data generation
├── requirements.txt
│
├── database/
│   ├── course_recommendation.db   # SQLite database (auto-created)
│   └── schema.sql                  # Schema reference
│
├── models/
│   ├── feature_engineering.py     # Department-specific features
│   ├── train_department_models.py  # Train all 10 models
│   ├── prerequisite_checker.py    # Prerequisite validation
│   ├── recommender.py             # Core recommendation engine
│   └── saved/                     # Serialized .pkl model files
│
└── utils/
    └── validators.py              # Input validation
```

---

##  Machine Learning Design

### One Model Per Department (Key Design Decision)
Rather than training a single model across all departments, the system trains
**two specialized models per department**:

| Department | Logistic (Pass/Fail) | Linear (Grade Prediction) |
|------------|---------------------|--------------------------|
| CS         | CS_logistic.pkl     | CS_linear.pkl            |
| MATH       | MATH_logistic.pkl   | MATH_linear.pkl          |
| STAT       | STAT_logistic.pkl   | STAT_linear.pkl          |
| IS         | IS_logistic.pkl     | IS_linear.pkl            |
| SE         | SE_logistic.pkl     | SE_linear.pkl            |

This means a CS course recommendation uses **only CS enrollment history** for training —
capturing the specific relationship between CS proficiency and CS course performance.

### Feature Engineering (11 features)
Each model receives department-specific features:
- **Primary**: `dept_proficiency` (the student's proficiency in the course's department)
- **Context**: `hardworking_level`, `cgpa`, `credits_completed`, `year`
- **Course**: `difficulty_level`, `avg_workload_hours`
- **Derived**:
  - `proficiency_difficulty_gap` = dept_proficiency − difficulty
  - `workload_capacity` = credits / (year + 1)
  - `hardwork_proficiency_product` = hardworking × dept_proficiency
  - `experience_factor` = year × credits / 100

### Grade Formula (Synthetic Data)
```
grade = (dept_proficiency × 0.6) + (hardworking × 0.2) + (cgpa × 0.2)
        − (difficulty × 0.3) + N(0, 0.8)
```

### Recommendation Score
```
score = (success_prob^0.6) × (expected_grade/10) × (proficiency_match^0.4) × 10
```

---

##  Data Generation

- **500 students** across 4 archetype clusters (CS-Theory, Applied-IS, Balanced, SE-Focused)
- **60 courses** across 5 departments with 3 difficulty tiers (basic/intermediate/advanced)
- **15,000+ enrollments** with prerequisite enforcement
- Student proficiencies are correlated but not identical within clusters

---

##  Prerequisite System

The `PrerequisiteChecker` class validates course chains at inference time:

```python
checker = PrerequisiteChecker()
met, warning = checker.check_prerequisites(course_id=14, completed_course_ids=[1, 2, 3])
# met = False
# warning = " Warning: CS301 (Advanced Algorithms) is a prerequisite for CS401 (Machine Learning)..."
```

**Readiness Score** [0-1]:
- 0.0 = prerequisite not completed
- 0.5 = prerequisite passed at minimum grade (5.0)
- 1.0 = prerequisite passed with full marks

---

##  Dashboard Features

1. **Sidebar input**: Enter any student profile without database writes
2. **Prerequisite warnings**: Prominent alerts for missing prerequisites
3. **Recommendations table**: Color-coded by department with all key metrics
4. **Success Probability chart**: Bar chart per recommended course
5. **Expected Grade chart**: Predicted grades visualized
6. **Scatter landscape**: Success vs Grade with bubble size = recommendation score
7. **Radar chart**: Student proficiency profile across all 5 domains
8. **Explanation panel**: "Why this recommendation?" for top 3 courses

---

##  Example: Priya Sharma

Profile: CGPA 8.2 | Hardworking 9 | CS: 9 | Math: 5 | Stat: 7 | IS: 8 | SE: 9

Expected results:
- **CS and SE courses ranked highest** (proficiency 9 in both)
- **Math courses ranked lower** (proficiency 5)
- **Advanced courses flagged** if prerequisites not met
- Radar chart shows strong CS/IS/SE quadrant, weaker Math/Stat

---

##  Manual Setup (optional, for development)

```bash
# Step 1: Create database schema
python setup_database.py

# Step 2: Generate synthetic data
python data_generator.py

# Step 3: Train models
python models/train_department_models.py

# Step 4: Launch app
streamlit run app.py
```

---

##  Dependencies

| Package       | Purpose                          |
|---------------|----------------------------------|
| streamlit     | Web dashboard                    |
| pandas        | Data manipulation                |
| numpy         | Numerical operations             |
| scikit-learn  | ML models (Logistic/Linear)      |
| plotly        | Interactive visualizations       |
| joblib        | Model serialization              |

---

*Built with ❤️ — CourseIQ uses classical ML (no neural networks) for transparency, speed, and interpretability.*
