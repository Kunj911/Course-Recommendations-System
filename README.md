# 🎓 Course Recommendation System

AI-powered course recommendation engine that predicts student performance using **department-specialized** machine learning models. Each of the 5 departments (CS, Math, Statistics, IS, SE) has its own trained Logistic Regression (pass/fail) and Linear Regression (grade prediction) model.

---

## ✨ Features

- **10 Specialized Models** – One Logistic + one Linear model per department
- **Custom Student Entry** – Enter a profile directly; no database insertion needed
- **Prerequisite Checking** – Clear warnings for missing prerequisite courses
- **Smart Scoring** – Recommendations ranked by success probability, expected grade, and proficiency match
- **Interactive Dashboard** – Streamlit UI with Plotly charts, radar profiles, and detailed course cards

---

## 📁 Project Structure

```
course-recommendation-antigravity/
├── database/
│   ├── course_recommendation.db   # Generated SQLite database
│   └── schema.sql                 # Table definitions
├── models/
│   ├── __init__.py
│   ├── feature_engineering.py     # Department-specific features
│   ├── train_department_models.py # Training script (10 models)
│   ├── prerequisite_checker.py    # Prerequisite validation
│   ├── recommender.py             # Multi-model recommendation engine
│   └── saved/                     # Persisted .pkl models & scalers
├── utils/
│   ├── __init__.py
│   └── validators.py              # Input validation
├── setup_database.py              # Schema creation
├── data_generator.py              # Synthetic data generation
├── app.py                         # Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Initialise Database & Generate Data

```bash
python setup_database.py
python data_generator.py
```

### 3. Train Models

```bash
python -m models.train_department_models
```

### 4. Launch Dashboard

```bash
streamlit run app.py
```

---

## 🧠 How It Works

1. **Data Generation** – 500 students, 60 courses, 15 000+ enrolments with realistic proficiency-driven grade outcomes.
2. **Feature Engineering** – Each department model uses the student's *department-specific* proficiency as its primary feature, along with derived features like `proficiency_difficulty_gap` and `hardwork_proficiency_product`.
3. **Training** – Logistic Regression predicts pass/fail; Linear Regression predicts grade (0–10). Both are fitted per department with StandardScaler normalisation.
4. **Recommendation** – For a custom student profile the engine queries all 60 courses, selects the right department model, checks prerequisites, and ranks by a composite score.
5. **Dashboard** – Streamlit app displays ranked recommendations, success charts, grade charts, a scatter landscape, and a radar chart of the student's proficiency profile.

---

## 📊 Grade Formula

```
grade = (proficiency × 0.6) + (hardworking × 0.2) + (cgpa × 0.2) - (difficulty × 0.3) + noise
```

- **Pass threshold**: 5.0 / 10

---

## 📋 Example

| Input Field | Value |
|---|---|
| Name | Priya Sharma |
| CGPA | 8.2 |
| Hardworking | 9 |
| CS | 9 |
| Math | 5 |
| Stat | 7 |
| IS | 8 |
| SE | 9 |

**Expected output:** CS and SE courses ranked highest; Math courses ranked lower; prerequisite warnings for advanced courses missing their chain.

---

## 🛠️ Tech Stack

- Python 3.10+
- Streamlit
- scikit-learn
- Plotly
- Pandas / NumPy
- SQLite
