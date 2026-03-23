# 🎓 Course Recommendation System

AI-powered course recommendation engine that predicts student performance using **department-specialized** machine learning models. Each of the 5 departments (CS, Math, Statistics, IS, SE) has its own trained ensemble of models.

---

## ✨ Features

- **10+ Specialized Models** – Ensemble models per department
- **Custom Student Entry** – Enter a profile directly; no database insertion needed
- **Prerequisite Checking** – Clear warnings for missing prerequisite courses
- **Smart Scoring** – Recommendations ranked by success probability, expected grade, and proficiency match
- **Modern Web Dashboard** – Fast, responsive frontend built with HTML/JS/CSS, powered by a FastAPI backend.
- **Advanced Real-Time Analytics** – Visualizations, feature importances, and department-level insights.

---

## 📁 Project Structure

```text
course-recommendation-antigravity/
├── database/
│   ├── course_recommendation.db   # Generated SQLite database
│   └── schema.sql                 # Table definitions
├── models/
│   ├── __init__.py
│   ├── ensemble_config.py         # Advanced model configurations
│   ├── feature_engineering.py     # Department-specific features
│   ├── train_department_models.py # Training script for all models
│   ├── prerequisite_checker.py    # Prerequisite validation
│   ├── recommender.py             # Multi-model recommendation engine
│   └── saved/                     # Persisted .pkl models & scalers
├── utils/
│   ├── validators.py              # Input validation
├── api.py                         # FastAPI backend & static file server
├── app.py                         # Legacy Streamlit dashboard
├── landing.html                   # New Web UI - Landing Page
├── index.html                     # New Web UI - Main Dashboard
├── analytics.html                 # New Web UI - Analytics Page
├── profile.html                   # New Web UI - User Profile
├── setup_database.py              # Schema creation
├── data_generator.py              # Synthetic data generation
└── requirements.txt
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

### 3. Train Models (Optional, API does this automatically if missing)

```bash
python -m models.train_department_models
```

### 4. Launch Full-Stack Application

Run the FastAPI server which also serves the frontend:
```bash
python api.py
```
Or using uvicorn directly:
```bash
uvicorn api:app --reload --port 8000
```

Then open [http://localhost:8000/landing.html](http://localhost:8000/landing.html) in your browser.

---

## 🧠 How It Works

1. **Data Generation** – Generates realistic synthetic student profiles, courses, and enrolment histories with proficiency-driven grade outcomes.
2. **Feature Engineering** – Extracts department-specific proficiencies, `proficiency_difficulty_gap`, and other derived metrics.
3. **Training** – Trains an ensemble of models (Gradient Boosting, Random Forest, SVR, etc.) per department to predict pass/fail probability and expected grades.
4. **Recommendation** – A FastAPI backend queries all available courses, checks prerequisites, and ranks them by a composite score.
5. **Dashboard** – A sleek HTML/JS frontend communicates via REST API to display ranked recommendations, expected grades, feature importances, and a radar chart of the student's profile.

---

## 📊 Grade Formula

```text
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

- **Backend:** Python 3.10+, FastAPI, Uvicorn
- **Frontend:** HTML5, CSS3, Vanilla JavaScript, Chart.js
- **Machine Learning:** scikit-learn, Pandas, NumPy
- **Database:** SQLite
