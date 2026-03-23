"""
CourseIQ – FastAPI Backend
===========================
Lightweight REST API that bridges the HTML frontend with the
Python ML ensemble recommender.

Launch:  uvicorn api:app --reload --port 8000
"""

import os, sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

# Ensure project root is on the path so `models.*` imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(title="CourseIQ API", version="1.0.0")

# Allow requests from any origin (file://, localhost, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Auto-training guard (same logic as app.py)
# ---------------------------------------------------------------------------
def _ensure_models():
    saved_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "saved")
    marker = os.path.join(saved_dir, "CS_gb_classifier.pkl")
    if not os.path.exists(marker):
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "database", "course_recommendation.db")
        if not os.path.exists(db_path):
            from setup_database import setup_database
            setup_database()
            from data_generator import generate_all_data
            generate_all_data()
        from models.train_department_models import train_all_models
        train_all_models(verbose=True)

_ensure_models()

# ---------------------------------------------------------------------------
# Load recommender once at startup
# ---------------------------------------------------------------------------
from models.recommender import CourseRecommender
recommender = CourseRecommender()

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class StudentProfile(BaseModel):
    name: str = "Student"
    cgpa: float = Field(ge=0, le=10, default=7.5)
    hardworking_level: int = Field(ge=1, le=10, default=7)
    cs_proficiency: float = Field(ge=0, le=10, default=7.0)
    math_proficiency: float = Field(ge=0, le=10, default=6.0)
    stat_proficiency: float = Field(ge=0, le=10, default=6.5)
    is_proficiency: float = Field(ge=0, le=10, default=6.0)
    se_proficiency: float = Field(ge=0, le=10, default=6.5)
    credits_completed: int = Field(ge=0, le=200, default=60)
    year: int = Field(ge=1, le=4, default=2)
    top_n: int = Field(ge=3, le=20, default=10)

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {"status": "ok", "models_loaded": len(recommender._models)}

@app.post("/api/recommend")
def recommend(profile: StudentProfile):
    student = profile.model_dump()
    top_n = student.pop("top_n", 10)
    recs = recommender.recommend(student, top_n=top_n)
    return {"recommendations": recs, "count": len(recs)}

@app.post("/api/analytics")
def analytics(profile: StudentProfile):
    """Return per-department aggregated analytics for Charts."""
    student = profile.model_dump()
    student.pop("top_n", None)
    all_recs = recommender.recommend(student, top_n=100)  # get everything

    # Aggregate by department
    dept_data = {}
    for r in all_recs:
        d = r["department"]
        if d not in dept_data:
            dept_data[d] = {"success_sum": 0, "grade_sum": 0, "conf_sum": 0,
                            "count": 0, "courses": []}
        dept_data[d]["success_sum"] += r["success_probability"]
        dept_data[d]["grade_sum"] += r["expected_grade"]
        dept_data[d]["conf_sum"] += r["confidence_score"]
        dept_data[d]["count"] += 1
        dept_data[d]["courses"].append({
            "code": r["course_code"],
            "name": r["course_name"],
            "success": round(r["success_probability"] * 100, 1),
            "grade": r["expected_grade"],
            "confidence": round(r["confidence_score"] * 100, 1),
            "difficulty": r["difficulty"],
            "score": r["recommendation_score"],
        })

    departments = {}
    for d, v in dept_data.items():
        n = v["count"]
        departments[d] = {
            "avg_success": round(v["success_sum"] / n * 100, 1),
            "avg_grade": round(v["grade_sum"] / n, 2),
            "avg_confidence": round(v["conf_sum"] / n * 100, 1),
            "course_count": n,
            "courses": sorted(v["courses"], key=lambda c: c["score"], reverse=True),
        }

    # Feature importances per department
    importances = {}
    for dept in dept_data.keys():
        importances[dept] = recommender.get_feature_importances(dept, top_n=5)

    # Overall stats
    total = len(all_recs)
    overall_success = sum(r["success_probability"] for r in all_recs) / total * 100
    overall_grade = sum(r["expected_grade"] for r in all_recs) / total

    return {
        "departments": departments,
        "feature_importances": importances,
        "overall": {
            "avg_success": round(overall_success, 1),
            "avg_grade": round(overall_grade, 2),
            "total_courses": total,
        },
    }

# Serve the HTML frontend
app.mount("/", StaticFiles(directory=os.path.dirname(os.path.abspath(__file__)), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    print("\nStarting CourseIQ Full-Stack App...")
    print("Open http://localhost:8000/landing.html in your browser")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)


