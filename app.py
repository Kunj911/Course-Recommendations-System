"""
CourseIQ – Streamlit Dashboard
==============================
Clean light academic theme. All ML logic preserved.
Launch: streamlit run app.py
"""

import os
import textwrap
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CourseIQ – Smart Course Recommender",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

/* ── Reset & font ── */
html, body, [class*="css"], * {
    font-family: 'DM Sans', sans-serif !important;
}

/* ── App background ── */
.stApp, .main, .block-container {
    background-color: #F4F6FB !important;
}
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 1200px;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #4F6AF0 !important;
}
section[data-testid="stSidebar"] > div {
    background-color: #FFFFFF !important;
    padding: 1.5rem 1.2rem !important;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {
    color: #374151 !important;
    font-weight: 500 !important;
    font-size: 13px !important;
}

/* ── ALL inputs — force white with visible border ── */
input, textarea, select,
div[data-baseweb="input"],
div[data-baseweb="textarea"],
div[data-baseweb="select"],
.stTextInput > div > div,
.stNumberInput > div > div > div {
    background-color: #FFFFFF !important;
    border: 1.5px solid #CBD5E1 !important;
    border-radius: 8px !important;
    color: #1E293B !important;
}
input:focus,
div[data-baseweb="input"]:focus-within {
    border-color: #4F6AF0 !important;
    box-shadow: 0 0 0 3px rgba(79,106,240,0.12) !important;
    outline: none !important;
}
input[type="number"] {
    background-color: #FFFFFF !important;
    color: #1E293B !important;
}
/* Number input stepper buttons */
.stNumberInput button {
    background-color: #F8FAFC !important;
    color: #374151 !important;
    border: none !important;
}
/* Input text color override */
.stTextInput input,
.stNumberInput input {
    color: #1E293B !important;
    background-color: #FFFFFF !important;
}

/* ── Slider ── */
div[data-testid="stSlider"] > div {
    color: #374151 !important;
}
.stSlider [data-baseweb="slider"] [role="slider"] {
    background-color: #4F6AF0 !important;
    border-color: #4F6AF0 !important;
}
.stSlider [data-baseweb="slider"] div[class*="thumb"] {
    background-color: #4F6AF0 !important;
}
.stSlider [data-baseweb="slider"] div[class*="track"]:first-child {
    background-color: #4F6AF0 !important;
}

/* ── Metric cards ── */
div[data-testid="metric-container"] {
    background-color: #FFFFFF !important;
    border: 1px solid #4F6AF0 !important;
    border-radius: 14px !important;
    padding: 18px 22px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
}
div[data-testid="metric-container"] label {
    color: #64748B !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #0F172A !important;
    font-size: 28px !important;
    font-weight: 700 !important;
}

/* ── Headings ── */
h1, h2, h3, h4 {
    color: #0F172A !important;
    font-weight: 700 !important;
}

/* ── Plotly chart container — the correct card wrapper ── */
[data-testid="stPlotlyChart"] {
    background-color: #FFFFFF !important;
    border: 1px solid #4F6AF0 !important;
    border-radius: 16px !important;
    padding: 8px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
    overflow: hidden;
}

/* ── Expander ── */
details {
    background-color: #FFFFFF !important;
    border: 1px solid #4F6AF0 !important;
    border-radius: 12px !important;
    padding: 4px 0 !important;
}
details summary {
    color: #0F172A !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 12px 16px !important;
}
details[open] summary {
    border-bottom: 1px solid #4F6AF0 !important;
}

/* ── Primary button ── */
.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    transition: all 0.15s ease !important;
    border: none !important;
}
.stButton > button[kind="primary"],
div[data-testid="stBaseButton-primary"] button,
section[data-testid="stSidebar"] .stButton > button {
    background-color: #4F6AF0 !important;
    color: #FFFFFF !important;
    border: none !important;
    height: 46px !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    box-shadow: 0 2px 8px rgba(79,106,240,0.25) !important;
}
.stButton > button[kind="primary"]:hover {
    background-color: #3D56D6 !important;
    box-shadow: 0 4px 12px rgba(79,106,240,0.35) !important;
    transform: translateY(-1px) !important;
}
/* Secondary button — prereq dismiss */
.stButton > button[kind="secondary"] {
    background-color: #FFFBEB !important;
    color: #92400E !important;
    border: 1.5px solid #F59E0B !important;
    font-size: 13px !important;
}
.stButton > button[kind="secondary"]:hover {
    background-color: #FEF3C7 !important;
}

/* ── Alerts ── */
div[data-testid="stAlert"] {
    background-color: #FFFBEB !important;
    border: 1px solid #F59E0B !important;
    border-radius: 10px !important;
}
div[data-testid="stAlert"] p {
    color: #92400E !important;
    font-weight: 600 !important;
}

/* ── Spinner ── */
.stSpinner > div {
    border-top-color: #4F6AF0 !important;
}

/* ── Global text ── */
.stMarkdown p, .stMarkdown li {
    color: #374151 !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background-color: transparent !important;
    border-bottom: 2px solid #4F6AF0 !important;
}
.stTabs [data-baseweb="tab"] {
    color: #64748B !important;
    font-weight: 600 !important;
    font-size: 13px !important;
}
.stTabs [aria-selected="true"] {
    color: #4F6AF0 !important;
    border-bottom: 2px solid #4F6AF0 !important;
}

/* ── Divider ── */
hr {
    border-color: #4F6AF0 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Department colours ────────────────────────────────────────────────────────
DEPT_COLORS = {
    "CS":   "#4F6AF0",
    "MATH": "#7C3AED",
    "STAT": "#0EA5E9",
    "IS":   "#F59E0B",
    "SE":   "#10B981",
}
DEPT_BG = {
    "CS":   "#EEF2FF",
    "MATH": "#F5F3FF",
    "STAT": "#E0F2FE",
    "IS":   "#FFFBEB",
    "SE":   "#ECFDF5",
}
DEPT_LABEL = {
    "CS": "Computer Science", "MATH": "Mathematics",
    "STAT": "Statistics", "IS": "Info Systems", "SE": "Software Eng.",
}

# ── render_html helper ────────────────────────────────────────────────────────
def render_html(html: str, height: int = 200):
    full = textwrap.dedent(f"""
        <!DOCTYPE html><html><head><meta charset="utf-8"/>
        <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap" rel="stylesheet"/>
        <style>
          *{{box-sizing:border-box;margin:0;padding:0;}}
          body{{font-family:'DM Sans',sans-serif;background:transparent;color:#0F172A;-webkit-font-smoothing:antialiased;}}
        </style></head><body>{html}</body></html>
    """)
    components.html(full, height=height, scrolling=False)

# ── Section header helper ─────────────────────────────────────────────────────
def section_header(title: str, subtitle: str = ""):
    sub_html = f'<div style="font-size:13px;color:#64748B;margin-top:3px;">{subtitle}</div>' if subtitle else ""
    render_html(f"""
<div style="padding:4px 0 12px;">
  <div style="display:flex;align-items:center;gap:10px;">
    <div style="width:4px;height:24px;background:#4F6AF0;border-radius:2px;flex-shrink:0;"></div>
    <div style="font-size:18px;font-weight:700;color:#0F172A;">{title}</div>
  </div>
  {sub_html}
</div>""", height=52 if subtitle else 44)

# ── Auto-training guard ───────────────────────────────────────────────────────
def ensure_data_and_models():
    saved_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "saved")
    marker    = os.path.join(saved_dir, "CS_gb_classifier.pkl")
    if not os.path.exists(marker):
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "database", "course_recommendation.db")
        is_populated = False
        if os.path.exists(db_path):
            import sqlite3
            try:
                conn = sqlite3.connect(db_path)
                count = conn.execute("SELECT COUNT(*) FROM courses").fetchone()[0]
                conn.close()
                is_populated = count > 0
            except Exception:
                pass
        if not is_populated:
            from setup_database import setup_database
            setup_database()
            from data_generator import main as generate_all_data
            generate_all_data()
        with st.spinner("First launch – training ensemble models…"):
            from models.train_department_models import train_all_models
            train_all_models(verbose=True)
        st.success("Ensemble models trained successfully!")

ensure_data_and_models()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    render_html("""
<div style="padding:0 0 14px;">
  <div style="font-size:18px;font-weight:700;color:#0F172A;letter-spacing:-0.01em;">CourseIQ</div>
  <div style="font-size:12px;color:#4f6af0;margin-top:2px;font-weight:500;">Student Profile Setup</div>
  <div style="height:1px;background:#4F6AF0;margin-top:14px;"></div>
</div>""", height=62)

    st.text_input("Full Name", placeholder="e.g. Alex Kumar", key="name_input")
    name = st.session_state.get("name_input", "")

    render_html("""
<div style="font-size:11px;font-weight:700;color:#4f6af0;text-transform:uppercase;
     letter-spacing:0.08em;padding:6px 0 4px;">Academic Metrics</div>""", height=30)

    cgpa        = st.number_input("CGPA (0–10)", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
    hardworking = st.slider("Hardworking Level", 1, 10, 6,
                            help="1 = minimal effort · 10 = maximum dedication")

    render_html("""
<div style="font-size:11px;font-weight:700;color:#4f6af0;text-transform:uppercase;
     letter-spacing:0.08em;padding:10px 0 4px;">Subject Proficiencies</div>""", height=30)

    cs_prof   = st.number_input("💻 CS Proficiency",    0.0, 10.0, 5.0, 0.1)
    math_prof = st.number_input("📐 Math Proficiency",  0.0, 10.0, 5.0, 0.1)
    stat_prof = st.number_input("📊 Stats Proficiency", 0.0, 10.0, 5.0, 0.1)
    is_prof   = st.number_input("🏢 IS Proficiency",    0.0, 10.0, 5.0, 0.1)
    se_prof   = st.number_input("⚙️ SE Proficiency",    0.0, 10.0, 5.0, 0.1)

    render_html("""
<div style="font-size:11px;font-weight:700;color:#4f6af0;text-transform:uppercase;
     letter-spacing:0.08em;padding:10px 0 4px;">Settings</div>""", height=30)

    top_n = st.slider("Number of Recommendations", 3, 15, 10)

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    get_recs = st.button("Get Recommendations", use_container_width=True, type="primary")

    render_html("""
<div style="margin-top:16px;padding:12px 14px;background:#F8FAFC;border:1px solid #4F6AF0;
            border-radius:10px;">
  <div style="font-size:11px;font-weight:700;color:#64748B;text-transform:uppercase;
              letter-spacing:0.06em;margin-bottom:8px;">Ensemble Engine</div>
  <div style="font-size:12px;color:#4f6af0;line-height:1.7;">
    4 models per department<br>
    Pass/Fail: GB 50% + RF 50%<br>
    Grade: GB 60% + SVR 40%
  </div>
</div>""", height=110)

# ── Hero banner ───────────────────────────────────────────────────────────────
render_html("""
<div style="background:linear-gradient(135deg,#FFFFFF 0%,#EEF2FF 100%);
            border:1px solid #4F6AF0;border-radius:20px;padding:36px 44px;
            display:flex;justify-content:space-between;align-items:center;
            box-shadow:0 1px 4px rgba(0,0,0,0.04);">
  <div style="flex:1;min-width:0;">
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">
      <div style="width:6px;height:6px;border-radius:50%;background:#4F6AF0;"></div>
      <div style="font-size:11px;font-weight:700;letter-spacing:0.1em;color:#4F6AF0;text-transform:uppercase;">
        AI-Powered Academic Advisor
      </div>
    </div>
    <div style="font-size:36px;font-weight:700;color:#0F172A;line-height:1.15;letter-spacing:-0.02em;margin-bottom:10px;">
      CourseIQ
    </div>
    <div style="font-size:15px;color:#64748B;line-height:1.7;max-width:520px;">
      Match yourself with courses where you are statistically most likely to succeed,
      using department-specialized machine-learning ensembles.
    </div>
  </div>
  <div style="display:flex;flex-direction:column;gap:10px;margin-left:32px;flex-shrink:0;">
    <div style="background:#FFFFFF;border:1px solid #4F6AF0;border-left:3px solid #4F6AF0;
                border-radius:10px;padding:11px 20px;min-width:140px;">
      <div style="font-size:22px;font-weight:700;color:#4F6AF0;line-height:1;">20</div>
      <div style="font-size:12px;color:#4f6af0;margin-top:2px;font-weight:500;">Ensemble Models</div>
    </div>
    <div style="background:#FFFFFF;border:1px solid #4F6AF0;border-left:3px solid #4F6AF0;
                border-radius:10px;padding:11px 20px;min-width:140px;">
      <div style="font-size:22px;font-weight:700;color:#4F6AF0;line-height:1;">5</div>
      <div style="font-size:12px;color:#4f6af0;margin-top:2px;font-weight:500;">Departments</div>
    </div>
    <div style="background:#FFFFFF;border:1px solid #4F6AF0;border-left:3px solid #4F6AF0;
                border-radius:10px;padding:11px 20px;min-width:140px;">
      <div style="font-size:22px;font-weight:700;color:#4F6AF0;line-height:1;">60</div>
      <div style="font-size:12px;color:#4f6af0;margin-top:2px;font-weight:500;">Courses Available</div>
    </div>
  </div>
</div>
""", height=230)

# ── Session state ─────────────────────────────────────────────────────────────
if "overridden_prereqs" not in st.session_state:
    st.session_state.overridden_prereqs = set()

if get_recs:
    st.session_state.show_recs = True

# ── Results state ─────────────────────────────────────────────────────────────
if st.session_state.get("show_recs"):
    from utils.validators import validate_student_profile

    overridden = list(st.session_state.overridden_prereqs)
    profile = {
        "name": name, "cgpa": cgpa, "hardworking_level": hardworking,
        "cs_proficiency": cs_prof, "math_proficiency": math_prof,
        "stat_proficiency": stat_prof, "is_proficiency": is_prof,
        "se_proficiency": se_prof, "credits_completed": 60, "year": 2,
        "overridden_prereqs": overridden,
    }
    errors = validate_student_profile(profile)
    if errors:
        for e in errors:
            st.error(e)
        st.stop()

    @st.cache_resource
    def load_recommender():
        from models.recommender import CourseRecommender
        return CourseRecommender()

    with st.spinner("Loading ensemble models…"):
        recommender = load_recommender()
    with st.spinner("Analyzing your profile across 20 specialized models…"):
        recs = recommender.recommend(profile, top_n=top_n)

    if not recs:
        st.warning("No recommendations generated. Please check that the database is populated.")
        st.stop()

    # ── Profile snapshot card ─────────────────────────────────────────────────
    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

    pills_html = ""
    pill_data = [
        ("CGPA",      f"{cgpa:.1f}",      "#F8FAFC", "#4F6AF0", "#0F172A"),
        ("Hardwork",  f"{hardworking}/10", "#F8FAFC", "#4F6AF0", "#0F172A"),
        ("CS",        f"{cs_prof:.1f}",    DEPT_BG["CS"],   DEPT_COLORS["CS"]+"33", DEPT_COLORS["CS"]),
        ("Math",      f"{math_prof:.1f}",  DEPT_BG["MATH"], DEPT_COLORS["MATH"]+"33", DEPT_COLORS["MATH"]),
        ("Stat",      f"{stat_prof:.1f}",  DEPT_BG["STAT"], DEPT_COLORS["STAT"]+"33", DEPT_COLORS["STAT"]),
        ("IS",        f"{is_prof:.1f}",    DEPT_BG["IS"],   DEPT_COLORS["IS"]+"33",   DEPT_COLORS["IS"]),
        ("SE",        f"{se_prof:.1f}",    DEPT_BG["SE"],   DEPT_COLORS["SE"]+"33",   DEPT_COLORS["SE"]),
    ]
    for lbl, val, pbg, brd, vcol in pill_data:
        pills_html += f"""
<div style="background:{pbg};border:1.5px solid {brd};border-radius:10px;
            padding:10px 18px;text-align:center;min-width:72px;">
  <div style="font-size:10px;color:#4f6af0;text-transform:uppercase;letter-spacing:0.06em;
              font-weight:700;margin-bottom:4px;">{lbl}</div>
  <div style="font-size:18px;color:{vcol};font-weight:700;font-variant-numeric:tabular-nums;">{val}</div>
</div>"""

    render_html(f"""
<div style="background:#FFFFFF;border:1px solid #4F6AF0;border-radius:16px;padding:22px 28px;
            box-shadow:0 1px 3px rgba(0,0,0,0.04);display:flex;
            justify-content:space-between;align-items:center;gap:20px;">
  <div style="flex-shrink:0;">
    <div style="border-left:3px solid #4F6AF0;padding-left:12px;">
      <div style="font-size:22px;font-weight:700;color:#0F172A;line-height:1.2;">
        {name or 'Anonymous'}
      </div>
      <div style="font-size:12px;color:#4f6af0;margin-top:3px;font-weight:500;">Academic Profile</div>
    </div>
  </div>
  <div style="display:flex;gap:8px;flex-wrap:wrap;justify-content:flex-end;">
    {pills_html}
  </div>
</div>""", height=115)

    # ── Top Recommendations ───────────────────────────────────────────────────
    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
    section_header("Top Recommendations", f"Ranked by ensemble score · showing {len(recs)} courses")

    for i, r in enumerate(recs, 1):
        dept       = r["department"]
        color      = DEPT_COLORS.get(dept, "#4F6AF0")
        bg         = DEPT_BG.get(dept, "#EEF2FF")
        readiness  = r.get("readiness_label", "")
        read_col   = {"High": "#10B981", "Medium": "#F59E0B", "Low": "#EF4444"}.get(readiness, "#64748B")
        prereq_ok  = r["prerequisite_met"]
        sp         = r["success_probability"] * 100
        conf_raw   = r.get("confidence_label", "")
        conf_clean = conf_raw.replace("🟢 ","").replace("🟡 ","").replace("🔴 ","")

        if sp >= 75:   succ_col = "#10B981"
        elif sp >= 55: succ_col = "#F59E0B"
        else:          succ_col = "#EF4444"

        if prereq_ok:
            pr_bg, pr_border, pr_col = "#ECFDF5", "#34D399", "#065F46"
            pr_text = "✅  All prerequisites satisfied"
        else:
            pr_bg, pr_border, pr_col = "#FFFBEB", "#FCD34D", "#92400E"
            pr_text = f"⚠️  Missing prerequisite: {r['prerequisite_code']} — {r['prerequisite_name']}"

        card_html = f"""
<style>
.c{i}{{background:#FFFFFF;border:1.5px solid #4F6AF0;border-radius:16px;
       padding:20px 24px;box-shadow:0 1px 4px rgba(0,0,0,0.04);
       font-family:'DM Sans',sans-serif;transition:border-color .18s,box-shadow .18s;}}
.c{i}:hover{{border-color:{color};box-shadow:0 4px 20px rgba(79,106,240,0.10);}}
.top{i}{{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:16px;}}
.badge{i}{{display:inline-flex;align-items:center;padding:3px 10px;border-radius:20px;
           font-size:11px;font-weight:700;background:{bg};color:{color};
           border:1px solid {color}44;margin-left:10px;}}
.strip{i}{{display:flex;background:#F8FAFC;border:1px solid #4F6AF0;
           border-radius:10px;overflow:hidden;}}
.cell{i}{{flex:1;padding:11px 8px;text-align:center;border-right:1px solid #4F6AF0;}}
.cell{i}:last-child{{border-right:none;}}
.cl{i}{{font-size:9px;text-transform:uppercase;letter-spacing:.07em;color:#4f6af0;font-weight:700;margin-bottom:4px;}}
.cv{i}{{font-size:14px;font-weight:700;color:#0F172A;font-variant-numeric:tabular-nums;}}
.pbar{i}{{display:flex;align-items:center;gap:8px;padding:9px 14px;border-radius:8px;
          margin-top:13px;background:{pr_bg};border-left:3px solid {pr_border};}}
</style>
<div class="c{i}">
  <div class="top{i}">
    <div style="display:flex;align-items:center;gap:14px;">
      <div style="font-size:26px;font-weight:800;color:{color};line-height:1;min-width:32px;">{i}</div>
      <div>
        <div style="display:flex;align-items:center;flex-wrap:wrap;gap:0;">
          <span style="font-size:18px;font-weight:700;color:#0F172A;">{r['course_code']}</span>
          <span class="badge{i}">{dept}</span>
        </div>
        <div style="font-size:13px;color:#64748B;margin-top:3px;font-weight:500;">{r['course_name']}</div>
      </div>
    </div>
    <div style="text-align:right;flex-shrink:0;margin-left:16px;">
      <div style="font-size:30px;font-weight:800;color:{succ_col};line-height:1;
                  font-variant-numeric:tabular-nums;">{sp:.1f}%</div>
      <div style="font-size:10px;text-transform:uppercase;letter-spacing:.07em;
                  color:#4f6af0;font-weight:700;margin-top:3px;">Success Rate</div>
    </div>
  </div>
  <div class="strip{i}">
    <div class="cell{i}"><div class="cl{i}">Grade</div>
      <div class="cv{i}">{r['expected_grade']}/10</div></div>
    <div class="cell{i}"><div class="cl{i}">Difficulty</div>
      <div class="cv{i}">{r['difficulty']}/10</div></div>
    <div class="cell{i}"><div class="cl{i}">Score</div>
      <div class="cv{i}">{r['recommendation_score']:.2f}</div></div>
    <div class="cell{i}"><div class="cl{i}">Readiness</div>
      <div class="cv{i}" style="color:{read_col};">{readiness}</div></div>
    <div class="cell{i}"><div class="cl{i}">Confidence</div>
      <div class="cv{i}">{conf_clean}</div></div>
  </div>
  <div class="pbar{i}">
    <span style="font-size:13px;font-weight:600;color:{pr_col};">{pr_text}</span>
  </div>
</div>"""
        render_html(card_html, height=232)

        if not prereq_ok:
            if st.button(
                f"I have completed {r['prerequisite_code']} ({r['prerequisite_name']}) — dismiss",
                key=f"override_{r['course_code']}",
                type="secondary",
            ):
                st.session_state.overridden_prereqs.add(r["prerequisite_code"])
                st.rerun()

        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

    # ── Reasoning expander ────────────────────────────────────────────────────
    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    with st.expander("Reasoning and Feature Importances"):
        for i, r in enumerate(recs[:5], 1):
            dept     = r["department"]
            prof_map = {"CS": cs_prof, "MATH": math_prof, "STAT": stat_prof,
                        "IS": is_prof, "SE": se_prof}
            prof_val = prof_map.get(dept, cs_prof)
            gap      = prof_val - r["difficulty"]
            icon     = "🟢" if gap >= 2 else ("🟡" if gap >= 0 else "🔴")
            st.markdown(
                f"**{i}. {r['course_code']} – {r['course_name']}**  \n"
                f"- *Model:* **{r['model_used']}**  \n"
                f"- {icon} {dept} proficiency ({prof_val}) vs difficulty ({r['difficulty']:.1f})"
                f": gap = **{gap:+.1f}**  \n"
                f"- Success: **{r['success_probability']*100:.1f}%** | Grade: **{r['expected_grade']}/10**  \n"
                f"- Confidence: **{r.get('confidence_label','')}** (score: {r.get('confidence_score',0):.2f})  \n"
                f"- {'✅ Prerequisites met' if r['prerequisite_met'] else '⚠️ '+r['prerequisite_warning']}"
            )
            feat_imp = recommender.get_feature_importances(dept, top_n=5)
            if feat_imp:
                fi_df = pd.DataFrame(feat_imp)
                fi_df["feature"] = fi_df["feature"].str.replace("_", " ").str.title()
                fig_fi = px.bar(
                    fi_df, x="importance", y="feature", orientation="h",
                    title=f"Feature Importances – {dept} model",
                    labels={"importance": "Importance", "feature": ""},
                    color="importance",
                    color_continuous_scale=["#EEF2FF", "#4F6AF0"],
                    text=fi_df["importance"].apply(lambda v: f"{v:.3f}"),
                )
                fig_fi.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="DM Sans", color="#374151", size=12),
                    height=220, margin=dict(l=10, r=10, t=40, b=10),
                    showlegend=False, coloraxis_showscale=False,
                    yaxis=dict(autorange="reversed"),
                )
                fig_fi.update_traces(textposition="outside", marker_line_width=0)
                st.plotly_chart(fig_fi, use_container_width=True, key=f"fi_{i}_{dept}")
            st.markdown("---")

    # ── Analytics ─────────────────────────────────────────────────────────────
    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
    section_header("Analytics Overview", "Predicted outcomes across your recommended courses")

    rec_df = pd.DataFrame(recs)
    rec_df["label"]       = rec_df["course_code"] + " – " + rec_df["course_name"]
    rec_df["success_pct"] = rec_df["success_probability"] * 100

    # Shorten labels for chart readability
    rec_df["short_label"] = rec_df["course_code"]

    CHART_LAYOUT = dict(
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(family="DM Sans", color="#374151", size=12),
        margin=dict(t=48, b=60, l=10, r=10),
        height=380,
        xaxis=dict(gridcolor="#F1F5F9", linecolor="#4F6AF0",
                   tickfont=dict(size=11), tickangle=-35),
        yaxis=dict(gridcolor="#F1F5F9", linecolor="#4F6AF0"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    )

    col_a, col_b = st.columns(2)
    col_c, col_d = st.columns(2)

    with col_a:
        fig1 = px.bar(
            rec_df, x="short_label", y="success_pct",
            color="department", color_discrete_map=DEPT_COLORS,
            title="Success Probability by Course",
            labels={"success_pct": "Success (%)", "short_label": ""},
            text=rec_df["success_pct"].apply(lambda v: f"{v:.0f}%"),
            custom_data=["label"],
        )
        fig1.update_layout(**CHART_LAYOUT)
        fig1.update_layout(margin=dict(t=48, b=80, l=10, r=10))
        fig1.update_traces(textposition="outside", marker_line_width=0,
                           hovertemplate="%{customdata[0]}<br>%{y:.1f}%<extra></extra>")
        st.plotly_chart(fig1, use_container_width=True)

    with col_b:
        fig2 = px.bar(
            rec_df, x="short_label", y="expected_grade",
            color="department", color_discrete_map=DEPT_COLORS,
            title="Expected Grade by Course",
            labels={"expected_grade": "Grade (0–10)", "short_label": ""},
            text=rec_df["expected_grade"].apply(lambda v: f"{v:.1f}"),
            custom_data=["label"],
        )
        fig2.update_layout(**CHART_LAYOUT)
        fig2.update_layout(margin=dict(t=48, b=80, l=10, r=10))
        fig2.update_traces(textposition="outside", marker_line_width=0,
                           hovertemplate="%{customdata[0]}<br>Grade: %{y:.1f}<extra></extra>")
        st.plotly_chart(fig2, use_container_width=True)

    with col_c:
        fig3 = px.scatter(
            rec_df, x="success_pct", y="expected_grade",
            color="department", color_discrete_map=DEPT_COLORS,
            size="recommendation_score", hover_name="label",
            title="Success vs Grade Landscape",
            labels={"success_pct": "Success Probability (%)", "expected_grade": "Expected Grade"},
        )
        fig3.update_layout(**CHART_LAYOUT)
        fig3.update_layout(margin=dict(t=48, b=40, l=10, r=10))
        st.plotly_chart(fig3, use_container_width=True)

    with col_d:
        cats = ["CS", "Math", "Statistics", "IS", "SE"]
        vals = [cs_prof, math_prof, stat_prof, is_prof, se_prof]
        fig4 = go.Figure()
        fig4.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=cats + [cats[0]],
            fill="toself",
            fillcolor="rgba(79,106,240,0.13)",
            line=dict(color="#4F6AF0", width=2.5),
            name="Proficiency",
        ))
        fig4.update_layout(
            polar=dict(
                bgcolor="#FAFCFF",
                radialaxis=dict(visible=True, range=[0, 10],
                                gridcolor="#4F6AF0", tickfont=dict(size=10, color="#4f6af0")),
                angularaxis=dict(gridcolor="#4F6AF0",
                                 tickfont=dict(size=12, color="#374151")),
            ),
            plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
            font=dict(family="DM Sans", color="#374151"),
            title=dict(text="Proficiency Radar", font=dict(size=14, color="#0F172A")),
            showlegend=False, height=380,
            margin=dict(t=60, b=30, l=40, r=40),
        )
        st.plotly_chart(fig4, use_container_width=True)

# ── Landing state ─────────────────────────────────────────────────────────────
else:
    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

    render_html("""
<div style="text-align:center;padding:36px 24px 28px;">
  <div style="display:inline-flex;align-items:center;justify-content:center;
              width:56px;height:56px;background:#EEF2FF;border-radius:16px;margin-bottom:16px;">
    <svg width="28" height="28" viewBox="0 0 24 24" fill="none"
         xmlns="http://www.w3.org/2000/svg">
      <path d="M12 2L2 7l10 5 10-5-10-5z" stroke="#4F6AF0" stroke-width="2"
            stroke-linejoin="round"/>
      <path d="M2 17l10 5 10-5M2 12l10 5 10-5" stroke="#4F6AF0" stroke-width="2"
            stroke-linejoin="round"/>
    </svg>
  </div>
  <div style="font-size:24px;font-weight:700;color:#0F172A;margin-bottom:10px;
              letter-spacing:-0.01em;">
    Get Your Course Recommendations
  </div>
  <div style="font-size:15px;color:#64748B;max-width:460px;margin:0 auto;line-height:1.7;">
    Fill in your academic profile in the sidebar and click
    <strong style="color:#4F6AF0;">Get Recommendations</strong>
    to see AI-powered course matches.
  </div>
</div>""", height=210)

    c1, c2, c3 = st.columns(3)
    c1.metric("Ensemble Models", "20", "4 per department")
    c2.metric("Departments", "5", "CS · Math · Stat · IS · SE")
    c3.metric("Available Courses", "60", "Across all levels")

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

    render_html("""
<style>
*{box-sizing:border-box;margin:0;padding:0;font-family:'DM Sans',sans-serif;}
.wrap{background:#FFFFFF;border:1px solid #4F6AF0;border-radius:20px;
      padding:32px 36px;box-shadow:0 1px 4px rgba(0,0,0,0.04);}
.title{font-size:20px;font-weight:700;color:#0F172A;margin-bottom:6px;}
.sub{font-size:14px;color:#64748B;line-height:1.7;margin-bottom:28px;}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:18px;}
.step{background:#F8FAFC;border:1px solid #4F6AF0;border-radius:14px;padding:22px;}
.num{display:inline-flex;width:34px;height:34px;align-items:center;justify-content:center;
     background:#EEF2FF;color:#4F6AF0;border-radius:8px;font-weight:700;
     font-size:15px;margin-bottom:13px;box-shadow:0 0 0 4px #EEF2FF;}
.sh{font-size:15px;font-weight:700;color:#4F6AF0;margin-bottom:7px;}
.sp{font-size:13px;color:#4F6AF0;line-height:1.65;}
</style>
<div class="wrap">
  <div class="title">How CourseIQ Works</div>
  <div class="sub">
    CourseIQ uses a multi-model ensemble intelligence engine to forecast your academic
    outcomes and match you to courses where you are most likely to thrive.
  </div>
  <div class="grid">
    <div class="step">
      <div class="num">1</div>
      <div class="sh">Department-Specific Models</div>
      <div class="sp">Every department is evaluated by 4 specialized ML models trained
        exclusively on that department's historical student data.</div>
    </div>
    <div class="step">
      <div class="num">2</div>
      <div class="sh">Algorithmic Voting Consensus</div>
      <div class="sp">Gradient Boosting and Random Forest vote on pass or fail.
        Gradient Boosting and SVR predict your expected grade via weighted consensus.</div>
    </div>
    <div class="step">
      <div class="num">3</div>
      <div class="sh">Confidence and Prerequisite Guards</div>
      <div class="sp">A confidence score reflects model agreement. Prerequisite
        checks ensure you are never recommended a course you are not eligible for.</div>
    </div>
  </div>
</div>""", height=370)