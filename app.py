"""
CourseIQ – Streamlit Dashboard (Light Theme)
=============================================
Premium, interactive UI for the multi-model ensemble course recommender.

Launch:  streamlit run app.py
"""

import os
import textwrap
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CourseIQ – Smart Course Recommender",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Light Theme CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Main background ── */
.stApp {
    background: #f4f6fb;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e5e7eb;
}
section[data-testid="stSidebar"] label {
    color: #374151 !important;
    font-weight: 500;
}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3,
section[data-testid="stSidebar"] .stMarkdown h4 {
    color: #111827;
}

/* ── Native Streamlit metric ── */
div[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 16px 20px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
div[data-testid="metric-container"] label {
    color: #6b7280 !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #111827 !important;
    font-size: 28px !important;
    font-weight: 800 !important;
}

/* ── Section headings ── */
h2, h3 {
    color: #111827 !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #ffffff !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 12px !important;
    color: #111827 !important;
    font-weight: 600 !important;
}

/* ── Hide Streamlit branding ── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Department colour mapping
# ---------------------------------------------------------------------------
DEPT_COLORS = {
    "CS":   "#3b82f6",
    "MATH": "#8b5cf6",
    "STAT": "#0ea5e9",
    "IS":   "#f59e0b",
    "SE":   "#10b981",
}

DEPT_BG = {
    "CS":   "#dbeafe",
    "MATH": "#ede9fe",
    "STAT": "#e0f2fe",
    "IS":   "#fef3c7",
    "SE":   "#d1fae5",
}

# ---------------------------------------------------------------------------
# Helper – render arbitrary HTML safely via iframe (avoids Markdown parser)
# ---------------------------------------------------------------------------
def render_html(html: str, height: int = 200):
    """Render raw HTML in an iframe to bypass Streamlit's Markdown engine."""
    full = textwrap.dedent(f"""
        <!DOCTYPE html>
        <html>
        <head>
          <meta charset="utf-8"/>
          <style>
            * {{ box-sizing: border-box; margin: 0; padding: 0; }}
            body {{ font-family: 'Inter', sans-serif; background: transparent; color: #111827; }}
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
          </style>
        </head>
        <body>{html}</body>
        </html>
    """)
    components.html(full, height=height, scrolling=False)

# ---------------------------------------------------------------------------
# Auto-training guard
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎓 Student Profile")
    st.markdown("---")
    name = st.text_input("Full Name", placeholder="e.g. Alex Kumar")

    st.markdown("#### Academic Metrics")
    cgpa        = st.number_input("CGPA (0-10)", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
    hardworking = st.slider("Hardworking Level", 1, 10, 6)

    st.markdown("#### Proficiency Scores")
    cs_prof   = st.number_input("CS Proficiency",   0.0, 10.0, 5.0, 0.1)
    math_prof = st.number_input("Math Proficiency", 0.0, 10.0, 5.0, 0.1)
    stat_prof = st.number_input("Stats Proficiency",0.0, 10.0, 5.0, 0.1)
    is_prof   = st.number_input("IS Proficiency",   0.0, 10.0, 5.0, 0.1)
    se_prof   = st.number_input("SE Proficiency",   0.0, 10.0, 5.0, 0.1)

    st.markdown("#### Settings")
    top_n    = st.slider("Number of Recommendations", 3, 15, 10)
    st.markdown("---")
    get_recs = st.button("Get Recommendations", use_container_width=True, type="primary")
    st.markdown("---")
    st.caption("Ensemble ML Engine\n\nPer-department: 4 models, weighted voting\n- Pass/Fail: GB(50%) + RF(50%)\n- Grade: GB(60%) + SVR(40%)")

# ---------------------------------------------------------------------------
# Hero banner
# ---------------------------------------------------------------------------
render_html("""
<div style="background:linear-gradient(135deg,#eef2ff 0%,#faf5ff 50%,#ecfdf5 100%);
            border:1px solid #e5e7eb;border-radius:20px;padding:48px 48px 44px;
            box-shadow:0 4px 20px rgba(0,0,0,0.04);margin-bottom:4px;">
  <div style="font-size:13px;font-weight:700;letter-spacing:.08em;color:#6366f1;
              text-transform:uppercase;margin-bottom:12px;">AI-Powered Academic Advisor</div>
  <div style="font-size:40px;font-weight:800;color:#111827;line-height:1.15;margin-bottom:14px;">
    CourseIQ <span style="color:#6366f1;">Ensemble</span> Suite
  </div>
  <div style="font-size:16px;color:#4b5563;line-height:1.7;max-width:620px;">
    Match yourself with courses where you're statistically most likely to succeed, using
    department-specialized machine-learning ensembles trained on real student outcomes.
  </div>
</div>
""", height=210)

# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------
if get_recs:
    from utils.validators import validate_student_profile

    profile = {
        "name": name, "cgpa": cgpa, "hardworking_level": hardworking,
        "cs_proficiency": cs_prof, "math_proficiency": math_prof,
        "stat_proficiency": stat_prof, "is_proficiency": is_prof,
        "se_proficiency": se_prof, "credits_completed": 60, "year": 2,
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

    # ── Profile Overview ──────────────────────────────────────────────
    st.markdown("## 👤 Your Academic Snapshot")
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Name",         name or "Anonymous")
    c2.metric("CGPA",         f"{cgpa}/10")
    c3.metric("Work Ethic",   f"{hardworking}/10")
    c4.metric("CS",   f"{cs_prof}/10")
    c5.metric("Math", f"{math_prof}/10")
    c6.metric("IS",   f"{is_prof}/10")
    c7.metric("SE",   f"{se_prof}/10")

    # Prereq warnings
    bad = [r for r in recs if not r["prerequisite_met"]]
    if bad:
        st.markdown("### ⚠️ Prerequisite Alerts")
        for w in bad:
            st.warning(w["prerequisite_warning"])

    st.markdown("---")

    # ── Top Recommendations (cards via iframe) ────────────────────────
    st.markdown("## 🏆 Top Recommendations")
    st.caption(f"Ranked by ensemble recommendation score · showing {len(recs)} courses")

    for i, r in enumerate(recs, 1):
        dept      = r["department"]
        color     = DEPT_COLORS.get(dept, "#6366f1")
        bg        = DEPT_BG.get(dept, "#e0e7ff")
        readiness = r["readiness_label"]
        read_col  = {"High": "#16a34a", "Medium": "#d97706", "Low": "#dc2626"}.get(readiness, "#6b7280")
        prereq_ok = r["prerequisite_met"]
        prereq_text = "✅ All prerequisites met" if prereq_ok else f"⚠️ Missing: {r['prerequisite_code']} — {r['prerequisite_name']}"
        prereq_style = ("background:#f0fdf4;border-left:4px solid #22c55e;color:#166534;"
                        if prereq_ok else
                        "background:#fef2f2;border-left:4px solid #ef4444;color:#991b1b;")

        card_html = f"""
<style>
.card{{background:#ffffff;border:1.5px solid #e5e7eb;border-radius:16px;padding:24px 28px;
      box-shadow:0 2px 12px rgba(0,0,0,0.04);font-family:'Inter',sans-serif;}}
.card:hover{{border-color:{color};box-shadow:0 8px 24px rgba(0,0,0,0.09);}}
.row{{display:flex;justify-content:space-between;align-items:flex-start;}}
.badge{{display:inline-block;padding:5px 14px;border-radius:30px;font-size:12px;
        font-weight:700;background:{bg};color:{color};border:1px solid {color}33;}}
.metrics-row{{display:flex;gap:0;flex-wrap:nowrap;margin-top:18px;
             background:#f9fafb;border-radius:12px;border:1px solid #f3f4f6;overflow:hidden;}}
.metric-pill{{flex:1;padding:14px 10px;text-align:center;border-right:1px solid #f3f4f6;}}
.metric-pill:last-child{{border-right:none;}}
.ml{{font-size:10px;text-transform:uppercase;letter-spacing:.06em;color:#9ca3af;font-weight:700;margin-bottom:5px;}}
.mv{{font-size:16px;font-weight:800;color:#111827;}}
.prereq{{padding:10px 16px;border-radius:0 8px 8px 0;font-size:13px;font-weight:600;
         margin-top:16px;{prereq_style}}}
.model-note{{font-size:11px;color:#9ca3af;margin-top:12px;font-weight:500;}}
</style>
<div class="card">
  <div class="row">
    <div>
      <div style="font-size:22px;font-weight:800;color:#111827;display:flex;align-items:center;gap:12px;">
        #{i} {r['course_code']} <span class="badge">{dept}</span>
      </div>
      <div style="font-size:15px;color:#6b7280;margin-top:6px;font-weight:500;">{r['course_name']}</div>
    </div>
    <div style="text-align:right;">
      <div style="font-size:34px;font-weight:900;color:{color};line-height:1;">{r['success_probability']*100:.1f}%</div>
      <div style="font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:#9ca3af;font-weight:700;margin-top:4px;">Success Rate</div>
    </div>
  </div>
  <div class="metrics-row">
    <div class="metric-pill"><div class="ml">Grade</div><div class="mv">⭐ {r['expected_grade']}/10</div></div>
    <div class="metric-pill"><div class="ml">Difficulty</div><div class="mv">⚡ {r['difficulty']}/10</div></div>
    <div class="metric-pill"><div class="ml">Score</div><div class="mv">🏅 {r['recommendation_score']:.2f}</div></div>
    <div class="metric-pill"><div class="ml">Readiness</div><div class="mv" style="color:{read_col};">{readiness}</div></div>
    <div class="metric-pill"><div class="ml">Confidence</div><div class="mv">{r['confidence_label']}</div></div>
  </div>
  <div class="prereq">{prereq_text}</div>
  <div class="model-note">🤖 Predicted by {r['model_used']}</div>
</div>
"""
        render_html(card_html, height=230)

    # ── Why These Recommendations ─────────────────────────────────────
    st.markdown("---")
    with st.expander("💡 Why These Recommendations? (Reasoning + Feature Importances)", expanded=False):
        for i, r in enumerate(recs[:5], 1):
            dept = r["department"]
            prof_map = {"CS": cs_prof, "MATH": math_prof, "STAT": stat_prof, "IS": is_prof, "SE": se_prof}
            prof_val = prof_map.get(dept, cs_prof)
            gap  = prof_val - r["difficulty"]
            icon = "🟢" if gap >= 2 else ("🟡" if gap >= 0 else "🔴")
            st.markdown(
                f"**{i}. {r['course_code']} – {r['course_name']}**  \n"
                f"- 🤖 *Model:* **{r['model_used']}**  \n"
                f"- {icon} {dept} proficiency ({prof_val}) vs difficulty ({r['difficulty']:.1f}): gap = **{gap:+.1f}**  \n"
                f"- 📊 Success: **{r['success_probability']*100:.1f}%** | Grade: **{r['expected_grade']}/10**  \n"
                f"- 🎯 Confidence: **{r['confidence_label']}** (score: {r['confidence_score']:.2f})  \n"
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
                    color_continuous_scale=["#bfdbfe", "#3b82f6"],
                    text=fi_df["importance"].apply(lambda v: f"{v:.3f}"),
                )
                fig_fi.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter", color="#374151", size=12),
                    height=220, margin=dict(l=10, r=10, t=40, b=10),
                    showlegend=False, coloraxis_showscale=False,
                    yaxis=dict(autorange="reversed"),
                )
                fig_fi.update_traces(textposition="outside", marker_line_width=0)
                st.plotly_chart(fig_fi, use_container_width=True, key=f"fi_{i}_{dept}")
            st.markdown("---")

    # ── Analytics Quadrants (2 × 2) ───────────────────────────────────
    st.markdown("## 📊 Analytics & Visualisations")

    rec_df = pd.DataFrame(recs)
    rec_df["label"]       = rec_df["course_code"] + " – " + rec_df["course_name"]
    rec_df["success_pct"] = rec_df["success_probability"] * 100

    LIGHT_LAYOUT = dict(
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        font=dict(family="Inter", color="#9ca3af"),
        margin=dict(t=55, b=100, l=10, r=10),
        height=420,
    )

    col_a, col_b = st.columns(2)
    col_c, col_d = st.columns(2)

    with col_a:
        fig1 = px.bar(
            rec_df, x="label", y="success_pct",
            color="department", color_discrete_map=DEPT_COLORS,
            title="Success Probability by Course",
            labels={"success_pct": "Success (%)", "label": ""},
            text=rec_df["success_pct"].apply(lambda v: f"{v:.0f}%"),
        )
        fig1.update_layout(**LIGHT_LAYOUT, xaxis_tickangle=-40)
        fig1.update_traces(textposition="outside", marker_line_width=0)
        st.plotly_chart(fig1, use_container_width=True)

    with col_b:
        fig2 = px.bar(
            rec_df, x="label", y="expected_grade",
            color="department", color_discrete_map=DEPT_COLORS,
            title="Expected Grade by Course",
            labels={"expected_grade": "Grade (0–10)", "label": ""},
            text=rec_df["expected_grade"].apply(lambda v: f"{v:.1f}"),
        )
        fig2.update_layout(**LIGHT_LAYOUT, xaxis_tickangle=-40)
        fig2.update_traces(textposition="outside", marker_line_width=0)
        st.plotly_chart(fig2, use_container_width=True)

    with col_c:
        fig3 = px.scatter(
            rec_df, x="success_pct", y="expected_grade",
            color="department", color_discrete_map=DEPT_COLORS,
            size="recommendation_score", hover_name="label",
            title="Success vs Grade Landscape",
            labels={"success_pct": "Success (%)", "expected_grade": "Expected Grade"},
        )
        fig3.update_layout(**{**LIGHT_LAYOUT, "margin": dict(t=55, b=20, l=10, r=10)})
        st.plotly_chart(fig3, use_container_width=True)

    with col_d:
        cats   = ["CS", "Math", "Statistics", "IS", "SE"]
        vals   = [cs_prof, math_prof, stat_prof, is_prof, se_prof]
        fig4   = go.Figure()
        fig4.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=cats + [cats[0]],
            fill="toself",
            fillcolor="rgba(99,102,241,0.18)",
            line=dict(color="#050505", width=3),
            name="Proficiency",
        ))
        fig4.update_layout(
            polar=dict(
                bgcolor="#ffffff",
                radialaxis=dict(visible=True, range=[0, 10], gridcolor="#050505"),
                angularaxis=dict(gridcolor="#050505"),
            ),
            plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
            font=dict(family="Inter", color="#050505"),
            title="Proficiency Radar", showlegend=False, height=420,
        )
        st.plotly_chart(fig4, use_container_width=True)

# ── Landing state ────────────────────────────────────────────────────
else:
    st.markdown("### 👈 Enter your profile in the sidebar to get AI-powered course recommendations")
    st.markdown("")

    # Stats row
    c1, c2, c3 = st.columns(3)
    c1.metric("Ensemble Models", "20")
    c2.metric("Departments", "5")
    c3.metric("Available Courses", "60")

    st.markdown("")

    render_html("""
<style>
*{box-sizing:border-box;margin:0;padding:0;font-family:'Inter',sans-serif;}
.wrap{background:#ffffff;border:1px solid #e5e7eb;border-radius:20px;padding:36px 40px;
      box-shadow:0 4px 16px rgba(0,0,0,0.05);}
.title{font-size:24px;font-weight:800;color:#111827;margin-bottom:10px;}
.sub{font-size:15px;color:#6b7280;line-height:1.7;margin-bottom:30px;max-width:720px;}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:20px;}
.step{background:#f9fafb;border:1px solid #f3f4f6;border-radius:14px;padding:24px;}
.num{display:inline-flex;width:36px;height:36px;align-items:center;justify-content:center;
     background:#eef2ff;color:#4f46e5;border-radius:9px;font-weight:800;font-size:15px;margin-bottom:14px;}
.sh{font-size:16px;font-weight:700;color:#111827;margin-bottom:8px;}
.sp{font-size:14px;color:#6b7280;line-height:1.65;}
</style>
<div class="wrap">
  <div class="title">How CourseIQ Works</div>
  <div class="sub">CourseIQ uses a <strong>Multi-Model Ensemble Intelligence Engine</strong> to forecast your academic outcomes
                   and match you to courses where you are most likely to thrive.</div>
  <div class="grid">
    <div class="step">
      <div class="num">1</div>
      <div class="sh">Department-Specific Models</div>
      <div class="sp">Every major (CS, Math, IS, SE, Stat) is evaluated by 4 specialized ML models trained exclusively
                       on historical student data for that department.</div>
    </div>
    <div class="step">
      <div class="num">2</div>
      <div class="sh">Algorithmic Voting Consensus</div>
      <div class="sp">Gradient Boosting + Random Forest vote on pass/fail; Gradient Boosting + SVR predict your
                       expected grade. Results are merged via weighted consensus.</div>
    </div>
    <div class="step">
      <div class="num">3</div>
      <div class="sh">Confidence &amp; Pre-req Guards</div>
      <div class="sp">A Confidence Score reflects model agreement. Prerequisite checks ensure you are
                       never recommended a course you are not eligible for.</div>
    </div>
  </div>
</div>
""", height=390)
