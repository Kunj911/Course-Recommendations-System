"""
app.py
-------
Course Recommendation System — Streamlit Dashboard.

Aesthetic direction: Dark academic — deep navy/slate tones, golden accents,
clean typography using Space Mono for headers and clear sans for body.
Professional but approachable; like a high-end university portal.
"""

import os
import sys
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

# ─── Page Config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="CourseIQ — Smart Course Recommendations",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
  }

  .main { background: #0d1117; }

  h1, h2, h3 { font-family: 'Space Mono', monospace !important; }

  .hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, #e2b96f 0%, #f5d89e 50%, #c9955a 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.2;
    margin-bottom: 0.25rem;
  }

  .hero-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 1.05rem;
    color: #8b949e;
    margin-bottom: 2rem;
  }

  .metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s;
  }

  .metric-card:hover { border-color: #e2b96f; }

  .dept-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.05em;
  }

  .warning-box {
    background: #1a1200;
    border-left: 4px solid #e2b96f;
    border-radius: 6px;
    padding: 0.75rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.9rem;
    color: #f0c27f;
  }

  .section-header {
    font-family: 'Space Mono', monospace;
    font-size: 1.1rem;
    color: #e2b96f;
    border-bottom: 1px solid #30363d;
    padding-bottom: 0.4rem;
    margin: 1.5rem 0 1rem 0;
    letter-spacing: 0.03em;
  }

  .profile-pill {
    background: #1c2128;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    display: inline-block;
    margin: 0.2rem;
    font-size: 0.88rem;
    color: #cdd9e5;
  }

  .stButton > button {
    background: linear-gradient(135deg, #e2b96f, #c9955a) !important;
    color: #0d1117 !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.5rem !important;
    font-size: 0.95rem !important;
    width: 100% !important;
    letter-spacing: 0.03em !important;
    transition: opacity 0.2s !important;
  }

  .stButton > button:hover { opacity: 0.88 !important; }

  .sidebar-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.05rem;
    font-weight: 700;
    color: #e2b96f;
    margin-bottom: 1.2rem;
    letter-spacing: 0.05em;
  }

  div[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #21262d !important;
  }

  .rec-row-cs    { border-left: 4px solid #4FC3F7; }
  .rec-row-math  { border-left: 4px solid #81C784; }
  .rec-row-stat  { border-left: 4px solid #FFB74D; }
  .rec-row-is    { border-left: 4px solid #CE93D8; }
  .rec-row-se    { border-left: 4px solid #F48FB1; }

  .stNumberInput input, .stTextInput input {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #cdd9e5 !important;
    border-radius: 6px !important;
  }

  .stSlider > div > div { color: #e2b96f !important; }

  footer { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ─── Lazy initialization helpers ──────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading recommendation engine...")
def get_recommender():
    """Load the recommender once and cache it for the session."""
    from models.recommender import CourseRecommender
    return CourseRecommender()


def ensure_data_and_models():
    """
    One-time setup: generates DB, trains models if they don't exist.
    Runs only on first launch.
    """
    db_path    = os.path.join(os.path.dirname(__file__), "database", "course_recommendation.db")
    model_path = os.path.join(os.path.dirname(__file__), "models", "saved", "CS_logistic.pkl")

    if not os.path.exists(db_path) or os.path.getsize(db_path) < 1000:
        with st.spinner("🏗️  First-time setup: generating database..."):
            from data_generator import run_data_generation
            run_data_generation()

    if not os.path.exists(model_path):
        with st.spinner("🧠 First-time setup: training department models..."):
            from models.train_department_models import train_all_models
            train_all_models()


# ─── Visualization helpers ─────────────────────────────────────────────────────

DEPT_COLORS = {
    "CS":   "#4FC3F7",
    "MATH": "#81C784",
    "STAT": "#FFB74D",
    "IS":   "#CE93D8",
    "SE":   "#F48FB1",
}

PLOT_BGCOLOR  = "#0d1117"
PLOT_PAPER    = "#0d1117"
GRID_COLOR    = "#21262d"
FONT_COLOR    = "#cdd9e5"
ACCENT        = "#e2b96f"


def make_success_chart(df: pd.DataFrame) -> go.Figure:
    colors = [DEPT_COLORS.get(d, "#888") for d in df["department"]]
    labels = df["course_code"] + "<br><span style='font-size:10px'>" + df["department"] + "</span>"

    fig = go.Figure(go.Bar(
        x=df.index,
        y=(df["success_probability"] * 100).round(1),
        text=(df["success_probability"] * 100).round(1).astype(str) + "%",
        textposition="outside",
        marker_color=colors,
        marker_line_width=0,
        customdata=df[["course_name", "department"]].values,
        hovertemplate="<b>%{customdata[0]}</b><br>Dept: %{customdata[1]}<br>Success: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Success Probability by Course", font=dict(color=ACCENT, family="Space Mono", size=14)),
        xaxis=dict(tickmode="array", tickvals=df.index, ticktext=df["course_code"],
                   tickfont=dict(size=10, color=FONT_COLOR), gridcolor=GRID_COLOR),
        yaxis=dict(title="Success Probability (%)", title_font=dict(color=FONT_COLOR),
                   tickfont=dict(color=FONT_COLOR), range=[0, 115], gridcolor=GRID_COLOR),
        plot_bgcolor=PLOT_BGCOLOR, paper_bgcolor=PLOT_PAPER,
        font=dict(color=FONT_COLOR),
        margin=dict(t=50, b=60, l=50, r=20),
        height=340,
        showlegend=False,
    )
    return fig


def make_grade_chart(df: pd.DataFrame) -> go.Figure:
    colors = [DEPT_COLORS.get(d, "#888") for d in df["department"]]

    fig = go.Figure(go.Bar(
        x=df.index,
        y=df["expected_grade"].round(2),
        text=df["expected_grade"].round(1).astype(str),
        textposition="outside",
        marker_color=colors,
        marker_line_width=0,
        customdata=df[["course_name", "department"]].values,
        hovertemplate="<b>%{customdata[0]}</b><br>Dept: %{customdata[1]}<br>Expected Grade: %{y:.1f}/10<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Expected Grade by Course", font=dict(color=ACCENT, family="Space Mono", size=14)),
        xaxis=dict(tickmode="array", tickvals=df.index, ticktext=df["course_code"],
                   tickfont=dict(size=10, color=FONT_COLOR), gridcolor=GRID_COLOR),
        yaxis=dict(title="Expected Grade (0–10)", title_font=dict(color=FONT_COLOR),
                   tickfont=dict(color=FONT_COLOR), range=[0, 12], gridcolor=GRID_COLOR),
        plot_bgcolor=PLOT_BGCOLOR, paper_bgcolor=PLOT_PAPER,
        font=dict(color=FONT_COLOR),
        margin=dict(t=50, b=60, l=50, r=20),
        height=340,
        showlegend=False,
    )
    return fig


def make_scatter_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for dept, grp in df.groupby("department"):
        fig.add_trace(go.Scatter(
            x=grp["success_probability"] * 100,
            y=grp["expected_grade"],
            mode="markers+text",
            text=grp["course_code"],
            textposition="top center",
            textfont=dict(size=9, color=FONT_COLOR),
            marker=dict(
                color=DEPT_COLORS.get(dept, "#888"),
                size=grp["recommendation_score"] * 2 + 4,
                opacity=0.85,
                line=dict(width=1, color="#21262d"),
            ),
            name=dept,
            hovertemplate="<b>%{text}</b><br>Success: %{x:.1f}%<br>Grade: %{y:.1f}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text="Success vs Grade Landscape (size = recommendation score)",
                   font=dict(color=ACCENT, family="Space Mono", size=13)),
        xaxis=dict(title="Success Probability (%)", title_font=dict(color=FONT_COLOR),
                   tickfont=dict(color=FONT_COLOR), gridcolor=GRID_COLOR),
        yaxis=dict(title="Expected Grade", title_font=dict(color=FONT_COLOR),
                   tickfont=dict(color=FONT_COLOR), range=[0, 11], gridcolor=GRID_COLOR),
        plot_bgcolor=PLOT_BGCOLOR, paper_bgcolor=PLOT_PAPER,
        font=dict(color=FONT_COLOR),
        legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1,
                    font=dict(color=FONT_COLOR)),
        margin=dict(t=50, b=50, l=60, r=20),
        height=400,
    )
    return fig


def make_radar_chart(student: dict) -> go.Figure:
    categories = ["CS", "Math", "Statistics", "Info Systems", "Soft. Eng."]
    values = [
        student["cs_proficiency"],
        student["math_proficiency"],
        student["stat_proficiency"],
        student["is_proficiency"],
        student["se_proficiency"],
    ]
    # Close the radar polygon
    categories += [categories[0]]
    values += [values[0]]

    fig = go.Figure(go.Scatterpolar(
        r=values,
        theta=categories,
        fill="toself",
        fillcolor="rgba(226,185,111,0.15)",
        line=dict(color=ACCENT, width=2.5),
        marker=dict(color=ACCENT, size=7),
        name=student.get("name", "Student"),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="#161b22",
            radialaxis=dict(visible=True, range=[0, 10], tickfont=dict(color=FONT_COLOR, size=9),
                            gridcolor=GRID_COLOR, linecolor=GRID_COLOR),
            angularaxis=dict(tickfont=dict(color=FONT_COLOR, size=11), gridcolor=GRID_COLOR,
                             linecolor=GRID_COLOR),
        ),
        paper_bgcolor=PLOT_PAPER,
        font=dict(color=FONT_COLOR),
        title=dict(text="Student Proficiency Profile",
                   font=dict(color=ACCENT, family="Space Mono", size=14)),
        showlegend=False,
        height=380,
        margin=dict(t=60, b=30, l=30, r=30),
    )
    return fig


# ─── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar() -> dict | None:
    with st.sidebar:
        st.markdown('<div class="sidebar-title">🎓 STUDENT PROFILE</div>', unsafe_allow_html=True)

        name = st.text_input("Full Name", placeholder="e.g. Alex Kumar", key="name")

        st.markdown("---")
        st.markdown("**Academic Standing**")
        cgpa  = st.number_input("CGPA (0–10)", min_value=0.0, max_value=10.0, value=7.5, step=0.1,
                                 help="Your cumulative GPA on a 10-point scale")
        hardworking = st.slider("Hardworking Level", min_value=1, max_value=10, value=7,
                                 help="1 = minimal effort, 10 = extremely dedicated")
        year  = st.selectbox("Current Year", options=[1, 2, 3, 4], index=1)
        credits = st.number_input("Credits Completed", min_value=0, max_value=200, value=60, step=5)

        st.markdown("---")
        st.markdown("**Subject Proficiencies** *(0–10)*")
        cs   = st.number_input("💻 CS Proficiency",    min_value=0.0, max_value=10.0, value=7.0, step=0.5,
                                help="Computer Science aptitude: algorithms, programming, systems")
        math = st.number_input("📐 Math Proficiency",  min_value=0.0, max_value=10.0, value=6.0, step=0.5,
                                help="Mathematical reasoning: calculus, linear algebra, discrete math")
        stat = st.number_input("📊 Statistics Proficiency", min_value=0.0, max_value=10.0, value=6.5, step=0.5,
                                help="Statistical thinking: probability, inference, data analysis")
        is_  = st.number_input("🏢 IS Proficiency",    min_value=0.0, max_value=10.0, value=6.0, step=0.5,
                                help="Information Systems: business processes, enterprise systems")
        se   = st.number_input("⚙️ SE Proficiency",    min_value=0.0, max_value=10.0, value=6.5, step=0.5,
                                help="Software Engineering: design patterns, architecture, testing")

        st.markdown("---")
        top_n = st.slider("Number of Recommendations", min_value=3, max_value=15, value=10)

        submitted = st.button("🎯 Get Recommendations", use_container_width=True)

        if submitted:
            if not name.strip():
                st.error("Please enter your name.")
                return None

            from utils.validators import build_student_dict, validate_student_profile
            student = build_student_dict(
                name=name, cgpa=cgpa, hardworking_level=hardworking,
                cs_proficiency=cs, math_proficiency=math, stat_proficiency=stat,
                is_proficiency=is_, se_proficiency=se,
                year=year, credits_completed=credits,
            )
            valid, errors = validate_student_profile(student)
            if not valid:
                for err in errors:
                    st.error(err)
                return None

            return {"student": student, "top_n": top_n}

    return None


# ─── Main content ─────────────────────────────────────────────────────────────

def render_hero():
    st.markdown('<div class="hero-title">CourseIQ</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">AI-powered course recommendations tailored to your unique academic profile. '
        'Five specialized models — one per department — predict your success before you enroll.</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card">📚 <b>60 Courses</b><br><span style="color:#8b949e;font-size:0.85rem">Across 5 departments</span></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">🧠 <b>10 ML Models</b><br><span style="color:#8b949e;font-size:0.85rem">Logistic + Linear per dept</span></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">🎯 <b>Prerequisite Aware</b><br><span style="color:#8b949e;font-size:0.85rem">Enforces course chains</span></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card">⚡ <b>Instant Inference</b><br><span style="color:#8b949e;font-size:0.85rem">No database writes</span></div>', unsafe_allow_html=True)


def render_profile_summary(student: dict):
    st.markdown('<div class="section-header">👤 STUDENT PROFILE</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"### {student['name']}")
        st.markdown(f"**Year {student['year']}** · {student['credits_completed']} credits")

    with col2:
        pills_html = "".join([
            f'<span class="profile-pill">CGPA <b>{student["cgpa"]}</b></span>',
            f'<span class="profile-pill">💪 Hardworking <b>{student["hardworking_level"]}/10</b></span>',
            f'<span class="profile-pill" style="border-color:#4FC3F7">💻 CS <b>{student["cs_proficiency"]}</b></span>',
            f'<span class="profile-pill" style="border-color:#81C784">📐 Math <b>{student["math_proficiency"]}</b></span>',
            f'<span class="profile-pill" style="border-color:#FFB74D">📊 Stat <b>{student["stat_proficiency"]}</b></span>',
            f'<span class="profile-pill" style="border-color:#CE93D8">🏢 IS <b>{student["is_proficiency"]}</b></span>',
            f'<span class="profile-pill" style="border-color:#F48FB1">⚙️ SE <b>{student["se_proficiency"]}</b></span>',
        ])
        st.markdown(pills_html, unsafe_allow_html=True)


def render_recommendations_table(df: pd.DataFrame):
    st.markdown('<div class="section-header">🏆 TOP RECOMMENDATIONS</div>', unsafe_allow_html=True)

    display_df = df.copy().reset_index()
    display_df["Rank"] = display_df["rank"]
    display_df["Code"] = display_df["course_code"]
    display_df["Course"] = display_df["course_name"]
    display_df["Dept"] = display_df["department"]
    display_df["Difficulty"] = display_df["difficulty"].apply(lambda x: f"{x:.1f}/10")
    display_df["Success %"] = (display_df["success_probability"] * 100).apply(lambda x: f"{x:.1f}%")
    display_df["Grade"] = display_df["expected_grade"].apply(lambda x: f"{x:.1f}/10")
    display_df["Score"] = display_df["recommendation_score"].apply(lambda x: f"{x:.2f}")
    display_df["Readiness"] = display_df["readiness_label"]
    display_df["Prereq"] = display_df["prereq_met"].apply(lambda x: "✅ Met" if x else "⚠️ Missing")

    table_cols = ["Rank", "Code", "Course", "Dept", "Difficulty", "Success %", "Grade", "Score", "Readiness", "Prereq"]
    styled = display_df[table_cols].style.apply(
        lambda row: [
            f"background-color: {'rgba(79,195,247,0.08)' if row['Dept']=='CS' else 'rgba(129,199,132,0.08)' if row['Dept']=='MATH' else 'rgba(255,183,77,0.08)' if row['Dept']=='STAT' else 'rgba(206,147,216,0.08)' if row['Dept']=='IS' else 'rgba(244,143,177,0.08)'}"
        ] * len(table_cols),
        axis=1,
    )
    st.dataframe(display_df[table_cols], use_container_width=True, hide_index=True)

def render_prereq_warnings(df: pd.DataFrame):
    warnings_df = df[~df["prereq_met"]]
    if warnings_df.empty:
        return

    st.markdown('<div class="section-header">⚠️ PREREQUISITE WARNINGS</div>', unsafe_allow_html=True)
    for _, row in warnings_df.iterrows():
        st.markdown(
            f'<div class="warning-box">{row["prereq_warning"]}</div>',
            unsafe_allow_html=True
        )


def render_explanation_panel(df: pd.DataFrame, student: dict, recommender):
    st.markdown('<div class="section-header">💡 WHY THESE RECOMMENDATIONS?</div>', unsafe_allow_html=True)

    top3 = df.head(3)
    cols = st.columns(3)
    for i, (_, row) in enumerate(top3.iterrows()):
        with cols[i]:
            dept_color = DEPT_COLORS.get(row["department"], "#888")
            explanation = recommender.get_student_explanation(student, row)
            st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid {dept_color}">
                <b style="color:{dept_color}">#{i+1} {row['course_code']}</b><br>
                <span style="font-size:0.85rem;color:#8b949e">{row['course_name']}</span><br><br>
                <pre style="font-size:0.78rem;color:#cdd9e5;white-space:pre-wrap;font-family:'DM Sans',sans-serif">{explanation}</pre>
            </div>
            """, unsafe_allow_html=True)


def render_visualizations(df: pd.DataFrame, student: dict):
    st.markdown('<div class="section-header">📈 ANALYTICS</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Success Probability", "📝 Expected Grades", "🌐 Landscape", "🕸️ Proficiency Radar"])

    with tab1:
        st.plotly_chart(make_success_chart(df), use_container_width=True)

    with tab2:
        st.plotly_chart(make_grade_chart(df), use_container_width=True)

    with tab3:
        st.plotly_chart(make_scatter_chart(df), use_container_width=True)
        st.caption("Bubble size reflects overall recommendation score. Top-right = high success + high grade.")

    with tab4:
        col_radar, col_info = st.columns([2, 1])
        with col_radar:
            st.plotly_chart(make_radar_chart(student), use_container_width=True)
        with col_info:
            st.markdown("**Proficiency Breakdown**")
            profs = {
                "💻 CS": student["cs_proficiency"],
                "📐 Math": student["math_proficiency"],
                "📊 Stat": student["stat_proficiency"],
                "🏢 IS": student["is_proficiency"],
                "⚙️ SE": student["se_proficiency"],
            }
            for label, val in profs.items():
                emoji = "🟢" if val >= 7 else "🟡" if val >= 5 else "🔴"
                st.markdown(f"{emoji} **{label}**: {val:.1f}/10")

            strongest = max(profs, key=profs.get)
            st.markdown(f"\n**Strongest domain**: {strongest}")
            st.markdown(f"**Recommended focus**: Courses in your strong domains will yield highest predicted performance.")


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main():
    ensure_data_and_models()

    render_hero()
    st.markdown("---")

    result = render_sidebar()

    if result is None:
        # Landing state — show instructions
        st.markdown("""
        <div style="text-align:center; padding: 4rem 2rem; color: #8b949e;">
            <div style="font-size:4rem;">🎓</div>
            <h3 style="font-family:'Space Mono',monospace; color:#cdd9e5;">Ready to find your perfect courses?</h3>
            <p>Enter your academic profile in the sidebar and click <b style="color:#e2b96f;">Get Recommendations</b>.</p>
            <br>
            <p style="font-size:0.9rem;">Five specialized ML models — trained on 500 student histories across 60 courses —
            will predict your success probability and expected grade for every course,
            then rank them to surface your best opportunities.</p>
        </div>
        """, unsafe_allow_html=True)

        # Show department legend
        st.markdown('<div class="section-header">🗂️ DEPARTMENT GUIDE</div>', unsafe_allow_html=True)
        cols = st.columns(5)
        dept_info = {
            "CS": ("💻", "Computer Science", "15 courses", "#4FC3F7"),
            "MATH": ("📐", "Mathematics", "12 courses", "#81C784"),
            "STAT": ("📊", "Statistics", "10 courses", "#FFB74D"),
            "IS": ("🏢", "Info Systems", "12 courses", "#CE93D8"),
            "SE": ("⚙️", "Soft. Engineering", "11 courses", "#F48FB1"),
        }
        for i, (dept, (emoji, name, count, color)) in enumerate(dept_info.items()):
            with cols[i]:
                st.markdown(f"""
                <div class="metric-card" style="border-left: 4px solid {color}; text-align:center">
                    <div style="font-size:2rem">{emoji}</div>
                    <b style="color:{color}">{dept}</b><br>
                    <span style="font-size:0.85rem;color:#cdd9e5">{name}</span><br>
                    <span style="font-size:0.75rem;color:#8b949e">{count}</span>
                </div>
                """, unsafe_allow_html=True)
        return

    # ── Got recommendations ──
    student = result["student"]
    top_n   = result["top_n"]

    recommender = get_recommender()

    with st.spinner("🔮 Running department-specialized models..."):
        df = recommender.recommend(student, top_n=top_n)

    if df.empty:
        st.warning("No recommendations available. All courses may require prerequisites.")
        return

    render_profile_summary(student)
    render_prereq_warnings(df)

    st.markdown("")
    render_recommendations_table(df)

    st.markdown("")
    render_explanation_panel(df, student, recommender)

    st.markdown("")
    render_visualizations(df, student)

    # Department distribution summary
    st.markdown('<div class="section-header">📊 DEPARTMENT DISTRIBUTION</div>', unsafe_allow_html=True)
    dept_counts = df["department"].value_counts().reset_index()
    dept_counts.columns = ["Department", "Count"]
    fig_dist = px.pie(
        dept_counts, values="Count", names="Department",
        color="Department",
        color_discrete_map=DEPT_COLORS,
        hole=0.45,
    )
    fig_dist.update_layout(
        paper_bgcolor=PLOT_PAPER, plot_bgcolor=PLOT_BGCOLOR,
        font=dict(color=FONT_COLOR),
        height=320,
        margin=dict(t=20, b=20, l=20, r=20),
        legend=dict(bgcolor="#161b22", font=dict(color=FONT_COLOR)),
    )
    col_pie, col_stats = st.columns([1, 1])
    with col_pie:
        st.plotly_chart(fig_dist, use_container_width=True)
    with col_stats:
        st.markdown("**Recommendation Statistics**")
        avg_success = df["success_probability"].mean() * 100
        avg_grade   = df["expected_grade"].mean()
        n_prereq_missing = (~df["prereq_met"]).sum()
        avg_score   = df["recommendation_score"].mean()

        st.metric("Avg Success Probability", f"{avg_success:.1f}%")
        st.metric("Avg Expected Grade",      f"{avg_grade:.2f}/10")
        st.metric("Prerequisite Warnings",   n_prereq_missing)
        st.metric("Avg Recommendation Score", f"{avg_score:.2f}/10")


if __name__ == "__main__":
    main()