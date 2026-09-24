"""
AI-Based Student Performance Prediction System - Streamlit Dashboard
Redesigned for Kaggle Student Performance Factors Dataset (6,607 records, 20 features).
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from prediction import StudentPredictor
from simulator import WhatIfSimulator
from goal_optimizer import GoalOptimizer
from recommendations import RecommendationEngine

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
        text-align: center;
        margin-bottom: 15px;
    }
    .metric-card h4 {
        color: #475569;
        font-size: 0.9rem;
        margin-bottom: 6px;
        font-weight: 600;
        text-transform: uppercase;
    }
    .metric-card .val {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1e3a8a;
    }
    .metric-card .sub {
        font-size: 0.82rem;
        color: #64748b;
        margin-top: 4px;
    }
    .badge-low {
        background-color: #d1fae5;
        color: #065f46;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-moderate {
        background-color: #fef3c7;
        color: #92400e;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-high {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }
    .focus-card {
        background-color: #eff6ff;
        border-left: 5px solid #2563eb;
        padding: 18px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    .focus-card h3 {
        color: #1e40af;
        margin-top: 0;
        margin-bottom: 8px;
        font-size: 1.25rem;
    }
    .strategy-box {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 10px;
        padding: 16px;
        color: #166534;
        font-size: 1.05rem;
        margin-bottom: 20px;
    }
    .sidebar-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_ml_components():
    predictor = StudentPredictor()
    simulator = WhatIfSimulator(predictor)
    optimizer = GoalOptimizer(predictor)
    recommend_engine = RecommendationEngine(predictor)
    return predictor, simulator, optimizer, recommend_engine

@st.cache_data
def get_dataset():
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'StudentPerformanceFactors.csv')
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return None

@st.cache_data
def get_metrics_summary():
    metrics_path = os.path.join(os.path.dirname(__file__), 'models', 'model_metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return None

def render_risk_badge(risk_level: str):
    if risk_level == "LOW RISK":
        return '<span class="badge-low">🟢 LOW RISK</span>'
    elif risk_level == "MODERATE RISK":
        return '<span class="badge-moderate">🟡 MODERATE RISK</span>'
    else:
        return '<span class="badge-high">🔴 HIGH RISK</span>'

# Sidebar Navigation
st.sidebar.markdown("<div class='sidebar-title'>🎓 Student Performance Prediction</div>", unsafe_allow_html=True)
st.sidebar.caption("Kaggle Real Dataset ML System")

page = st.sidebar.radio(
    "Navigate System",
    [
        "📌 Dashboard Overview",
        "🎯 Predict Performance",
        "⚡ What-If Simulator",
        "🚀 Goal-Based Prediction",
        "💡 Personalized Improvement Plan",
        "📊 Model Performance & Comparison"
    ]
)

predictor, simulator, optimizer, recommend_engine = get_ml_components()
df = get_dataset()
metrics_summary = get_metrics_summary()

st.sidebar.markdown("---")
st.sidebar.markdown("### 🏆 Active Kaggle ML Model")
if metrics_summary:
    st.sidebar.success(f"**Model:** {metrics_summary['best_model_name']}")
    best_r2 = metrics_summary['leaderboard'][metrics_summary['best_model_name']]['R2']
    best_rmse = metrics_summary['leaderboard'][metrics_summary['best_model_name']]['RMSE']
    st.sidebar.metric("R² Score", f"{best_r2:.4f}")
    st.sidebar.metric("RMSE Error", f"{best_rmse:.4f}")

# PAGE 1: DASHBOARD OVERVIEW
if page == "📌 Dashboard Overview":
    st.title("🎓 Student Performance Prediction — Analytics Dashboard")
    st.markdown("""
    This application utilizes machine learning regression trained on the official **Kaggle Student Performance Factors Dataset**
    (6,607 student records across 20 academic, behavioral, and demographic features).
    """)
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>Total Records</h4>
            <div class="val">6,607</div>
            <div class="sub">Kaggle Student Profiles</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        best_name = metrics_summary['best_model_name'] if metrics_summary else "Linear Regression"
        st.markdown(f"""
        <div class="metric-card">
            <h4>Selected ML Model</h4>
            <div class="val" style="font-size: 1.4rem;">{best_name}</div>
            <div class="sub">Top R² Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        r2_val = metrics_summary['leaderboard'][best_name]['R2'] if metrics_summary else 0.7711
        st.markdown(f"""
        <div class="metric-card">
            <h4>R² Model Accuracy</h4>
            <div class="val">{r2_val:.3f}</div>
            <div class="sub">Variance Explained</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h4>Features</h4>
            <div class="val">20</div>
            <div class="sub">Academic & Social Drivers</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("### 📊 Kaggle Dataset Exploratory Data Analysis")
    
    if df is not None:
        tab1, tab2, tab3 = st.tabs(["Target Distribution", "Feature Scatter Correlations", "Sample Data Records"])
        
        with tab1:
            fig_hist = px.histogram(
                df, x="Exam_Score", nbins=30, color_discrete_sequence=['#2563eb'],
                title="Exam Score Distribution across 6,607 Kaggle Student Records"
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            
        with tab2:
            st.write("#### Key Feature Relationships with Final Exam Score")
            sc_col1, sc_col2 = st.columns(2)
            with sc_col1:
                fig1 = px.scatter(df, x="Hours_Studied", y="Exam_Score", color="Attendance", color_continuous_scale="Viridis", title="Hours Studied vs Exam Score (Colored by Attendance)")
                st.plotly_chart(fig1, use_container_width=True)
            with sc_col2:
                fig2 = px.scatter(df, x="Previous_Scores", y="Exam_Score", color="Tutoring_Sessions", color_continuous_scale="Plasma", title="Previous Scores vs Exam Score (Colored by Tutoring Sessions)")
                st.plotly_chart(fig2, use_container_width=True)
                
        with tab3:
            st.dataframe(df.head(15), use_container_width=True)

# PAGE 2: PREDICT PERFORMANCE
elif page == "🎯 Predict Performance":
    st.title("🎯 Predict Final Exam Score (Kaggle Model)")
    st.markdown("Enter student attributes below across academic, habit, and background drivers to predict expected exam score.")
    
    col_in, col_res = st.columns([1.1, 0.9])
    
    with col_in:
        t1, t2, t3 = st.tabs(["📚 Academic Drivers", "🏃 Habits & Behavior", "🏡 Social & Environment"])
        
        with t1:
            hours_studied = st.slider("Hours Studied / Week", 1, 44, 20, 1)
            attendance = st.slider("Attendance Percentage (%)", 50, 100, 80, 1)
            previous_scores = st.slider("Previous Exam Score", 40, 100, 70, 1)
            tutoring_sessions = st.slider("Tutoring Sessions / Month", 0, 8, 2, 1)
            
        with t2:
            sleep_hours = st.slider("Sleep Duration (hrs/night)", 4, 10, 7, 1)
            physical_activity = st.slider("Physical Activity (hrs/week)", 0, 6, 3, 1)
            motivation_level = st.selectbox("Motivation Level", ["Low", "Medium", "High"], index=1)
            extracurricular = st.selectbox("Extracurricular Activities", ["No", "Yes"], index=1)

        with t3:
            parental_involvement = st.selectbox("Parental Involvement", ["Low", "Medium", "High"], index=1)
            access_to_resources = st.selectbox("Access to Resources", ["Low", "Medium", "High"], index=1)
            family_income = st.selectbox("Family Income", ["Low", "Medium", "High"], index=1)
            teacher_quality = st.selectbox("Teacher Quality", ["Low", "Medium", "High"], index=1)
            peer_influence = st.selectbox("Peer Influence", ["Negative", "Neutral", "Positive"], index=1)
            school_type = st.selectbox("School Type", ["Public", "Private"], index=0)
            internet_access = st.selectbox("Internet Access", ["No", "Yes"], index=1)
            learning_disabilities = st.selectbox("Learning Disabilities", ["No", "Yes"], index=0)
            parental_education = st.selectbox("Parental Education Level", ["High School", "College", "Postgraduate"], index=1)
            distance_from_home = st.selectbox("Distance from Home", ["Near", "Moderate", "Far"], index=1)
            gender = st.selectbox("Gender", ["Female", "Male"], index=0)
            
    input_profile = {
        'Hours_Studied': hours_studied,
        'Attendance': attendance,
        'Previous_Scores': previous_scores,
        'Tutoring_Sessions': tutoring_sessions,
        'Sleep_Hours': sleep_hours,
        'Physical_Activity': physical_activity,
        'Motivation_Level': motivation_level,
        'Extracurricular_Activities': extracurricular,
        'Parental_Involvement': parental_involvement,
        'Access_to_Resources': access_to_resources,
        'Family_Income': family_income,
        'Teacher_Quality': teacher_quality,
        'Peer_Influence': peer_influence,
        'School_Type': school_type,
        'Internet_Access': internet_access,
        'Learning_Disabilities': learning_disabilities,
        'Parental_Education_Level': parental_education,
        'Distance_from_Home': distance_from_home,
        'Gender': gender
    }
    
    with col_res:
        st.subheader("📈 ML Prediction Output")
        res = predictor.predict(input_profile)
        pred_marks = res['predicted_marks']
        category = res['performance_category']
        risk = res['risk_level']
        interp = res['interpretation']
        importances = res['feature_importances']
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = pred_marks,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Predicted Exam Score", 'font': {'size': 18}},
            number = {'suffix': " / 100", 'font': {'size': 32, 'color': "#1e3a8a"}},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "#2563eb"},
                'steps': [
                    {'range': [0, 60], 'color': '#fee2e2'},
                    {'range': [60, 80], 'color': '#fef3c7'},
                    {'range': [80, 100], 'color': '#d1fae5'}
                ]
            }
        ))
        fig_gauge.update_layout(height=240, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        st.markdown(f"""
        <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div><strong>Category:</strong> {category}</div>
                <div>{render_risk_badge(risk)}</div>
            </div>
            <p style="margin-top: 8px; font-size: 0.9rem; color: #334155; margin-bottom: 0;">💡 <em>{interp}</em></p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🔍 Top Feature Drivers (Kaggle Model Weights)")
    
    top_imp = pd.DataFrame({
        'Feature': [col.replace('_', ' ') for col in importances.keys()],
        'Importance Weight': list(importances.values())
    }).sort_values(by='Importance Weight', ascending=False).head(8)
    
    fig_imp = px.bar(
        top_imp.sort_values(by='Importance Weight', ascending=True),
        x='Importance Weight', y='Feature', orientation='h', color='Importance Weight',
        color_continuous_scale='Blues', text_auto='.1%', title="Primary Feature Drivers in Kaggle ML Model"
    )
    fig_imp.update_layout(height=320)
    st.plotly_chart(fig_imp, use_container_width=True)

# PAGE 3: WHAT-IF SIMULATOR
elif page == "⚡ What-If Simulator":
    st.title("⚡ What-If Performance Simulator (Kaggle Features)")
    st.markdown("Modify student habits and observe live predicted score changes computed by the Kaggle ML model.")
    
    col_b, col_s = st.columns(2)
    
    with col_b:
        st.subheader("📌 Baseline Student Profile")
        b_hrs = st.slider("Baseline Hours Studied", 1, 44, 15, key="b_hrs")
        b_att = st.slider("Baseline Attendance (%)", 50, 100, 70, key="b_att")
        b_tut = st.slider("Baseline Tutoring Sessions", 0, 8, 1, key="b_tut")
        b_prev = st.slider("Baseline Previous Scores", 40, 100, 65, key="b_prev")
        
    with col_s:
        st.subheader("🔮 What-If Scenario Modifications")
        s_hrs = st.slider("Scenario Hours Studied", 1, 44, 25, key="s_hrs")
        s_att = st.slider("Scenario Attendance (%)", 50, 100, 85, key="s_att")
        s_tut = st.slider("Scenario Tutoring Sessions", 0, 8, 3, key="s_tut")
        s_prev = st.slider("Scenario Previous Scores", 40, 100, 65, key="s_prev")
        
    baseline_dict = {'Hours_Studied': b_hrs, 'Attendance': b_att, 'Tutoring_Sessions': b_tut, 'Previous_Scores': b_prev}
    scenario_dict = {'Hours_Studied': s_hrs, 'Attendance': s_att, 'Tutoring_Sessions': s_tut, 'Previous_Scores': s_prev}
    
    sim_res = simulator.simulate_scenario(baseline_dict, scenario_dict)
    
    st.markdown("---")
    st.subheader("📊 Simulation Comparison Summary")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"<div class='metric-card'><h4>Current Prediction</h4><div class='val'>{sim_res['baseline_score']}</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='metric-card'><h4>Scenario Prediction</h4><div class='val' style='color:#2563eb;'>{sim_res['scenario_score']}</div></div>", unsafe_allow_html=True)
    with c3:
        delta = sim_res['mark_delta']
        color = "#16a34a" if delta >= 0 else "#dc2626"
        st.markdown(f"<div class='metric-card'><h4>Predicted Lift</h4><div class='val' style='color:{color};'>+{delta}</div></div>", unsafe_allow_html=True)

# PAGE 4: GOAL-BASED PREDICTION
elif page == "🚀 Goal-Based Prediction":
    st.title("🚀 Goal-Based Target Optimizer & Recommendation Report")
    st.markdown("Define a target final examination score and generate a comprehensive, actionable **Goal Achievement Recommendation Report**.")
    
    c_in, c_target = st.columns(2)
    with c_in:
        st.subheader("📋 Current Student Profile")
        g_hrs = st.slider("Weekly Study Hours (hrs/wk)", 1, 44, 15, key="g_hrs")
        g_att = st.slider("Class Attendance (%)", 50, 100, 72, key="g_att")
        g_tut = st.slider("Tutoring Sessions / Month", 0, 8, 1, key="g_tut")
        g_prev = st.slider("Previous Exam Score", 40, 100, 65, key="g_prev")
        
    cur_profile = {
        'Hours_Studied': g_hrs,
        'Attendance': g_att,
        'Tutoring_Sessions': g_tut,
        'Previous_Scores': g_prev
    }
    cur_score = predictor.predict(cur_profile)['predicted_marks']
    
    with c_target:
        st.subheader("🎯 Define Target Exam Score")
        st.info(f"Current Predicted Score: **{cur_score} / 100**")
        target_score = st.slider("Target Exam Score Goal", float(min(100.0, np.ceil(cur_score))), 100.0, min(95.0, max(cur_score + 5.0, 75.0)), 1.0)
        
        gen_report_btn = st.button("📄 Generate Goal Recommendation Report", type="primary", use_container_width=True)
        
    st.markdown("---")
    
    report_res = optimizer.generate_recommendation_report(cur_profile, target_score)
    
    if report_res['status'] == 'ALREADY_ACHIEVED':
        st.success(report_res['message'])
    elif report_res['status'] == 'UNFEASIBLE':
        st.error(report_res['message'])
    else:
        st.subheader("📄 Goal Achievement Recommendation Report")
        
        # Executive Summary Metric Cards
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.markdown(f"<div class='metric-card'><h4>Baseline Score</h4><div class='val'>{report_res['baseline_score']}</div></div>", unsafe_allow_html=True)
        with col_m2:
            st.markdown(f"<div class='metric-card'><h4>Target Goal</h4><div class='val' style='color:#2563eb;'>{report_res['target_score']}</div></div>", unsafe_allow_html=True)
        with col_m3:
            st.markdown(f"<div class='metric-card'><h4>Required Lift</h4><div class='val' style='color:#16a34a;'>+{report_res['gap']}</div></div>", unsafe_allow_html=True)
        with col_m4:
            st.markdown(f"<div class='metric-card'><h4>Strategy Intensity</h4><div class='val' style='font-size:1.3rem;'>{report_res['effort_level']}</div></div>", unsafe_allow_html=True)
            
        st.markdown("#### 🎯 Recommended Primary Habit Plan")
        p_plan = report_res['primary_plan']
        
        plan_df = pd.DataFrame([
            {
                'Controllable Metric': 'Weekly Study Hours',
                'Baseline Value': f"{g_hrs} hrs/wk",
                'Recommended Target': f"{p_plan['inputs']['Hours_Studied']} hrs/wk",
                'Delta Lift': f"+{p_plan['delta_hours']} hrs",
                'Impact Description': 'Boosts core concept retention & problem solving'
            },
            {
                'Controllable Metric': 'Class Attendance',
                'Baseline Value': f"{g_att}%",
                'Recommended Target': f"{p_plan['inputs']['Attendance']}%",
                'Delta Lift': f"+{p_plan['delta_attendance']}%",
                'Impact Description': 'Ensures complete lecture coverage'
            },
            {
                'Controllable Metric': 'Tutoring Sessions',
                'Baseline Value': f"{g_tut} /mo",
                'Recommended Target': f"{p_plan['inputs']['Tutoring_Sessions']} /mo",
                'Delta Lift': f"+{p_plan['delta_tutoring']} sessions",
                'Impact Description': 'Targeted 1-on-1 mentorship for weak modules'
            }
        ])
        st.table(plan_df)
        
        st.markdown("#### 🚀 Step-by-Step Action Roadmap")
        t_phase1, t_phase2, t_phase3 = st.tabs(["📅 Phase 1: Attendance", "⏱️ Phase 2: Study Hours", "👨‍🏫 Phase 3: Tutoring"])
        
        with t_phase1:
            st.write(f"- Increase class attendance from **{g_att}%** to **{p_plan['inputs']['Attendance']}%**.")
            st.write("- Review lecture summary notes within 24 hours of each session.")
            st.write("- Sit in front rows to minimize classroom distractions.")
            
        with t_phase2:
            st.write(f"- Expand weekly study time from **{g_hrs} hrs/wk** to **{p_plan['inputs']['Hours_Studied']} hrs/wk** (~{round(p_plan['inputs']['Hours_Studied']/7, 1)} hrs/day).")
            st.write("- Implement 50-minute focused study sessions with 10-minute breaks.")
            st.write("- Focus on active recall and practice problems instead of passive reading.")
            
        with t_phase3:
            st.write(f"- Increase tutoring sessions from **{g_tut}/mo** to **{p_plan['inputs']['Tutoring_Sessions']}/mo**.")
            st.write("- Prepare a list of specific weak topics prior to attending tutoring.")
            
        st.markdown("---")
        st.download_button(
            label="📥 Download Full Goal Recommendation Report (.md)",
            data=report_res['report_md'],
            file_name=f"Goal_Recommendation_Report_Target_{target_score}.md",
            mime="text/markdown",
            use_container_width=True
        )

# PAGE 5: PERSONALIZED IMPROVEMENT PLAN
elif page == "💡 Personalized Improvement Plan":
    st.title("💡 Counterfactual Advisor & Improvement Plan")
    
    p_hrs = st.slider("Current Study Hours / Week", 1, 44, 15, key="plan_hrs")
    p_att = st.slider("Current Attendance (%)", 50, 100, 70, key="plan_att")
    p_tut = st.slider("Current Tutoring Sessions / Month", 0, 8, 1, key="plan_tut")
    p_prev = st.slider("Previous Score", 40, 100, 65, key="plan_prev")
    
    stu_profile = {'Hours_Studied': p_hrs, 'Attendance': p_att, 'Tutoring_Sessions': p_tut, 'Previous_Scores': p_prev}
    
    st.markdown("---")
    st.subheader("🎯 Best Improvement Focus (Counterfactual Analysis)")
    
    analysis = recommend_engine.analyze_best_improvement(stu_profile)
    best_focus = analysis['best_focus']
    
    if best_focus:
        st.markdown(f"""
        <div class="focus-card">
            <h3>🌟 Recommended Primary Focus: {best_focus['icon']} {best_focus['name']}</h3>
            <p>Increasing <strong>{best_focus['name']}</strong> from <strong>{best_focus['current_value']} {best_focus['unit']}</strong> to 
            <strong>{best_focus['simulated_value']} {best_focus['unit']}</strong> yields the single largest expected gain of 
            <strong style="color: #15803d;">+{best_focus['predicted_gain']} exam marks</strong>.</p>
        </div>
        """, unsafe_allow_html=True)
        
    gains_df = pd.DataFrame(analysis['factor_gains'])
    fig_sens = px.bar(
        gains_df, x='name', y='predicted_gain', color='predicted_gain', color_continuous_scale='Greens',
        text='predicted_gain', title="Simulated Mark Lift by Changing One Factor in Isolation"
    )
    fig_sens.update_traces(texttemplate='+%{text} marks', textposition='outside')
    st.plotly_chart(fig_sens, use_container_width=True)

# PAGE 6: MODEL PERFORMANCE
elif page == "📊 Model Performance & Comparison":
    st.title("📊 ML Model Benchmark Leaderboard (Kaggle Dataset)")
    
    if metrics_summary:
        leaderboard_df = pd.DataFrame(metrics_summary['leaderboard']).T.sort_values(by=['R2', 'RMSE'], ascending=[False, True])
        st.subheader("🏆 Algorithm Leaderboard (6,607 Student Profiles)")
        st.success(f"🥇 **Selected Best Model:** `{metrics_summary['best_model_name']}` with R² Score = **{leaderboard_df.loc[metrics_summary['best_model_name'], 'R2']:.4f}**")
        st.table(leaderboard_df.style.highlight_max(subset=['R2'], color='#bbf7d0').highlight_min(subset=['MAE', 'RMSE'], color='#bbf7d0'))


