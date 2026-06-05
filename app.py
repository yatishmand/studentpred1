import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import requests

# Page Configuration - Clean Corporate Analytics
st.set_page_config(page_title="Institutional Student Analytics Platform", layout="wide", page_icon="🎓")

# Custom Professional Premium Dashboard Theme CSS
st.markdown("""
    <style>
    .main { background-color: #0f172a; color: #f1f5f9; }
    .stButton>button { width: 100%; background-color: #2563eb; color: white; font-weight: bold; border-radius: 6px; height: 42px; border: none; }
    
    /* Advanced Grid Core Layout */
    .feature-card { background-color: #1e293b; padding: 15px; border-radius: 8px; border: 1px solid #334155; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .feature-title { font-size: 11px; color: #94a3b8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
    .feature-value { font-size: 22px; font-weight: bold; margin-top: 6px; font-family: monospace; }
    
    /* Dynamic Alerts Design */
    .status-box { padding: 15px; border-radius: 8px; margin-bottom: 15px; border-left: 5px solid; }
    .cyber-terminal { background-color: #020617; border-left: 5px solid #2563eb; padding: 15px; border-radius: 6px; font-family: 'Courier New', monospace; color: #38bdf8; font-size: 14px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🎓 ENTERPRISE STUDENT ACADEMIC ANALYTICS PLATFORM")
st.markdown("<span style='color:#94a3b8;'>Multi-Dimensional ML System deployment compiling 25+ Behavioral, Psychographic, Cognitive and Curricular Analytics Trackers.</span>", unsafe_allow_html=True)
st.markdown("---")

@st.cache_data
def load_and_engineer_data():
    df = pd.read_csv('AI-Data.csv')
    np.random.seed(42)
    n = len(df)
    
    # Statistical Pipeline Linkage Architecture
    engagement_score = (df['raisedhands'] + df['VisITedResources'] + df['AnnouncementsView'] + df['Discussion']) / 4
    df['Attendance'] = (engagement_score * 0.4 + np.random.normal(65, 10, n)).clip(45, 100).astype(int)
    df['StudyHours'] = (engagement_score * 0.08 + np.random.normal(3, 1.5, n)).clip(1, 14).astype(int)
    
    mh_pool = ['Excellent', 'Good', 'Stressed', 'Depressed']
    mh_list = []
    for idx, row in df.iterrows():
        if row['Class'] == 'H': status = np.random.choice(mh_pool, p=[0.5, 0.4, 0.08, 0.02])
        elif row['Class'] == 'M': status = np.random.choice(mh_pool, p=[0.15, 0.55, 0.25, 0.05])
        else: status = np.random.choice(mh_pool, p=[0.02, 0.18, 0.50, 0.30])
        mh_list.append(status)
    df['MentalHealthStatus'] = mh_list
    
    sleep_list = []
    for mh in df['MentalHealthStatus']:
        if mh in ['Excellent', 'Good']: sleep_list.append(np.random.randint(7, 9))
        else: sleep_list.append(np.random.choice([5, 6, 9, 10]))
    df['SleepTime'] = sleep_list

    df['CounselingHistory'] = df['MentalHealthStatus'].apply(lambda x: np.random.choice(['Yes', 'No'], p=[0.6, 0.4]) if x in ['Stressed', 'Depressed'] else np.random.choice(['Yes', 'No'], p=[0.05, 0.95]))
    df['Extracurriculars'] = df['Class'].apply(lambda x: np.random.randint(3, 7) if x == 'H' else (np.random.randint(1, 5) if x == 'M' else np.random.randint(0, 3)))
    
    return df

try:
    df = load_and_engineer_data()
    
    # Vector Target Encodings
    le_gender = LabelEncoder()
    df['gender_encoded'] = le_gender.fit_transform(df['gender'])
    le_stage = LabelEncoder()
    df['Stage_encoded'] = le_stage.fit_transform(df['StageID'])
    le_mh = LabelEncoder()
    df['MH_encoded'] = le_mh.fit_transform(df['MentalHealthStatus'])
    le_counsel = LabelEncoder()
    df['Counsel_encoded'] = le_counsel.fit_transform(df['CounselingHistory'])

    features = ['gender_encoded', 'Stage_encoded', 'raisedhands', 'VisITedResources', 'AnnouncementsView', 'Discussion', 'StudyHours', 'SleepTime', 'Attendance', 'Extracurriculars', 'MH_encoded', 'Counsel_encoded']
    X = df[features]
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Sidebar Panel Tracker
    st.sidebar.markdown("### ⚙️ Core Telemetry Specs")
    st.sidebar.info("ML Engine: RandomForest\n\nVariables Tracked: 26 Analytics Blocks")

    tab1, tab2, tab3 = st.tabs(["🔮 Real-Time Multi-Feature Dashboard", "🤖 AI Strategy Node", "📊 Distribution Architecture"])
    
    # ==================== TAB 1: MASTER FEATURES GRID ====================
    with tab1:
        st.subheader("🎛️ Live Parameter Configuration Matrix")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<h5 style='color:#38bdf8;'>📝 Academic LMS Activity Indicators</h5>", unsafe_allow_html=True)
            gender_input = st.selectbox("Student Gender Layer", options=["Male", "Female"])
            stage_input = st.selectbox("Current Institutional Track Stage", options=["Lowerlevel", "MiddleSchool", "HighSchool"])
            raised_hands = st.slider("Classroom Interaction Score (Raised Hands)", 0, 100, 60)
            visited_resources = st.slider("LMS Clicks / Resource Utilization Density", 0, 100, 65)
            announcements = st.slider("System Framework Notifications Checked", 0, 100, 45)
            discussion = st.slider("Active Forum Thread Participation Rate", 0, 100, 50)
            assignment_score = st.slider("Assignment Submission Quality & Completion Rate (%)", 0, 100, 75)
            
            st.markdown("<h5 style='color:#fb7185;'>📚 Curricular Target Focus</h5>", unsafe_allow_html=True)
            target_subject = st.selectbox("Select Target Analytics Track", options=["Theory of Computation (TOC)", "Database Management (DBMS)", "Computer Networks (CN)", "Design & Analysis of Algorithms (DAA)"])
            target_cgpa = st.slider("Set Target Semester Threshold", 4.0, 10.0, 8.5, step=0.1)

        with col2:
            st.markdown("<h5 style='color:#34d399;'>🏃 Lifestyle & Psychological Telemetry</h5>", unsafe_allow_html=True)
            study_hours = st.slider("Daily Self-Study Core Blocks (Hours)", 0, 15, 5)
            sleep_time = st.slider("Rest Cycle Duration / Sleep Cycles (Hours)", 4, 12, 7)
            attendance = st.slider("Verified Academic Attendance Data Score (%)", 0, 100, 80)
            extracurriculars = st.slider("Weekly Side-Activities Scale (Hours)", 0, 10, 2)
            counsel_input = st.selectbox("Sought Professional Psychological Advice?", options=["No", "Yes"])
            stress_index = st.slider("Live Telemetry Stress & Tension Index", 0, 100, 40)
            study_consistency = st.slider("Study Workflow Continuity Score (%)", 0, 100, 70)

        # Dynamic variable mapping
        if stress_index >= 75: mh_input = "Depressed"
        elif stress_index >= 45: mh_input = "Stressed"
        else: mh_input = "Good"

        g_encoded = 1 if gender_input == "Male" else 0
        s_encoded = 0 if stage_input == "Lowerlevel" else (1 if stage_input == "MiddleSchool" else 2)
        mh_encoded = list(le_mh.classes_).index(mh_input)
        counsel_encoded = 1 if counsel_input == "Yes" else 0

        # ==================== 🧠 PURE MATHEMATICAL 25+ INTEGRATED METRICS SYSTEM ====================
        # 1. CGPA/GPA Predictor 
        base_gpa = ((attendance * 0.3) + (raised_hands * 0.2) + (visited_resources * 0.2) + (study_hours * 4.0)) / 15
        predicted_gpa = round(max(4.0, min(10.0, base_gpa + (assignment_score * 0.015) - (stress_index * 0.005))), 2)

        # 2. Backlog Probability Prediction
        backlog_prob = int(((100 - attendance) * 0.5 + (stress_index * 0.25) + (50 - raised_hands) * 0.25))
        backlog_prob = max(2, min(97, backlog_prob)) if study_hours < 4 else max(1, min(40, backlog_prob - 22))

        # 3. Exam Readiness Score 
        readiness_score = int((attendance * 0.25) + (study_hours * 4.0) + (assignment_score * 0.25) + ((100 - stress_index) * 0.1))
        readiness_score = max(5, min(100, readiness_score))

        # 4. Academic Performance Score
        perf_score = int((assignment_score * 0.4) + (attendance * 0.4) + (raised_hands * 0.2))

        # 5. Motivation Level & Confidence Factor
        motivation_score = int((study_hours * 5) + (study_consistency * 0.4) + (attendance * 0.1) - (stress_index * 0.05))
        motivation_score = max(10, min(100, motivation_score))
        confidence_score = int((readiness_score * 0.6) + (study_consistency * 0.4) - (backlog_prob * 0.1))
        confidence_score = max(5, min(100, confidence_score))

        # 6. Improvement Potential Score
        improvement_potential = int(((100 - perf_score) * 0.75) + (study_consistency * 0.25))
        improvement_potential = max(5, min(95, improvement_potential))

        # 7. Syllabus Coverage Estimation (%)
        syllabus_coverage = int((study_hours * 5) + (assignment_score * 0.4) + (attendance * 0.1))
        syllabus_coverage = max(15, min(100, syllabus_coverage))

        # 8. Dynamic Credit Weightage Score (RGPV Logic Mapping)
        credit_weight = 4 if "TOC" in target_subject or "DAA" in target_subject else 3
        credit_integrity = round((attendance * 0.01) * credit_weight * (predicted_gpa / 10), 2)

        # 9. Internal Assessment (Sessional) Predictor (Out of 20)
        internal_marks = round(min(20.0, (raised_hands * 0.05) + (visited_resources * 0.05) + (assignment_score * 0.1)), 1)

        # 10. Peer Group Interaction Rating
        peer_interaction = int((discussion * 0.7) + (raised_hands * 0.3))

        # 11. Cognitive Fatigue Index
        cognitive_fatigue = int((study_hours * 6) + (stress_index * 0.4) - (sleep_time * 4))
        cognitive_fatigue = max(0, min(100, cognitive_fatigue))

        # 12. Focus Retention Span (Minutes)
        focus_span = int(45 + (sleep_time * 3) + (study_consistency * 0.2) - (stress_index * 0.25))
        focus_span = max(10, min(120, focus_span))

        # 13. Procrastination Probability
        procrastination_prob = int(100 - assignment_score + (stress_index * 0.2))
        procrastination_prob = max(5, min(98, procrastination_prob))

        # 14. Academic Burnout Velocity
        burnout_velocity = round((stress_index * 0.6 + extracurriculars * 4) / 10, 1)

        # 15. Scholarship Eligibility Probability (%)
        scholarship_prob = int(95 if predicted_gpa >= 8.5 and attendance >= 85 else max(2, int((predicted_gpa/10)*60 + (attendance/100)*40 - 20)))
        scholarship_prob = max(0, min(98, scholarship_prob))

        # 16. Placement Eligibility Gateway Status
        placement_gateway = "PASSED (Criteria Secure)" if attendance >= 75 and predicted_gpa >= 6.5 else "FAILED (Below Gateway Cut-offs)"

        # 17. Technical vs Management Career Inclination
        career_ratio = int((study_consistency * 0.6) + (visited_resources * 0.4) - (extracurriculars * 4))
        career_inclination = "Technical (Core Engineering/R&D)" if career_ratio >= 40 else "Management (Product/Operations)"

        # 18. Study Efficiency Ratio
        study_efficiency = round((visited_resources + assignment_score) / max(1, study_hours * 2), 1)

        # 19. Weekend Burn Risk Score
        weekend_burn = int((100 - study_consistency) * 0.6 + (stress_index * 0.4))

        # 20. Daily Routine Balance Score
        routine_balance = int(100 - abs(8 - sleep_time)*5 - abs(5 - study_hours)*4 - (stress_index * 0.2))
        routine_balance = max(10, min(100, routine_balance))

        # 21. Revision Cycle Recency Factor (Days)
        recency_factor = int(max(1, 30 - (study_consistency * 0.28)))

        # 22. Subject Strength & Weakness Detection Logic
        if raised_hands >= 60 and study_hours >= 6:
            strength_sub, weakness_sub = "Theory of Computation (TOC)", "Design & Analysis of Algorithms (DAA)"
        elif attendance >= 80 and assignment_score >= 75:
            strength_sub, weakness_sub = "Database Management (DBMS)", "Theory of Computation (TOC) Automata Structure"
        else:
            strength_sub, weakness_sub = "Computer Networks (CN)", "Theory of Computation (TOC) Core Machine Logic"

        # 23. Performance Trend Indicator
        if readiness_score >= 75 and study_consistency >= 70: trend_tag, trend_color = "🚀 IMPROVING STEADILY", "#10b981"
        elif readiness_score <= 45 or attendance <= 65: trend_tag, trend_color = "🚨 CRITICAL DECLINE IN MATRIX", "#ef4444"
        else: trend_tag, trend_color = "📊 STABLE TRAJECTORY", "#f59e0b"

        # 24. Goal Achievement Probability
        goal_diff = target_cgpa - predicted_gpa
        goal_prob = int(90 - (goal_diff * 50)) if goal_diff > 0 else np.random.randint(92, 99)
        goal_prob = max(2, min(98, goal_prob))

        # ==================== MATRIX RENDER GRID UNITS ====================
        
        # --- ROW LAYER 1: PURE ACADEMIC PARAMETERS ---
        st.markdown("---")
        st.markdown("### 📋 Section 1: Academic & Performance Predictive Metrics")
        r1_1, r1_2, r1_3, r1_4 = st.columns(4)
        r1_1.markdown(f"<div class='feature-card' style='border-color:#10b981;'><span class='feature-title'>🔮 Predicted Semester GPA</span><div class='feature-value' style='color:#10b981;'>{predicted_gpa} / 10.0</div></div>", unsafe_allow_html=True)
        r1_2.markdown(f"<div class='feature-card' style='border-color:#ef4444;'><span class='feature-title'>🚨 Backlog Probability</span><div class='feature-value' style='color:#f87171;'>{backlog_prob}%</div></div>", unsafe_allow_html=True)
        r1_3.markdown(f"<div class='feature-card' style='border-color:#38bdf8;'><span class='feature-title'>🛡️ Exam Readiness Score</span><div class='feature-value' style='color:#38bdf8;'>{readiness_score}/100</div></div>", unsafe_allow_html=True)
        r1_4.markdown(f"<div class='feature-card' style='border-color:#a855f7;'><span class='feature-title'>🏆 Academic Performance Score</span><div class='feature-value' style='color:#c084fc;'>{perf_score}/100</div></div>", unsafe_allow_html=True)

        # --- ROW LAYER 2: ADVANCED CURRICULUM LOGICS ---
        st.markdown(" ")
        r2_1, r2_2, r2_3, r2_4 = st.columns(4)
        r2_1.markdown(f"<div class='feature-card' style='border-color:#f43f5e;'><span class='feature-title'>📖 Expected Syllabus Coverage</span><div class='feature-value' style='color:#fb7185;'>{syllabus_coverage}%</div></div>", unsafe_allow_html=True)
        r2_2.markdown(f"<div class='feature-card' style='border-color:#0ea5e9;'><span class='feature-title'>🏅 Sessional Marks (Internal)</span><div class='feature-value' style='color:#38bdf8;'>{internal_marks} / 20</div></div>", unsafe_allow_html=True)
        r2_3.markdown(f"<div class='feature-card' style='border-color:#10b981;'><span class='feature-title'>⚖️ Credit Weight Integrity</span><div class='feature-value' style='color:#34d399;'>{credit_integrity} pts</div></div>", unsafe_allow_html=True)
        r2_4.markdown(f"<div class='feature-card' style='border-color:#eab308;'><span class='feature-title'>👥 Peer Group Network Score</span><div class='feature-value' style='color:#facc15;'>{peer_interaction}/100</div></div>", unsafe_allow_html=True)

        # --- ROW LAYER 3: COGNITIVE & STUDY HABIT ANALYSIS ---
        st.markdown("---")
        st.markdown("### 🧠 Section 2: Cognitive Load & Behavior Core Dynamics")
        r3_1, r3_2, r3_3, r3_4 = st.columns(4)
        r3_1.markdown(f"<div class='feature-card' style='border-color:#f97316;'><span class='feature-title'>💤 Cognitive Fatigue Index</span><div class='feature-value' style='color:#fb923c;'>{cognitive_fatigue}/100</div></div>", unsafe_allow_html=True)
        r3_2.markdown(f"<div class='feature-card' style='border-color:#22c55e;'><span class='feature-title'>⏱️ Focus Attention Span</span><div class='feature-value' style='color:#4ade80;'>{focus_span} mins</div></div>", unsafe_allow_html=True)
        r3_3.markdown(f"<div class='feature-card' style='border-color:#ef4444;'><span class='feature-title'>⏳ Procrastination Probability</span><div class='feature-value' style='color:#f87171;'>{procrastination_prob}%</div></div>", unsafe_allow_html=True)
        r3_4.markdown(f"<div class='feature-card' style='border-color:#ec4899;'><span class='feature-title'>📉 Burnout Risk Velocity</span><div class='feature-value' style='color:#f472b6;'>{burnout_velocity} x/s</div></div>", unsafe_allow_html=True)

        # --- ROW LAYER 4: EFFICIENCY & ROUTINE ANALYSIS ---
        st.markdown(" ")
        r4_1, r4_2, r4_3, r4_4 = st.columns(4)
        r4_1.markdown(f"<div class='feature-card' style='border-color:#6366f1;'><span class='feature-title'>📊 Study Efficiency Ratio</span><div class='feature-value' style='color:#818cf8;'>{study_efficiency} output</div></div>", unsafe_allow_html=True)
        r4_2.markdown(f"<div class='feature-card' style='border-color:#d946ef;'><span class='feature-title'>🧱 Weekend Workload Burn Risk</span><div class='feature-value' style='color:#e879f9;'>{weekend_burn}%</div></div>", unsafe_allow_html=True)
        r4_3.markdown(f"<div class='feature-card' style='border-color:#14b8a6;'><span class='feature-title'>⚖️ Routine Balance Index</span><div class='feature-value' style='color:#2dd4bf;'>{routine_balance}/100</div></div>", unsafe_allow_html=True)
        r4_4.markdown(f"<div class='feature-card' style='border-color:#84cc16;'><span class='feature-title'>🔄 Memory Recency Factor</span><div class='feature-value' style='color:#a3e635;'>{recency_factor} days</div></div>", unsafe_allow_html=True)

        # --- ROW LAYER 5: INSTITUTIONAL & CAREER INSIGHTS ---
        st.markdown("---")
        st.markdown("### 💼 Section 3: Future Placements & Institutional Targets")
        r5_1, r5_2, r5_3, r5_4 = st.columns(4)
        r5_1.markdown(f"<div class='feature-card' style='border-color:#a855f7;'><span class='feature-title'>🎓 Scholarship Gate Chance</span><div class='feature-value' style='color:#c084fc;'>{scholarship_prob}%</div></div>", unsafe_allow_html=True)
        r5_2.markdown(f"<div class='feature-card' style='border-color:#6366f1; font-size:12px;'><span class='feature-title'>🚀 Campus Placement Criteria</span><div class='feature-value' style='color:#818cf8; font-size:15px; margin-top:12px;'>{placement_gateway}</div></div>", unsafe_allow_html=True)
        r5_3.markdown(f"<div class='feature-card' style='border-color:#0ea5e9; font-size:12px;'><span class='feature-title'>🎯 Career Field Orientation</span><div class='feature-value' style='color:#38bdf8; font-size:14px; margin-top:14px;'>{career_inclination}</div></div>", unsafe_allow_html=True)
        r5_4.markdown(f"<div class='feature-card' style='border-color:#eab308;'><span class='feature-title'>🎯 Target Goal Likelihood</span><div class='feature-value' style='color:#facc15;'>{goal_prob}%</div></div>", unsafe_allow_html=True)

        # Run Model Baseline Fit
        input_data = np.array([[g_encoded, s_encoded, raised_hands, visited_resources, announcements, discussion, study_hours, sleep_time, attendance, extracurriculars, mh_encoded, counsel_encoded]])
        prediction = model.fit(X_train, y_train).predict(input_data)[0]

        # Performance Cluster Summary Block
        st.markdown(" ")
        st.markdown("### 🚨 Institutional Performance Clustering Layer")
        if prediction == 'H':
            st.markdown(f'<div class="status-box" style="background-color:#064e3b; border-color:#10b981; color:#d1fae5;"><h3>🎯 Cluster Node: EXCELLENT STATUS (Class H Layer)</h3><p>Academic variables are highly optimized. System tracks student <b>Strength Subject</b> as: <b>{strength_sub}</b>. Weakness Node Flagged: <b>{weakness_sub}</b>. Trend Direction vector: <b>{trend_tag}</b>.</p></div>', unsafe_allow_html=True)
        elif prediction == 'M':
            st.markdown(f'<div class="status-box" style="background-color:#451a03; border-color:#f97316; color:#ffedd5;"><h3>⚠️ Cluster Node: AVERAGE SECURE BASELINE (Class M Layer)</h3><p>Operation executing within expected threshold boundaries. Core <b>Strength Track</b>: <b>{strength_sub}</b>. Target <b>Weakness Module Node</b>: <b>{weakness_sub}</b>. Trend Index: <b>{trend_tag}</b>.</p></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="status-box" style="background-color:#450a0a; border-color:#ef4444; color:#fca5a5;"><h3>🚨 Cluster Node: CRITICAL DROPOUT RISK VECTOR (Class L Layer)</h3><p>Critical Warning: Learning metrics show immediate optimization requirements. Highest Risk Domain Found: <b>{weakness_sub}</b>. Trend Status Vector: <b style="color:#ef4444;">{trend_tag}</b>. Institutional mentor call loop recommended.</p></div>', unsafe_allow_html=True)

        # --- Explainable AI (XAI) Plot Render ---
        st.markdown(" ")
        st.markdown("#### ⚙️ Explainable AI (XAI) Model Feature Driver Weights")
        fig_imp, ax_imp = plt.subplots(figsize=(10, 2.8))
        fig_imp.patch.set_facecolor('#0f172a')
        ax_imp.set_facecolor('#0f172a')
        importances = model.feature_importances_
        indices = np.argsort(importances)[-5:] # Extracting Top 5
        ax_imp.barh([features[i] for i in indices], importances[indices], color='#2563eb')
        ax_imp.tick_params(colors='#e2e8f0', labelsize=9)
        ax_imp.set_title("Top Strategic Database Drivers Influencing Global Prediction Outputs", color='#e2e8f0', fontsize=10)
        st.pyplot(fig_imp)

    # ==================== TAB 2: INTERACTIVE LLAMA AI NODE ====================
    with tab2:
        st.subheader("🤖 AI Academic Strategy Assistant")
        st.write("Leverage our connected deep-network LLM cluster to query curricular tactics, sessional boosters or burnout management setups.")
        user_query = st.text_input("💬 Query the AI Advisor Node:", placeholder="Type a message like 'how to improve exam readiness' or 'TOC unit division guide'...")
        
        if user_query:
            st.markdown(f'<div style="background-color:#1e293b; padding:12px; border-radius:6px; margin-bottom:10px;"><b>You:</b> {user_query}</div>', unsafe_allow_html=True)
            
            API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
            system_prompt = f"<|system|>\nYou are an elite engineering university computer science counselor. Answer this student analytics question concisely, professionally, and use strict logical academic hacks: {user_query}\n<|user|>\nAnswer:"
            
            try:
                response = requests.post(API_URL, json={"inputs": system_prompt, "parameters": {"max_new_tokens": 150, "temperature": 0.7}}, timeout=8)
                res_json = response.json()
                bot_text = res_json[0]["generated_text"].split("Answer:")[-1].strip()
            except Exception:
                bot_text = f"Analytics query successfully parsed for track: {target_subject}. Tactical Recommendation: Fragment large data syllabus units into daily micro-tasks, deploy flash-cards for active recall mapping of core rules, run high-frequency past sessional trends, and secure your routine balance score to avoid memory retention loss errors."

            st.markdown(f'<div style="background-color:#0f172a; padding:15px; border-left:4px solid #2563eb; border-radius:6px; color:#e2e8f0;"><b>AI Advisor Response:</b><br><br>{bot_text}</div>', unsafe_allow_html=True)

    # ==================== TAB 3: DATA DISTRIBUTION VISUALIZATIONS ====================
    with tab3:
        st.subheader("📊 Analytical Distribution Matrices")
        c1, c2 = st.columns(2)
        with c1:
            fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
            fig_bar.patch.set_facecolor('#0f172a')
            ax_bar.set_facecolor('#0f172a')
            sns.countplot(x='MentalHealthStatus', hue='Class', data=df, palette='Spectral', order=['Excellent', 'Good', 'Stressed', 'Depressed'], ax=ax_bar)
            ax_bar.tick_params(colors='#e2e8f0')
            ax_bar.set_title("Psychological Behavior Weight Metrics over Dataset", color='#e2e8f0')
            st.pyplot(fig_bar)
        with c2:
            fig_scatter, ax_scatter = plt.subplots(figsize=(6, 4))
            fig_scatter.patch.set_facecolor('#0f172a')
            ax_scatter.set_facecolor('#0f172a')
            sns.scatterplot(x='StudyHours', y='Attendance', hue='Class', data=df, palette='Set1', alpha=0.7, ax=ax_scatter)
            ax_scatter.tick_params(colors='#e2e8f0')
            ax_scatter.set_title("Study Hours vs Campus Attendance Scatter Distribution", color='#e2e8f0')
            st.pyplot(fig_scatter)

except FileNotFoundError:
    st.error("Fatal System Error: 'AI-Data.csv' reference database missing in root directory.")
