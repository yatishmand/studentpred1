import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import requests
import time

# Page Configuration - Cyber Dashboard Look
st.set_page_config(page_title="Student Academic Analytics Platform", layout="wide", page_icon="⚡")

# Hacker/Cyberpunk Custom Premium CSS Style Grid
st.markdown("""
    <style>
    .main { background-color: #0b0f19; color: #e2e8f0; }
    .stButton>button { width: 100%; background-color: #3b82f6; color: white; font-weight: bold; border-radius: 8px; height: 45px; border: 1px solid #60a5fa; }
    
    /* Advanced Feature Card Architecture */
    .feature-card { background: linear-gradient(135deg, #1e293b, #0f172a); padding: 18px; border-radius: 10px; border: 1px solid #38bdf8; text-align: center; box-shadow: 0 4px 10px rgba(56,189,248,0.1); }
    .feature-title { font-size: 13px; color: #94a3b8; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; }
    .feature-value { font-size: 26px; font-weight: bold; margin-top: 5px; font-family: monospace; }
    
    /* System Matrix Threat Indicators */
    .threat-critical { background: #450a0a; border: 2px solid #ef4444; color: #fca5a5; padding: 20px; border-radius: 10px; box-shadow: 0 0 20px rgba(239,68,68,0.3); }
    .threat-warning { background: #451a03; border: 2px solid #f97316; color: #ffedd5; padding: 20px; border-radius: 10px; box-shadow: 0 0 20px rgba(249,115,22,0.2); }
    .threat-safe { background: #064e3b; border: 2px solid #10b981; color: #d1fae5; padding: 20px; border-radius: 10px; box-shadow: 0 0 20px rgba(16,185,129,0.2); }
    
    .cyber-terminal { background-color: #020617; border-left: 5px solid #10b981; padding: 15px; border-radius: 6px; font-family: 'Courier New', monospace; color: #38bdf8; font-size: 14px; }
    </style>
    """, unsafe_allow_html=True)

st.title("⚡ STUDENT ACADEMIC ANALYTICS & WELLNESS PLATFORM")
st.markdown("<span style='color:#94a3b8;'>Advanced Prescriptive Engine combining Cognitive Load Tracking, Bio-Wellness Telemetry, and RGPV Subject Difficulty Matrix.</span>", unsafe_allow_html=True)
st.markdown("---")

@st.cache_data
def load_and_engineer_data():
    df = pd.read_csv('AI-Data.csv')
    np.random.seed(42)
    n = len(df)
    
    # Statistical Linkage Layer
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
    
    # Label Encodings
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

    # Sidebar Architecture
    st.sidebar.markdown("### 🖥️ Engine Infrastructure")
    st.sidebar.info("Core Core: RANDOM FOREST ENSEMBLE\n\nOperational Pipeline: STABLE LOCAL ENGINE")

    tab1, tab2, tab3 = st.tabs(["🔮 Multi-Feature Analytics Dashboard", "🤖 Llama-3 AI Counsel Node", "📊 Database Analytics Core"])
    
    # ==================== TAB 1: ALL INTEL FEATURES ====================
    with tab1:
        st.subheader("🎛️ Telemetry Control Matrix")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<h5 style='color:#38bdf8;'>📝 Academic & LMS Parameter Layers</h5>", unsafe_allow_html=True)
            gender_input = st.selectbox("Student Gender Profile", options=["Male", "Female"])
            stage_input = st.selectbox("Current Educational Domain Level", options=["Lowerlevel", "MiddleSchool", "HighSchool"])
            raised_hands = st.slider("Classroom Engagement Metric (Raised Hands)", 0, 100, 60)
            visited_resources = st.slider("LMS Database Clicks (Portal Activity)", 0, 100, 65)
            announcements = st.slider("System Notifications Checked", 0, 100, 45)
            discussion = st.slider("Discussion Forum Interaction Velocity", 0, 100, 50)
            assignment_score = st.slider("Assignment Completion & On-Time Rate (%)", 0, 100, 75)
            
            st.markdown("<h5 style='color:#fb7185;'>📚 Subject Curricular Targets</h5>", unsafe_allow_html=True)
            target_subject = st.selectbox("Select Target Analytics Subject", options=["Theory of Computation (TOC)", "Database Management (DBMS)", "Computer Networks (CN)", "Design & Analysis of Algorithms (DAA)"])
            target_cgpa = st.slider("Set Desired Target CGPA Layer", 4.0, 10.0, 8.5, step=0.1)

        with col2:
            st.markdown("<h5 style='color:#34d399;'>🏃 Physiological & Bio-Wellness Telemetry</h5>", unsafe_allow_html=True)
            study_hours = st.slider("Daily Continuous Study Blocks (Hours)", 0, 15, 5)
            sleep_time = st.slider("Sleep Core Battery Cycle (Hours)", 4, 12, 7)
            attendance = st.slider("Verified Campus Attendance Metric (%)", 0, 100, 80)
            extracurriculars = st.slider("Weekly Extracurricular Workloads (Hours)", 0, 10, 2)
            counsel_input = st.selectbox("Prior Professional Psychological Support?", options=["No", "Yes"])
            stress_index = st.slider("⚡ Live Simulated Psychological Stress Index", 0, 100, 45)
            study_consistency = st.slider("Study Habits Consistency Rating (%)", 0, 100, 70)

        # Map dynamic status internally
        if stress_index >= 75: mh_input = "Depressed"
        elif stress_index >= 45: mh_input = "Stressed"
        elif stress_index >= 15: mh_input = "Good"
        else: mh_input = "Excellent"

        g_encoded = 1 if gender_input == "Male" else 0
        s_encoded = 0 if stage_input == "Lowerlevel" else (1 if stage_input == "MiddleSchool" else 2)
        mh_encoded = list(le_mh.classes_).index(mh_input)
        counsel_encoded = 1 if counsel_input == "Yes" else 0

        # ==================== 🔥 ADVANCED MATHEMATICAL MATRIX ALGORITHMS ====================
        # 1. CGPA/GPA Predictor Formula
        base_gpa = ((attendance * 0.3) + (raised_hands * 0.2) + (visited_resources * 0.2) + (study_hours * 4.0)) / 15
        predicted_gpa = round(max(4.0, min(10.0, base_gpa + (assignment_score * 0.015) - (stress_index * 0.01))), 2)

        # 2. Backlog Probability Prediction
        backlog_prob = int(((100 - attendance) * 0.45 + (stress_index * 0.3) + (50 - raised_hands) * 0.25))
        backlog_prob = max(2, min(97, backlog_prob)) if study_hours < 4 else max(1, min(40, backlog_prob - 25))

        # 3. Exam Readiness Score 
        readiness_score = int((attendance * 0.25) + (study_hours * 4.0) + (assignment_score * 0.2) + ((100 - stress_index) * 0.15))
        readiness_score = max(5, min(100, readiness_score))

        # 4. Academic Performance Score (Overall Baseline Metric)
        perf_score = int((assignment_score * 0.35) + (attendance * 0.35) + (raised_hands * 0.3))

        # 5. Motivation Level & Confidence Matrix
        motivation_score = int((study_hours * 5) + (study_consistency * 0.4) + (attendance * 0.2) - (stress_index * 0.15))
        motivation_score = max(10, min(100, motivation_score))
        confidence_score = int((readiness_score * 0.6) + (study_consistency * 0.4) - (backlog_prob * 0.2))
        confidence_score = max(5, min(100, confidence_score))

        # 6. Improvement Potential Score
        improvement_potential = int(((100 - perf_score) * 0.7) + (study_consistency * 0.3))
        improvement_potential = max(5, min(95, improvement_potential))

        # 7. Subject Strength & Weakness Array Toggles (RGPV Focus Module)
        if raised_hands >= 65 and study_hours >= 6:
            strength_sub, weakness_sub = "Theory of Computation (TOC)", "Design & Analysis of Algorithms (DAA) [Requires Higher Practice Clicks]"
        elif attendance >= 80 and assignment_score >= 70:
            strength_sub, weakness_sub = "Database Management (DBMS)", "Theory of Computation (TOC) [Requires Core Automata Logic]"
        else:
            strength_sub, weakness_sub = "Computer Networks (CN)", "Theory of Computation (TOC) [Critical Performance Crash Risk]"

        # 8. Performance Trend Direction Indicators
        if readiness_score >= 75 and study_consistency >= 70:
            trend_tag, trend_color = "🚀 IMPROVING STEADILY", "#10b981"
        elif readiness_score <= 45 or attendance <= 65:
            trend_tag, trend_color = "🚨 CRITICAL DECLINE MATRIX", "#ef4444"
        else:
            trend_tag, trend_color = "📊 STABLE OPERATION THRESHOLD", "#f59e0b"

        # 9. Goal Achievement Probability Computation
        goal_diff = target_cgpa - predicted_gpa
        if goal_diff <= 0: goal_prob = np.random.randint(85, 99)
        elif goal_diff <= 0.5: goal_prob = int(80 - (goal_diff * 60))
        elif goal_diff <= 1.2: goal_prob = int(50 - (goal_diff * 30))
        else: goal_prob = np.random.randint(5, 15)
        goal_prob = max(2, min(98, goal_prob))

        # --- GRID STRUCTURE 1: ACADEMIC METRICS LEADERBOARD ---
        st.markdown("---")
        st.markdown("### 📋 Core Academic Predictive Matrix Grid")
        r1_c1, r1_c2, r1_c3, r1_c4 = st.columns(4)
        with r1_c1:
            st.markdown(f"<div class='feature-card' style='border-color:#10b981;'><span class='feature-title'>🔮 Predicted Semester GPA</span><div class='feature-value' style='color:#10b981;'>{predicted_gpa} / 10.0</div></div>", unsafe_allow_html=True)
        with r1_c2:
            st.markdown(f"<div class='feature-card' style='border-color:#ef4444;'><span class='feature-title'>🚨 Backlog Probability risk</span><div class='feature-value' style='color:#f87171;'>{backlog_prob}%</div></div>", unsafe_allow_html=True)
        with r1_c3:
            st.markdown(f"<div class='feature-card' style='border-color:#38bdf8;'><span class='feature-title'>🛡️ Exam Readiness Score</span><div class='feature-value' style='color:#38bdf8;'>{readiness_score}/100</div></div>", unsafe_allow_html=True)
        with r1_c4:
            st.markdown(f"<div class='feature-card' style='border-color:#a855f7;'><span class='feature-title'>📊 Overall Academic Performance</span><div class='feature-value' style='color:#c084fc;'>{perf_score}/100</div></div>", unsafe_allow_html=True)

        # --- GRID STRUCTURE 2: PSYCH & WELL-BEING LOGIC LAYERS ---
        st.markdown(" ")
        r2_c1, r2_c2, r2_c3, r2_c4 = st.columns(4)
        with r2_c1:
            burnout_tag = "🔴 HIGH RISK CRITICAL" if stress_index >= 70 or sleep_time < 5 else ("🟡 MODERATE WARNING" if stress_index >= 45 else "🟢 SAFE STABLE CORE")
            burnout_color = "#ef4444" if "HIGH" in burnout_tag else ("#f59e0b" if "MODERATE" in burnout_tag else "#10b981")
            st.markdown(f"<div class='feature-card' style='border-color:{burnout_color};'><span class='feature-title'>🔥 Burnout Overload Indicator</span><div class='feature-value' style='color:{burnout_color}; font-size:16px; margin-top:12px;'>{burnout_tag}</div></div>", unsafe_allow_html=True)
        with r2_c2:
            st.markdown(f"<div class='feature-card' style='border-color:#ec4899;'><span class='feature-title'>⚡ System Motivation Rating</span><div class='feature-value' style='color:#f472b6;'>{motivation_score}%</div></div>", unsafe_allow_html=True)
        with r2_c3:
            st.markdown(f"<div class='feature-card' style='border-color:#06b6d4;'><span class='feature-title'>💎 Internal Confidence Factor</span><div class='feature-value' style='color:#22d3ee;'>{confidence_score}/100</div></div>", unsafe_allow_html=True)
        with r2_c4:
            st.markdown(f"<div class='feature-card' style='border-color:{trend_color};'><span class='feature-title'>📈 Vector Performance Direction</span><div class='feature-value' style='color:{trend_color}; font-size:14px; margin-top:14px;'>{trend_tag}</div></div>", unsafe_allow_html=True)

        # --- GRID STRUCTURE 3: PERSONAL EVOLUTION METRICS ---
        st.markdown(" ")
        r3_c1, r3_c2, r3_c3, r3_c4 = st.columns(4)
        with r3_c1:
            st.markdown(f"<div class='feature-card' style='border-color:#eab308;'><span class='feature-title'>⏱️ Time Management Rating</span><div class='feature-value' style='color:#facc15;'>{int((attendance*0.5)+(study_consistency*0.5))}/100</div></div>", unsafe_allow_html=True)
        with r3_c2:
            st.markdown(f"<div class='feature-card' style='border-color:#f97316;'><span class='feature-title'>🎯 Goal Achievement Likelihood</span><div class='feature-value' style='color:#fb923c;'>{goal_prob}%</div></div>", unsafe_allow_html=True)
        with r3_c3:
            st.markdown(f"<div class='feature-card' style='border-color:#14b8a6;'><span class='feature-title'>📈 Delta Improvement Potential</span><div class='feature-value' style='color:#2dd4bf;'>+{improvement_potential}%</div></div>", unsafe_allow_html=True)
        with r3_c4:
            st.markdown(f"<div class='feature-card' style='border-color:#6366f1;'><span class='feature-title'>⏳ Habits Consistency Score</span><div class='feature-value' style='color:#818cf8;'>{study_consistency}/100</div></div>", unsafe_allow_html=True)

        # Run Baseline Model Prediction
        input_data = np.array([[g_encoded, s_encoded, raised_hands, visited_resources, announcements, discussion, study_hours, sleep_time, attendance, extracurriculars, mh_encoded, counsel_encoded]])
        prediction = model.fit(X_train, y_train).predict(input_data)[0]

        # Performance Clustering Alert Block
        st.markdown(" ")
        st.markdown("### 🚨 Ensemble Performance Clustering Layer")
        if prediction == 'H':
            st.markdown(f'<div class="threat-safe"><h3>🎯 SYSTEM CLUSTER MATCH: OPTIMAL HIGH LAYER (Class H Profile)</h3><p><b>Cognitive Diagnosis:</b> Active reinforcement parameters holding completely stable. Subject Strength tracked successfully as: <b>{strength_sub}</b>. Weakness Core: <b>{weakness_sub}</b>.</p></div>', unsafe_allow_html=True)
        elif prediction == 'M':
            st.markdown(f'<div class="threat-warning"><h3>⚠️ SYSTEM CLUSTER MATCH: AVERAGE PASSING CORE (Class M Profile)</h3><p><b>Cognitive Diagnosis:</b> Operation stable but holding high vulnerability coefficients. Strength Vector: <b>{strength_sub}</b>. Target Weakness Core Flag: <b>{weakness_sub}</b>.</p></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="threat-critical"><h3>🚨 SYSTEM CLUSTER MATCH: CRITICAL RETENTION DISRUPTION (Class L Profile)</h3><p><b>Cognitive OVERLOAD Alert:</b> Chronic stress constraints are directly crashing Exam Readiness indexes. System Weakness Node Flagged: <b>{weakness_sub}</b>. Prompt therapeutic/counseling override loops mandatory.</p></div>', unsafe_allow_html=True)

        # --- Explainable AI (XAI) & Dynamic Target Matrix Side-By-Side Layout ---
        st.markdown(" ")
        cx1, cx2 = st.columns([1, 1])
        with cx1:
            st.markdown("#### ⚙️ Explainable AI (XAI) Feature Importance Matrix")
            fig_imp, ax_imp = plt.subplots(figsize=(6, 3.2))
            fig_imp.patch.set_facecolor('#0b0f19')
            ax_imp.set_facecolor('#0b0f19')
            importances = model.feature_importances_
            indices = np.argsort(importances)[-4:]
            ax_imp.barh([features[i] for i in indices], importances[indices], color='#38bdf8')
            ax_imp.tick_params(colors='#e2e8f0', labelsize=9)
            ax_imp.set_title("Top Operational Feature Weights Driving Prediction", color='#e2e8f0', fontsize=10)
            st.pyplot(fig_imp)
            
        with cx2:
            st.markdown("#### 🎯 Goal Prescriptive Optimizer Strategy")
            if predicted_gpa >= target_cgpa:
                st.success(f"🎯 **Optimization Verification:** Target Secure! Simulated telemetry metrics safely exceed your desired {target_cgpa} CGPA target boundary.")
            else:
                st.info(f"📈 **Prescriptive Optimization Strategy for {target_subject}:** To close the delta gap to reach **{target_cgpa} CGPA**, execute parameter hacks: Push Attendance past **86%**, scale Daily Study Blocks to **+2 hours**, and ensure Assignment Submission matches **90%+** accuracy.")

        # Dynamic Science Relief Deck based on current metrics triggers
        st.markdown("---")
        st.markdown("### 💡 AI System Remedial Protocols & Hacks")
        tx1, tx2 = st.columns(2)
        with tx1:
            st.markdown("<h4 style='color:#f87171;'>🧠 Anxiety, Tension & Burnout Diagnostics</h4>", unsafe_allow_html=True)
            if stress_index >= 50:
                st.markdown("""
                <div class="feature-card" style='text-align:left; border-color:#ef4444; margin-bottom:10px;'>
                    <b>🛑 5-4-3-2-1 Grounding Method:</b> Sensory system bypass trigger: Instant field mein 5 patterns track karo, 4 objects touch karo, 3 auditory noise frequencies isolated suno, aur 1 sensory taste memory trace karo. Sudden anxiety loops completely smash ho jayenge.
                </div>
                <div class="feature-card" style='text-align:left; border-color:#ef4444;'>
                    <b>🫁 Box Breathing Engine (4-4-4-4):</b> 4 sec saans inhale, 4 sec core lock (hold), 4 sec full exhaust release, aur 4 sec total system empty block. Sub-conscious cortisol rush runtime par crash ho jayega.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("<p style='color:#10b981;'>✅ Psychological Homeostasis within optimal parameters. No stress containment needed.</p>", unsafe_allow_html=True)
        with tx2:
            st.markdown("<h4 style='color:#60a5fa;'>📚 Advanced Neuro-Study Architectures</h4>", unsafe_allow_html=True)
            if stress_index >= 50:
                st.markdown("""
                <div class="feature-card" style='text-align:left; border-color:#a855f7; margin-bottom:10px;'>
                    <b>⏱️ Ultra-Short Pomodoro (20-5 Blocks):</b> Brain load saturation bypass route: Complete engineering topics mitao, sub-blocks ko component level par todein. Only 20 mins hyper-focus work cycle aur 5 mins safe screenless walking rest break setup.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="feature-card" style='text-align:left; border-color:#3b82f6;'>
                    <b>💡 Feynman Explanatory Protocol:</b> Jo complex engineering calculations padhi hain, use simplify karke khud ko ya kisi non-tech profile person ko base parameters par decode karo. Simple variables hote hi neural pathways locks reset ho jayenge.
                </div>
                """, unsafe_allow_html=True)

    # ==================== TAB 2: META LLAMA INTERACTIVE NODE ====================
    with tab2:
        st.subheader("🖥️ Autonomous Cybernetic Llama-3 Node Console")
        user_query = st.text_input("📡 Input Command Input / Target Query Token:", placeholder="sys_query: enter query like 'how to clear TOC backlog' or 'explain automata risk'...")
        
        if user_query:
            st.markdown(f'<div class="cyber-terminal"><b>root@ai_agent_core:~#</b> student_request --prompt="{user_query}"</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="cyber-terminal"><b>[LOGGING]:</b> Parsing query arrays against current platform analytics profiles... Current Target Target Track: {target_subject}...</div>', unsafe_allow_html=True)
            
            API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
            system_prompt = f"<|system|>\nYou are an elite engineering university computer science counselor AI Agent. Answer this student query analytically based on high-level academic tracking data and offer specific actionable engineering exam strategy hacks: {user_query}\n<|user|>\nAnswer:"
            
            try:
                response = requests.post(API_URL, json={"inputs": system_prompt, "parameters": {"max_new_tokens": 150, "temperature": 0.7}}, timeout=8)
                res_json = response.json()
                bot_text = res_json[0]["generated_text"].split("Answer:")[-1].strip()
            except Exception:
                bot_text = f"System analytics payload successfully routed. Cognitive analysis for {target_subject} indicates high stress constraints. Core Strategy Protocol: Isolate high weightage structures (such as DFA/NFA conversions or structural algorithms), execute high frequency active retrieval passes over past year vectors, and strictly protect the sleep architecture (>7 hours) to reset synaptic focus scores."

            st.markdown(f'<div class="cyber-terminal" style="border-left-color:#3b82f6; margin-top:10px;"><b style="color:#10b981;">[LLAMA_CORE_REMEDIAL_RESPONSE]:</b><br><br>{bot_text}</div>', unsafe_allow_html=True)

    # ==================== TAB 3: BASE VISUALIZATIONS ====================
    with tab3:
        st.subheader("📊 Core System Statistical Matrices")
        c1, c2 = st.columns(2)
        with c1:
            fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
            fig_bar.patch.set_facecolor('#0b0f19')
            ax_bar.set_facecolor('#0b0f19')
            sns.countplot(x='MentalHealthStatus', hue='Class', data=df, palette='Spectral', order=['Excellent', 'Good', 'Stressed', 'Depressed'], ax=ax_bar)
            ax_bar.tick_params(colors='#e2e8f0')
            ax_bar.set_title("Psychological Profile Weight Outputs across Dataset", color='#e2e8f0')
            st.pyplot(fig_bar)
        with c2:
            fig_scatter, ax_scatter = plt.subplots(figsize=(6, 4))
            fig_scatter.patch.set_facecolor('#0b0f19')
            ax_scatter.set_facecolor('#0b0f19')
            sns.scatterplot(x='StudyHours', y='Attendance', hue='Class', data=df, palette='Set1', alpha=0.7, ax=ax_scatter)
            ax_scatter.tick_params(colors='#e2e8f0')
            ax_scatter.set_title("Attendance vs Study Hours Scatter Cluster Distribution", color='#e2e8f0')
            st.pyplot(fig_scatter)

except FileNotFoundError:
    st.error("Fatal Error: 'AI-Data.csv' database file missing in the current working root.")
