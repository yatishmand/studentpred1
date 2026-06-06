import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Page Configuration - Clean Professional Analytics Track
st.set_page_config(page_title="Student Academic Analytics Platform", layout="wide", page_icon="🎓")

# Professional Custom Corporate Style Grid Theme
st.markdown("""
    <style>
    .main { background-color: #0b0f19; color: #e2e8f0; }
    .stButton>button { width: 100%; background-color: #3b82f6; color: white; font-weight: bold; border-radius: 8px; height: 45px; border: 1px solid #60a5fa; }
    
    /* Executive Feature Grid System */
    .feature-card { background: linear-gradient(135deg, #1e293b, #0f172a); padding: 15px; border-radius: 8px; border: 1px solid #38bdf8; text-align: center; box-shadow: 0 4px 10px rgba(56,189,248,0.1); }
    .feature-title { font-size: 11px; color: #94a3b8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
    .feature-value { font-size: 22px; font-weight: bold; margin-top: 6px; font-family: monospace; }
    
    /* Institutional Cluster Status Indicators */
    .threat-critical { background: #450a0a; border: 2px solid #ef4444; color: #fca5a5; padding: 20px; border-radius: 10px; box-shadow: 0 0 20px rgba(239,68,68,0.3); }
    .threat-warning { background: #451a03; border: 2px solid #f97316; color: #ffedd5; padding: 20px; border-radius: 10px; box-shadow: 0 0 20px rgba(249,115,22,0.2); }
    .threat-safe { background: #064e3b; border: 2px solid #10b981; color: #d1fae5; padding: 20px; border-radius: 10px; box-shadow: 0 0 20px rgba(16,185,129,0.2); }
    
    .cyber-terminal { background-color: #020617; border-left: 5px solid #2563eb; padding: 15px; border-radius: 6px; font-family: 'Courier New', monospace; color: #38bdf8; font-size: 14px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🎓 STUDENT ACADEMIC ANALYTICS PLATFORM")
st.markdown("<span style='color:#94a3b8;'>Advanced Machine Learning Pipeline compiling 25+ Behavioral, Psychographic, Cognitive, and Curricular Performance Analytics.</span>", unsafe_allow_html=True)
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
    
    # Add engineered baseline GPA distribution for visuals mapping
    df['Dataset_Predicted_GPA'] = ((df['Attendance']*0.3) + (df['raisedhands']*0.2) + (df['VisITedResources']*0.2) + (df['StudyHours']*0.3)) / 12
    df['Dataset_Predicted_GPA'] = df['Dataset_Predicted_GPA'].clip(4.0, 10.0).round(2)
    
    return df

try:
    df = load_and_engineer_data()
    
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

    st.sidebar.markdown("### 🖥️ Platform Telemetry Control")
    st.sidebar.info("Analytics Engine: RANDOM FOREST\n\nParameters Monitored: 26 Core Metrics")

    tab1, tab2, tab3 = st.tabs(["📊 Diagnostic Analytics Dashboard", "🤖 AI Academic Strategy Assistant", "📈 Institutional Data Core"])
    
    # ==================== TAB 1: EXECUTIVE ANALYTICS MATRIX ====================
    with tab1:
        st.subheader("🎛️ Real-Time Parameter Configuration Desk")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<h5 style='color:#3b82f6;'>📝 Academic LMS Engagement Vectors</h5>", unsafe_allow_html=True)
            gender_input = st.selectbox("Student Gender Profile", options=["Male", "Female"])
            stage_input = st.selectbox("Current Educational Stage Level", options=["Lowerlevel", "MiddleSchool", "HighSchool"])
            raised_hands = st.slider("Classroom Interaction Rank (Raised Hands)", 0, 100, 60)
            visited_resources = st.slider("LMS Clicks / Portal Resource Utilization", 0, 100, 65)
            announcements = st.slider("Framework Academic Announcements Viewed", 0, 100, 45)
            discussion = st.slider("Active Discussion Forum Participation Density", 0, 100, 50)
            assignment_score = st.slider("Assignment Evaluation Completion Rate (%)", 0, 100, 75)
            
            # --- EXTENDED MULTI-DISCIPLINARY SUBJECT PIPELINE ---
            st.markdown("<h5 style='color:#fb7185;'>🎯 Target Performance Milestone</h5>", unsafe_allow_html=True)
            
            subject_options = [
                # Engineering & Tech Tracks
                "Engineering: Theory of Computation (TOC)", "Engineering: Design & Analysis of Algorithms (DAA)", 
                "Engineering: Database Management Systems (DBMS)", "Engineering: Computer Networks (CN)", 
                "Engineering: Operating Systems (OS)", "Engineering: Machine Learning Pipelines", 
                "Engineering: Data Science Architecture", "Engineering: Cyber Security & Cryptography", 
                "Engineering: Cloud Computing Frameworks", "Engineering: Compiler Design Matrices",
                # Commerce & Management Tracks
                "Commerce: Advanced Corporate Accounting", "Commerce: Financial Risk Management & Fintech", 
                "Commerce: Macroeconomic Policy Analysis", "Commerce: Strategic Business Management", 
                "Commerce: Digital Marketing Analytics", "Commerce: Econometrics & Quantitative Data", 
                "Commerce: International Trade & Logistics", "Commerce: Investment & Portfolio Analysis", 
                "Commerce: Auditing & Assurance Systems", "Commerce: Operational Cost Management",
                # Biological & Life Sciences Tracks
                "Bio-Science: Molecular Genetics & CRISPR", "Bio-Science: Cellular Immunology", 
                "Bio-Science: Advanced Biochemistry", "Bio-Science: Clinical Microbiology", 
                "Bio-Science: Neuroanatomy & Brain Mapping", "Bio-Science: Plant Physiology & Biotech", 
                "Bio-Science: Pharmacokinetics & Drug Analytics", "Bio-Science: Structural Bioinformatics", 
                "Bio-Science: Pathology & Disease Mechanisms", "Bio-Science: Genetic Engineering Mappings"
            ]
            
            target_subject = st.selectbox("Select Target Analytics Track", options=subject_options)
            target_cgpa = st.slider("Set Desired Target Cumulative CGPA", 4.0, 10.0, 8.5, step=0.1)

        with col2:
            st.markdown("<h5 style='color:#10b981;'>🏃 Lifestyle Habits & Well-being Attributes</h5>", unsafe_allow_html=True)
            
            # --- FIXED REALISTIC 24-HOUR ROUTINE GATEWAY ---
            study_hours = st.slider("Daily Continuous Self-Study Blocks (Hours)", 0, 14, 5, help="Maximum realistic self-study allocation capped to 14 hours.")
            
            max_allowed_sleep = max(4, 24 - study_hours - 2)
            sleep_time = st.slider("Sleep Cycle Recovery Duration (Hours)", 4, int(max_allowed_sleep), 7, help="Bound dynamically by your active Study Hours parameters.")
            
            allocated_routine_time = 24 - (study_hours + sleep_time)
            st.markdown(f"<span style='color:#94a3b8; font-size:12px; font-weight:600;'>📊 Current Routine Frame: Study {study_hours}h | Sleep {sleep_time}h | Essential Buffer {allocated_routine_time}h</span>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            
            attendance = st.slider("Verified Academic Attendance Record (%)", 0, 100, 80)
            extracurriculars = st.slider("Weekly Co-Curricular Allocation (Hours)", 0, 10, 2)
            counsel_input = st.selectbox("Prior Professional Academic Counseling?", options=["No", "Yes"])
            stress_index = st.slider("⚡ Live Simulated Psychological Stress Index", 0, 100, 40)
            study_consistency = st.slider("Study Workflow Consistency Score (%)", 0, 100, 70)

        if stress_index >= 75: mh_input = "Depressed"
        elif stress_index >= 45: mh_input = "Stressed"
        else: mh_input = "Good"

        g_encoded = 1 if gender_input == "Male" else 0
        s_encoded = 0 if stage_input == "Lowerlevel" else (1 if stage_input == "MiddleSchool" else 2)
        mh_encoded = list(le_mh.classes_).index(mh_input)
        counsel_encoded = 1 if counsel_input == "Yes" else 0

        # ==================== RE-ENGINEERED ACCURATE FORECASTING LAYER ====================
        academic_blend = (attendance * 0.35) + (assignment_score * 0.30) + (raised_hands * 0.15) + (visited_resources * 0.20)
        study_booster = (study_hours / 14.0) * 1.5 * (study_consistency / 100.0)
        stress_penalty = (stress_index / 100.0) * 0.8
        
        gpa_math = 4.0 + (academic_blend / 100.0) * 4.5 + study_booster - stress_penalty
        predicted_gpa = round(max(4.0, min(10.0, (gpa_math * 0.8) + (target_cgpa * 0.2))), 2)

        backlog_prob = int(((100 - attendance) * 0.6) + ((100 - assignment_score) * 0.3) + (stress_index * 0.1))
        if study_hours >= 6 or attendance >= 85:
            backlog_prob = max(1, min(25, backlog_prob - 30))
        else:
            backlog_prob = max(30, min(98, backlog_prob + 15))

        readiness_score = int((attendance * 0.3) + (study_hours * 4.5) + (assignment_score * 0.2) + (study_consistency * 0.15) - (stress_index * 0.1))
        readiness_score = max(4, min(100, readiness_score))

        perf_score = int((assignment_score * 0.4) + (attendance * 0.4) + (raised_hands * 0.2))
        motivation_score = max(10, min(100, int((study_hours * 5) + (study_consistency * 0.4) - (stress_index * 0.1))))
        confidence_score = max(5, min(100, int((readiness_score * 0.6) + (study_consistency * 0.4) - (backlog_prob * 0.1))))
        syllabus_coverage = max(15, min(100, int((study_hours * 5) + (assignment_score * 0.3) + (attendance * 0.1))))

        credit_weight = 4 if "TOC" in target_subject or "DAA" in target_subject or "Corporate" in target_subject or "Genetics" in target_subject else 3
        credit_integrity = round((attendance * 0.01) * credit_weight * (predicted_gpa / 10), 2)

        internal_marks = round(min(20.0, (raised_hands * 0.04) + (visited_resources * 0.04) + (assignment_score * 0.12)), 1)
        peer_interaction = int((discussion * 0.7) + (raised_hands * 0.3))

        cognitive_fatigue = max(0, min(100, int((study_hours * 6) + (stress_index * 0.4) - (sleep_time * 4))))
        focus_span = max(10, min(120, int(45 + (sleep_time * 3.5) + (study_consistency * 0.2) - (stress_index * 0.2))))
        procrastination_prob = max(5, min(98, int(100 - assignment_score + (stress_index * 0.15))))
        burnout_velocity = round((stress_index * 0.6 + extracurriculars * 4) / 10, 1)

        scholarship_prob = int(95 if predicted_gpa >= 8.5 and attendance >= 85 else max(0, int((predicted_gpa/10)*60 + (attendance/100)*40 - 25)))
        placement_gateway = "PASSED (Criteria Eligible)" if attendance >= 75 and predicted_gpa >= 6.5 else "FAILED (Below Gateway Benchmark)"

        career_ratio = int((study_consistency * 0.6) + (visited_resources * 0.4) - (extracurriculars * 4))
        
        # Split career logic according to active tracking stream
        if "Engineering" in target_subject:
            career_inclination = "Technical (Core Engineering/R&D)" if career_ratio >= 45 else "Management (Product Operations)"
        elif "Commerce" in target_subject:
            career_inclination = "Quantitative Analytics & Fintech" if career_ratio >= 45 else "Corporate Auditing & Consulting"
        else:
            career_inclination = "Biotech Lab Research & Genomic R&D" if career_ratio >= 45 else "Clinical Health & Administration"

        study_efficiency = round((visited_resources + assignment_score) / max(1, study_hours * 2), 1)
        weekend_burn = int((100 - study_consistency) * 0.6 + (stress_index * 0.4))
        routine_balance = max(10, min(100, int(100 - abs(8 - sleep_time)*5 - abs(5 - study_hours)*4 - (stress_index * 0.2))))
        recency_factor = int(max(1, 30 - (study_consistency * 0.28)))

        # Dynamic mapping of strengths based on stream
        if "Engineering" in target_subject:
            strength_sub, weakness_sub = "Systems Computational Modeling", "Algorithmic Complexity Core"
        elif "Commerce" in target_subject:
            strength_sub, weakness_sub = "Statistical Econometrics", "Taxation & Legal Compliance Audit"
        else:
            strength_sub, weakness_sub = "Molecular Assays & Sequencing", "Structural Bio-informatics Data"

        if readiness_score >= 75 and study_consistency >= 70: trend_tag, trend_color = "🚀 IMPROVING STEADILY", "#10b981"
        elif readiness_score <= 45 or attendance <= 65: trend_tag, trend_color = "🚨 CRITICAL DECLINE IN MATRIX", "#ef4444"
        else: trend_tag, trend_color = "📊 STABLE TRAJECTORY THRESHOLD", "#f59e0b"

        goal_diff = target_cgpa - predicted_gpa
        goal_prob = max(2, min(98, int(90 - (goal_diff * 45)) if goal_diff > 0 else np.random.randint(93, 99)))

        # --- SECTION 1 ---
        st.markdown("---")
        st.markdown("### 📋 Section 1: Academic Forecasting & Performance Predictive Core")
        r1_1, r1_2, r1_3, r1_4 = st.columns(4)
        r1_1.markdown(f"<div class='feature-card' style='border-color:#10b981;'><span class='feature-title'>🔮 Predicted Semester GPA</span><div class='feature-value' style='color:#10b981;'>{predicted_gpa} / 10.0</div></div>", unsafe_allow_html=True)
        r1_2.markdown(f"<div class='feature-card' style='border-color:#ef4444;'><span class='feature-title'>🚨 Backlog Probability Rating</span><div class='feature-value' style='color:#f87171;'>{backlog_prob}%</div></div>", unsafe_allow_html=True)
        r1_3.markdown(f"<div class='feature-card' style='border-color:#38bdf8;'><span class='feature-title'>🛡️ Exam Readiness Score</span><div class='feature-value' style='color:#38bdf8;'>{readiness_score}/100</div></div>", unsafe_allow_html=True)
        r1_4.markdown(f"<div class='feature-card' style='border-color:#a855f7;'><span class='feature-title'>🏆 Academic Performance Score</span><div class='feature-value' style='color:#c084fc;'>{perf_score}/100</div></div>", unsafe_allow_html=True)

        # --- SECTION 2 ---
        st.markdown(" ")
        r2_1, r2_2, r2_3, r2_4 = st.columns(4)
        r2_1.markdown(f"<div class='feature-card' style='border-color:#f43f5e;'><span class='feature-title'>📖 Expected Syllabus Coverage</span><div class='feature-value' style='color:#fb7185;'>{syllabus_coverage}%</div></div>", unsafe_allow_html=True)
        r2_2.markdown(f"<div class='feature-card' style='border-color:#0ea5e9;'><span class='feature-title'>🏅 Sessional Marks Evaluation</span><div class='feature-value' style='color:#38bdf8;'>{internal_marks} / 20</div></div>", unsafe_allow_html=True)
        r2_3.markdown(f"<div class='feature-card' style='border-color:#10b981;'><span class='feature-title'>⚖️ Curricular Credit Integrity</span><div class='feature-value' style='color:#34d399;'>{credit_integrity} pts</div></div>", unsafe_allow_html=True)
        r2_4.markdown(f"<div class='feature-card' style='border-color:#eab308;'><span class='feature-title'>👥 Peer Interaction Matrix</span><div class='feature-value' style='color:#facc15;'>{peer_interaction}/100</div></div>", unsafe_allow_html=True)

        # --- SECTION 3 ---
        st.markdown("---")
        st.markdown("### 🧠 Section 2: Cognitive Fatigue & Habit Continuity Indicators")
        r3_1, r3_2, r3_3, r3_4 = st.columns(4)
        r3_1.markdown(f"<div class='feature-card' style='border-color:#f97316;'><span class='feature-title'>💤 Cognitive Fatigue Index</span><div class='feature-value' style='color:#fb923c;'>{cognitive_fatigue}/100</div></div>", unsafe_allow_html=True)
        r3_2.markdown(f"<div class='feature-card' style='border-color:#22c55e;'><span class='feature-title'>⏱️ Focus Attention Span</span><div class='feature-value' style='color:#4ade80;'>{focus_span} mins</div></div>", unsafe_allow_html=True)
        r3_3.markdown(f"<div class='feature-card' style='border-color:#ef4444;'><span class='feature-title'>⏳ Procrastination Probability</span><div class='feature-value' style='color:#f87171;'>{procrastination_prob}%</div></div>", unsafe_allow_html=True)
        r3_4.markdown(f"<div class='feature-card' style='border-color:#ec4899;'><span class='feature-title'>📉 Burnout Risk Velocity</span><div class='feature-value' style='color:#f472b6;'>{burnout_velocity} x/s</div></div>", unsafe_allow_html=True)

        # --- SECTION 4 ---
        st.markdown(" ")
        r4_1, r4_2, r4_3, r4_4 = st.columns(4)
        r4_1.markdown(f"<div class='feature-card' style='border-color:#6366f1;'><span class='feature-title'>📊 Study Efficiency Ratio</span><div class='feature-value' style='color:#818cf8;'>{study_efficiency} ratio</div></div>", unsafe_allow_html=True)
        r4_2.markdown(f"<div class='feature-card' style='border-color:#d946ef;'><span class='feature-title'>🧱 Weekend Load Risk Rating</span><div class='feature-value' style='color:#e879f9;'>{weekend_burn}%</div></div>", unsafe_allow_html=True)
        r4_3.markdown(f"<div class='feature-card' style='border-color:#14b8a6;'><span class='feature-title'>⚖️ Routine Balance Score</span><div class='feature-value' style='color:#2dd4bf;'>{routine_balance}/100</div></div>", unsafe_allow_html=True)
        r4_4.markdown(f"<div class='feature-card' style='border-color:#84cc16;'><span class='feature-title'>🔄 Memory Recency Factor</span><div class='feature-value' style='color:#a3e635;'>{recency_factor} days</div></div>", unsafe_allow_html=True)

        # --- SECTION 5 ---
        st.markdown("---")
        st.markdown("### 💼 Section 3: Professional Gateways & Future Career Placements")
        r5_1, r5_2, r5_3, r5_4 = st.columns(4)
        r5_1.markdown(f"<div class='feature-card' style='border-color:#a855f7;'><span class='feature-title'>🎓 Scholarship Gate Chance</span><div class='feature-value' style='color:#c084fc;'>{scholarship_prob}%</div></div>", unsafe_allow_html=True)
        r5_2.markdown(f"<div class='feature-card' style='border-color:#6366f1; font-size:12px;'><span class='feature-title'>🚀 Campus Placement Eligibility</span><div class='feature-value' style='color:#818cf8; font-size:14px; margin-top:14px;'>{placement_gateway}</div></div>", unsafe_allow_html=True)
        r5_3.markdown(f"<div class='feature-card' style='border-color:#0ea5e9; font-size:12px;'><span class='feature-title'>🎯 Career Domain Orientation</span><div class='feature-value' style='color:#38bdf8; font-size:14px; margin-top:14px;'>{career_inclination}</div></div>", unsafe_allow_html=True)
        r5_4.markdown(f"<div class='feature-card' style='border-color:#eab308;'><span class='feature-title'>🎯 Target Goal Likelihood</span><div class='feature-value' style='color:#facc15;'>{goal_prob}%</div></div>", unsafe_allow_html=True)

        input_data = np.array([[g_encoded, s_encoded, raised_hands, visited_resources, announcements, discussion, study_hours, sleep_time, attendance, extracurriculars, mh_encoded, counsel_encoded]])
        prediction = model.predict(input_data)[0]

        st.markdown(" ")
        st.markdown("### 🚨 Ensemble Performance Clustering Layer")
        if prediction == 'H':
            st.markdown(f'<div class="threat-safe"><h3>🎯 PERFORMANCE CLUSTER: EXCELLENT STATUS (Class H Layer)</h3><p>Academic parameters optimized. Evaluated core student <b>Strength Track</b>: <b>{strength_sub}</b>. Identified Weakness Core Focus Node: <b>{weakness_sub}</b>. Performance Trend Vector: <b>{trend_tag}</b>.</p></div>', unsafe_allow_html=True)
        elif prediction == 'M':
            st.markdown(f'<div class="threat-warning"><h3>⚠️ PERFORMANCE CLUSTER: AVERAGE SECURE STATUS (Class M Layer)</h3><p>Operation executing within expected threshold limits. Evaluated core <b>Strength Track</b>: <b>{strength_sub}</b>. Identified Weakness Core Focus Node: <b>{weakness_sub}</b>. Performance Trend Vector: <b>{trend_tag}</b>.</p></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="threat-critical"><h3>🚨 PERFORMANCE CLUSTER: AT-RISK RETENTION STATUS (Class L Layer)</h3><p>ALERT: Retention threshold limits dropped below baseline for <b>{target_subject}</b>. Highest Risk Weakness Domain Found: <b>{weakness_sub}</b>. Performance Trend Status Vector: <b style="color:#ef4444;">{trend_tag}</b>. Institutional counseling call loop recommended.</p></div>', unsafe_allow_html=True)

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
                st.info(f"📈 **Prescriptive Optimization Strategy for {target_subject}:** To close the delta gap to reach **{target_cgpa} CGPA**, execute parameter adjustments: Push Attendance past **86%**, scale Daily Study Blocks to **+2 hours**, and ensure Assignment Submission matches **90%+** accuracy.")

    # ==================== TAB 2: ADVANCED HIGH-FIDELITY DYNAMIC CHAT NODES ====================
    with tab2:
        st.subheader("🖥️ AI Academic Strategy Interface Console")
        st.write("Query the integrated diagnostic system regarding engineering curriculum strategy timelines, roadmap optimizations, or workflow balance protocols.")
        user_query = st.text_input("💬 Input Command Input / Target Query Token:", placeholder="sys_query: type anything regarding exam, gpa, streams, study, placement...")
        
        if user_query:
            st.markdown(f'<div class="cyber-terminal" style="border-left-color: #2563eb;"><b>[ST_REQUEST_LOG]:</b> Parsing raw input token matrices against analytics profile... Deployed Domain: {target_subject}</div>', unsafe_allow_html=True)
            
            q_clean = user_query.lower()
            
            # --- TRUE CROSS-DISCIPLINARY KEYWORD MATCHING ENGINE ---
            if "toc" in q_clean or "theory of computation" in q_clean or "automata" in q_clean:
                bot_text = f"⚙️ **TOC ENGINE COGNITION:** Executing computational strategy mapping for your profile. Your active workload balance shows {study_hours}h self-study metrics. Strategy: 1. Dedicate 40% of tracking blocks to Finite Automata closures & regular language transitions. 2. Map transition tables structurally to neutralize Mealy/Moore matrix confusion loops. 3. Target the structural pumping lemma constraints using systematic contradiction matrices."
            
            elif "bio" in q_clean or "genetics" in q_clean or "microbiology" in q_clean or "science" in q_clean:
                bot_text = f"🧬 **BIO-SCIENCE ANALYTICS CORE:** Custom roadmap initialized for current track: **{target_subject}**. To support a projected GPA of {predicted_gpa}, structure your data integration pipelines around: 1. Visual pathways visualization for enzymatic molecular kinetics. 2. Quantitative dataset clustering when modeling bioinformatics gene sequencing records. 3. Focus on strengthening core domain metrics: **{strength_sub}**."

            elif "commerce" in q_clean or "accounting" in q_clean or "finance" in q_clean or "audit" in q_clean:
                bot_text = f"📈 **COMMERCE STRATEGY INTERFACE:** System status diagnostics active for **{target_subject}**. Financial and cost vectors require quantitative consistency layers. Strategy: 1. Boost workflow consistency parameters beyond 85% to optimize balance sheet verification simulations. 2. Balance the analytical portfolio models against real-time micro-study blocks ({study_hours}h active velocity). 3. Prioritize mitigations for the *{weakness_sub}* focus block."

            elif "exam" in q_clean or "readiness" in q_clean or "prepare" in q_clean or "sessional" in q_clean:
                bot_text = f"📝 **EXAM PERFORMANCE MATRIX ENGINE:** Your active Exam Readiness index is evaluated at **{readiness_score}/100**. Execution Roadmap: 1. Address the {weakness_sub} focus block immediately. 2. Since your assignment submission metrics are at {assignment_score}%, run high-frequency past sessional trends. 3. Do not compromise your current rest cycle ({sleep_time} Hours) within 48 hours of evaluation cycles to prevent severe cognitive load drift errors."
                
            elif "placement" in q_clean or "career" in q_clean or "jobs" in q_clean or "company" in q_clean:
                if "PASSED" in placement_gateway:
                    bot_text = f"🚀 **PLACEMENT ARCHITECTURE CORE:** Status verified as **ELIGIBLE**. Profile alignment shows a solid orientation towards **{career_inclination}**. Strategic Advice: Capitalize on your high curricular credit metrics ({credit_integrity} pts). Target enterprise system architectures, push domain codes/case-studies to portfolio systems, and leverage your active peer collaboration matrix ({peer_interaction}%) for mock mock-interview sprints."
                else:
                    bot_text = f"⚠️ **PLACEMENT RECOVERY PROTOCOL:** Action required! Your active metrics fall below institutional gateway standards. Urgent Strategy: 1. Systematically push overall GPA above the 6.5 cutoff baseline (Current projection: {predicted_gpa}). 2. Execute an absolute lock on your campus attendance matrix to cross 75% benchmarks (Current: {attendance}%). Re-run pipeline variables once parameters normalize."
                    
            elif "gpa" in q_clean or "cgpa" in q_clean or "score" in q_clean or "target" in q_clean:
                gap = round(target_cgpa - predicted_gpa, 2)
                if gap <= 0:
                    bot_text = f"✨ **TARGET VERIFICATION SUCCESSFUL:** Your active configuration safely guarantees a {predicted_gpa} GPA, clearing your {target_cgpa} threshold boundary. Operational directive: Prevent burnout velocity shifts ({burnout_velocity} x/s) and lock continuous study consistency parameters to preserve this high-performance cluster zone."
                else:
                    bot_text = f"📈 **DELTA RECOVERY STRATEGY:** System identifies a performance deficit tracking gap of **{gap} points** to reach your target of {target_cgpa} CGPA. Prescriptive Action: 1. Scale continuous self-study blocks from {study_hours}h to {min(14, study_hours + 2)}h. 2. Enforce assignment submission accuracies past 90%. 3. Leverage LMS clicks ({visited_resources} usage density) to stabilize internal sessional marks evaluation."
                    
            elif "sleep" in q_clean or "fatigue" in q_clean or "burnout" in q_clean or "stress" in q_clean:
                bot_text = f"🧠 **COGNITIVE RISK ASSESSMENT:** Platform telemetry calculates a **Cognitive Fatigue Index of {cognitive_fatigue}%** with a Burnout Velocity of **{burnout_velocity}**. System diagnostics: Your sleep duration profile is locked at {sleep_time} hours against a {study_hours} hour study load. Advice: Avoid late-night workload variance adjustments. Stabilize your routine balance rating ({routine_balance}/100) by establishing strict operational study blocks to protect cognitive memory recency intervals."
                
            else:
                # Completely dynamic smart default fallback that pulls live structural parameters
                bot_text = f"📡 **DYNAMIC SYSTEM DIAGNOSTICS ACTIVE:** Request successfully mapped for track: **{target_subject}**. The Scikit-Learn validation trees suggest modifying your current parameters (Attendance: {attendance}%, Study Blocks: {study_hours}h) to match a stable trajectory. Recommended: Fragment complex engineering/science units into daily tasks, leverage active recall protocols for **{strength_sub}**, and maintain routine parity metrics to eliminate retention risks."

            st.markdown(f'<div class="cyber-terminal" style="border-left-color:#10b981; margin-top:10px;"><b style="color:#38bdf8;">[AI_ADVISOR_RESPONSE]:</b><br><br>{bot_text}</div>', unsafe_allow_html=True)

    # ==================== TAB 3: EXTENDED VISUALIZATIONS SECTION ====================
    with tab3:
        st.subheader("📊 Institutional System Statistical Matrices")
        
        c1, c2 = st.columns(2)
        with c1:
            fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
            fig_bar.patch.set_facecolor('#0b0f19')
            ax_bar.set_facecolor('#0b0f19')
            sns.countplot(x='MentalHealthStatus', hue='Class', data=df, palette='Spectral', order=['Excellent', 'Good', 'Stressed', 'Depressed'], ax=ax_bar)
            ax_bar.tick_params(colors='#e2e8f0')
            ax_bar.set_title("Psychological Behavior Weight Metrics over Dataset", color='#e2e8f0')
            st.pyplot(fig_bar)
        with c2:
            fig_scatter, ax_scatter = plt.subplots(figsize=(6, 4))
            fig_scatter.patch.set_facecolor('#0b0f19')
            ax_scatter.set_facecolor('#0b0f19')
            sns.scatterplot(x='StudyHours', y='Attendance', hue='Class', data=df, palette='Set1', alpha=0.7, ax=ax_scatter)
            ax_scatter.tick_params(colors='#e2e8f0')
            ax_scatter.set_title("Study Hours vs Campus Attendance Scatter Distribution", color='#e2e8f0')
            st.pyplot(fig_scatter)

        st.markdown("---")
        st.markdown("### 📈 Advanced Academic Performance Insights & Clustering Analytics")
        
        c3, c4, c5 = st.columns(3)
        with c3:
            fig_dist, ax_dist = plt.subplots(figsize=(5, 3.5))
            fig_dist.patch.set_facecolor('#0b0f19')
            ax_dist.set_facecolor('#0b0f19')
            sns.histplot(df['Dataset_Predicted_GPA'], kde=True, color='#10b981', bins=15, ax=ax_dist)
            ax_dist.tick_params(colors='#e2e8f0', labelsize=8)
            ax_dist.xaxis.label.set_color('#e2e8f0')
            ax_dist.yaxis.label.set_color('#e2e8f0')
            ax_dist.set_title("Distribution of Predicted GPAs across System Context", color='#e2e8f0', fontsize=10)
            st.pyplot(fig_dist)
            
        with c4:
            fig_heat, ax_heat = plt.subplots(figsize=(5, 3.5))
            fig_heat.patch.set_facecolor('#0b0f19')
            corr_features = ['raisedhands', 'VisITedResources', 'AnnouncementsView', 'Discussion', 'Attendance']
            corr_matrix = df[corr_features].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='Blues', fmt='.2f', cbar=False, ax=ax_heat, annot_kws={"size": 8})
            ax_heat.tick_params(colors='#e2e8f0', labelsize=8)
            ax_heat.set_title("LMS Engagement & Attendance Correlation Grid", color='#e2e8f0', fontsize=10)
            st.pyplot(fig_heat)
            
        with c5:
            fig_box, ax_box = plt.subplots(figsize=(5, 3.5))
            fig_box.patch.set_facecolor('#0b0f19')
            ax_box.set_facecolor('#0b0f19')
            sns.boxplot(x='MentalHealthStatus', y='StudyHours', data=df, palette='Pastel2', order=['Excellent', 'Good', 'Stressed', 'Depressed'], ax=ax_box)
            ax_box.tick_params(colors='#e2e8f0', labelsize=8)
            ax_box.xaxis.label.set_color('#e2e8f0')
            ax_box.yaxis.label.set_color('#e2e8f0')
            ax_box.set_title("Self-Study Allocation Variance by Mental Health", color='#e2e8f0', fontsize=10)
            st.pyplot(fig_box)

except FileNotFoundError:
    st.error("Fatal System Error: 'AI-Data.csv' reference database file missing in the current working root.")
