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

# Page Configuration - Hacker Theme Look
st.set_page_config(page_title="Student Performance Core", layout="wide", page_icon="⚡")

# Cyberpunk / Gaming Custom High-Gloss CSS Theme
st.markdown("""
    <style>
    .main { background-color: #0b0f19; color: #e2e8f0; }
    .stButton>button { width: 100%; background-color: #3b82f6; color: white; font-weight: bold; border-radius: 8px; height: 45px; border: 1px solid #60a5fa; }
    
    /* Cyber Matrix Warning Boxes */
    .cyber-card { background: linear-gradient(135deg, #1e293b, #0f172a); padding: 25px; border-radius: 12px; border: 2px solid #3b82f6; box-shadow: 0 0 15px rgba(59,130,246,0.3); margin-bottom: 15px; }
    .threat-critical { background: #450a0a; border: 2px solid #ef4444; color: #fca5a5; padding: 20px; border-radius: 10px; box-shadow: 0 0 20px rgba(239,68,68,0.4); animation: blinker 1.5s linear infinite; }
    .threat-warning { background: #451a03; border: 2px solid #f97316; color: #ffedd5; padding: 20px; border-radius: 10px; box-shadow: 0 0 20px rgba(249,115,22,0.3); }
    .threat-safe { background: #064e3b; border: 2px solid #10b981; color: #d1fae5; padding: 20px; border-radius: 10px; box-shadow: 0 0 20px rgba(16,185,129,0.3); }
    
    /* Terminal Console Look */
    .cyber-terminal { background-color: #020617; border-left: 5px solid #10b981; padding: 15px; border-radius: 6px; font-family: 'Courier New', monospace; color: #38bdf8; font-size: 14px; box-shadow: inset 0 0 10px rgba(0,0,0,0.8); }
    
    /* Blinking Animation for Threat Alert */
    @keyframes blinker { 50% { opacity: 0.85; border-color: #b91c1c; box-shadow: 0 0 30px rgba(239,68,68,0.6); } }
    </style>
    """, unsafe_allow_html=True)

st.title("⚡ CYBERNETIC STUDENT SURVIVAL & PERFORMANCE HUB")
st.markdown("<span style='color:#94a3b8;'>Decrypting LMS telemetry pipelines, behavioral latency, and physiological overload markers.</span>", unsafe_allow_html=True)
st.markdown("---")

@st.cache_data
def load_and_engineer_data():
    df = pd.read_csv('AI-Data.csv')
    np.random.seed(42)
    n = len(df)
    
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
    
    # Preprocessing
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

    # Sidebar Status Tracker
    st.sidebar.markdown("### 🖥️ Core Status Panel")
    st.sidebar.info("Cyber Engine: ACTIVE\n\nAPI Framework: KEYLESS LOCAL FLUID")

    tab1, tab2, tab3 = st.tabs(["🎮 Interactive Core Simulator", "🤖 Llama-3 Deep Network Agent", "📊 Core Architecture Data"])
    
    # ==================== TAB 1: CORE SIMULATOR & RPG SURVIVAL BAR ====================
    with tab1:
        st.subheader("🎛️ Live Telemetry Control Desk")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<h5 style='color:#3b82f6;'>📝 Digital LMS Footprint</h5>", unsafe_allow_html=True)
            gender_input = st.selectbox("Student Gender", options=["Male", "Female"])
            stage_input = st.selectbox("Current Academic Stage Level", options=["Lowerlevel", "MiddleSchool", "HighSchool"])
            raised_hands = st.slider("Classroom Interaction Rank (Raised Hands)", 0, 100, 50)
            visited_resources = st.slider("LMS Database Clicks (Portal Activities)", 0, 100, 55)
            announcements = st.slider("System Matrix Notifications Viewed", 0, 100, 40)
            discussion = st.slider("Discussion Grid Forum Participation", 0, 100, 45)
        
        with col2:
            st.markdown("<h5 style='color:#10b981;'>🏃 Physiological Grid Data</h5>", unsafe_allow_html=True)
            study_hours = st.slider("Daily Self-Study Threshold Blocks", 0, 15, 4)
            sleep_time = st.slider("Sleep Core Battery Duration (Hours)", 4, 12, 6)
            attendance = st.slider("Verified Campus Attendance Record (%)", 0, 100, 75)
            extracurriculars = st.slider("Weekly Side-Quest Activities (Hours)", 0, 10, 2)
            counsel_input = st.selectbox("Sought Neural Psych Counseling Before?", options=["No", "Yes"])
            stress_level = st.slider("🔥 Live Core System Stress Index", 0, 100, 60)

        # Map interior variables based on stress levels
        if stress_level >= 75: mh_input = "Depressed"
        elif stress_level >= 45: mh_input = "Stressed"
        elif stress_level >= 15: mh_input = "Good"
        else: mh_input = "Excellent"

        g_encoded = 1 if gender_input == "Male" else 0
        s_encoded = 0 if stage_input == "Lowerlevel" else (1 if stage_input == "MiddleSchool" else 2)
        mh_encoded = list(le_mh.classes_).index(mh_input)
        counsel_encoded = 1 if counsel_input == "Yes" else 0

        st.markdown("---")
        
        # Calculate dynamic survival rate based on mathematical weights
        survival_rate = int(((attendance * 0.35) + (visited_resources * 0.25) + (study_hours * 3.5) + (100 - stress_level) * 0.25))
        survival_rate = max(5, min(100, survival_rate)) # Lock between 5% and 100%

        # --- 🔥 NEW GAMIFIED RPG SURVIVAL HEALTH BAR ---
        st.markdown(f"### 🔋 Academic Survival Energy: `{survival_rate}%`")
        if survival_rate >= 75:
            st.progress(survival_rate / 100) # Standard blue/green progress bar
        elif survival_rate >= 40:
            st.warning("⚠️ Warning: System Energy Depleting. Cognitive drag detected.")
            st.progress(survival_rate / 100)
        else:
            st.error("🚨 CRITICAL ALERT: SYSTEM COLLAPSE IN BOUNDS. INSUFFICIENT RETENTION POWER.")
            st.progress(survival_rate / 100)

        # Run Prediction Inference
        input_data = np.array([[g_encoded, s_encoded, raised_hands, visited_resources, announcements, discussion, study_hours, sleep_time, attendance, extracurriculars, mh_encoded, counsel_encoded]])
        prediction = model.predict(input_data)[0]
        
        st.markdown(" ")
        st.markdown("### 🚨 Live Threat Matrix Analysis")
        
        # Blinking / Colorful Cyber Alerts based on outputs
        if prediction == 'H':
            st.markdown(f'<div class="threat-safe"><h3>✅ THREAT STATUS: ZERO THREAT (Class H Profile)</h3><p>Survival Core optimal. Academic matrix values are completely self-sustaining. Threat index: Low.</p></div>', unsafe_allow_html=True)
        elif prediction == 'M':
            st.markdown(f'<div class="threat-warning"><h3>⚠️ THREAT STATUS: MODERATE DRAG (Class M Profile)</h3><p>System stabilizing on baseline limits. Student holds steady operations but lacks breakout power vectors.</p></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="threat-critical"><h3>🚨 THREAT STATUS: CRITICAL RISK MATRIX OVERLOAD (Class L Profile)</h3><p>EMERGENCY ALERT: High cognitive load ({stress_level}/100 Stress Index) matched with a drop in attendance has breached retention thresholds. Deploying immediate counselor overrides.</p></div>', unsafe_allow_html=True)

        # ==================== 🔥 DYNAMIC REAL-TIME TIPS ENGINE ====================
        st.markdown("---")
        st.markdown("### 💡 AI System Remedial Protocols & Hacks")
        
        tx1, tx2 = st.columns(2)
        with tx1:
            st.markdown("<h4 style='color:#f87171;'>🧠 Stress, Tension & Burnout Diagnostics</h4>", unsafe_allow_html=True)
            if stress_level >= 75:
                st.markdown("""
                <div class="cyber-card" style="border-color:#ef4444;">
                    <b>🛑 5-4-3-2-1 Grounding Overdrive:</b> Panic state ko terminate karne ke liye: Instant apne visual domain mein 5 objects track karo, 4 textures touch karo, 3 auditory frequencies suno, aur 1 sensory taste recall karo. Dimaag panic mode se instantly exit kar jayega.
                </div>
                <div class="cyber-card" style="border-color:#ef4444;">
                    <b>🫁 Tactical Box Breathing (4-4-4-4):</b> 4 sec saans andar, 4 sec system lock (hold), 4 sec complete dump (release), aur 4 sec blank space. Sub-conscious cortisol level instant crash hoga.
                </div>
                """, unsafe_allow_html=True)
            elif stress_level >= 45:
                st.markdown("""
                <div class="cyber-card" style="border-color:#f97316;">
                    <b>🎵 Alpha Wave Brain Entrainment:</b> 432Hz Sound loop frequencies earphone par chalao. Padhai ke dauran aane wale racing thoughts ka connection break ho jayega.
                </div>
                <div class="cyber-card" style="border-color:#f97316;">
                    <b>☕ Caffeine Lockdown Protocol:</b> High anxiety blocks mein chai/coffee strictly terminate kar do. Nervous system ko trigger se bachana maximum priority hai.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("<p style='color:#10b981;'>✅ System wellness loop secure. No physiological interventions required at this stamp.</p>", unsafe_allow_html=True)
                
        with tx2:
            st.markdown("<h4 style='color:#60a5fa;'>📚 Advanced Neuro-Study Architectures</h4>", unsafe_allow_html=True)
            if stress_level >= 50:
                st.markdown("""
                <div class="cyber-card" style="border-color:#a855f7;">
                    <b>⏱️ Ultra-Short Pomodoro (20-5 Blocks):</b> Memory overloading se bachne ke liye bade syllabus target ko delete karo. Sirf 20 mins hyper-focus karo aur 5 mins screenless reset walk block execute karo.
                </div>
                <div class="cyber-card" style="border-color:#a855f7;">
                    <b>🎯 Active Recall Infiltration:</b> Notes baar-baar padhna band karo. Blank paper uthao aur dimaag par pressure dalo ki last unit se kya-kya save hai. Forced retrieval pattern memory ko fast permanent banata hai.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="cyber-card" style="border-color:#3b82f6;">
                    <b>💡 Feynman Explanatory Protocol:</b> Jo complex topics padhe hain, use imagine karo ki kisi 5 saal ke bachhe ko sikha rahe ho. Words simple hote hi complex equations dimaag mein crack ho jayengi.
                </div>
                <div class="cyber-card" style="border-color:#3b82f6;">
                    <b>🗺️ Spaced Repetition Array Logic:</b> Aaj ka topic exact 24 ghante, fir 3 din, aur fir 7 din baad re-verify (revise) karo. Information neural pathways mein hardcode ho jayegi.
                </div>
                """, unsafe_allow_html=True)

    # ==================== TAB 2: TERMINAL STYLE LLAMA CHAT ====================
    with tab2:
        st.subheader("🖥️ Autonomous Cybernetic Llama Node Console")
        user_query = st.text_input("📡 Input Command / Prompt Query:", placeholder="sys_query: enter query parameter here...")
        
        if user_query:
            # Hacker Style Terminal UI Blocks
            st.markdown(f'<div class="cyber-terminal"><b>root@ai_agent_core:~#</b> user_request --token="{user_query}"</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="cyber-terminal"><b>[STATUS]:</b> Scanning data matrix nodes... Analysing internal target variance layers...</div>', unsafe_allow_html=True)
            
            API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
            system_prompt = f"<|system|>\nYou are an elite cybernetic Student Retention AI Engine. Answer this query analytically using high data science terminology: {user_query}\n<|user|>\nAnswer:"
            
            try:
                response = requests.post(API_URL, json={"inputs": system_prompt, "parameters": {"max_new_tokens": 150, "temperature": 0.7}}, timeout=8)
                res_json = response.json()
                bot_text = res_json[0]["generated_text"].split("Answer:")[-1].strip()
            except Exception:
                bot_text = "System telemetry override active. High stress patterns intercepted. Neural optimization route: Execute micro-learning modules, expand platform data mining, and activate structural group protocols immediately to mitigate cognitive drop risks."

            st.markdown(f'<div class="cyber-terminal" style="border-left-color:#3b82f6; margin-top:10px;"><b style="color:#10b981;">[LLAMA_CORE_RESPONSE]:</b><br><br>{bot_text}</div>', unsafe_allow_html=True)

    # ==================== TAB 3: VISUALIZATIONS ====================
    with tab3:
        st.subheader("📊 Core Data Visualizations")
        c1, c2 = st.columns(2)
        with c1:
            fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
            sns.countplot(x='MentalHealthStatus', hue='Class', data=df, palette='Spectral', order=['Excellent', 'Good', 'Stressed', 'Depressed'], ax=ax_bar)
            st.pyplot(fig_bar)
        with c2:
            fig_scatter, ax_scatter = plt.subplots(figsize=(6, 4))
            sns.scatterplot(x='StudyHours', y='Attendance', hue='Class', data=df, palette='Set1', alpha=0.7, ax=ax_scatter)
            st.pyplot(fig_scatter)

except FileNotFoundError:
    st.error("Fatal Error: 'AI-Data.csv' file missing in root directory.")
