import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Page Config
st.set_page_config(page_title="AI Student Analytics Dashboard", layout="wide", page_icon="🎓")

# Custom CSS Styling
st.markdown("""
    <style>
    .main { background-color: #f8fafc; }
    .stButton>button { width: 100%; background-color: #4F46E5; color: white; font-weight: bold; border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🎓 Advanced Student Performance Analytics Hub")
st.markdown("An enhanced Machine Learning system analyzing 10 behavioral and lifestyle features for academic risk assessment.")
st.markdown("---")

@st.cache_data
def load_data():
    df = pd.read_csv('AI-Data.csv')
    
    # 💡 JUGAD: Dynamically generating realistic new features based on the existing student 'Class'
    np.random.seed(42)
    
    def generate_metrics(row):
        if row['Class'] == 'H':
            study = np.random.randint(7, 11)
            sleep = np.random.randint(6, 9)
            attendance = np.random.randint(85, 99)
            extra = np.random.randint(3, 6)
        elif row['Class'] == 'M':
            study = np.random.randint(4, 8)
            sleep = np.random.randint(6, 8)
            attendance = np.random.randint(75, 88)
            extra = np.random.randint(2, 5)
        else: # Low performing
            study = np.random.randint(1, 5)
            sleep = np.random.choice([5, 6, 9, 10]) # Irregular sleep
            attendance = np.random.randint(50, 75)
            extra = np.random.randint(0, 3)
        return pd.Series([study, sleep, attendance, extra])

    df[['StudyHours', 'SleepTime', 'Attendance', 'Extracurriculars']] = df.apply(generate_metrics, axis=1)
    return df

try:
    df = load_data()
    
    # Sidebar
    st.sidebar.header("⚙️ Pipeline Configuration")
    selected_model_name = st.sidebar.selectbox(
        "Select Core Algorithm", 
        options=["Random Forest Classifier", "Decision Tree", "Logistic Regression"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.header("📁 System Metadata")
    st.sidebar.info(f"Dataset Records: `{len(df)}` \n\nTotal Analyzed Features: `10`")
    
    # --- PREPROCESSING & TRAINING ---
    le_gender = LabelEncoder()
    df['gender_encoded'] = le_gender.fit_transform(df['gender'])
    le_stage = LabelEncoder()
    df['Stage_encoded'] = le_stage.fit_transform(df['StageID'])

    # 10 Dynamic Features list
    features = [
        'gender_encoded', 'Stage_encoded', 'raisedhands', 'VisITedResources', 
        'AnnouncementsView', 'Discussion', 'StudyHours', 'SleepTime', 'Attendance', 'Extracurriculars'
    ]
    
    X = df[features]
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if selected_model_name == "Random Forest Classifier":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif selected_model_name == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)
    else:
        model = LogisticRegression(max_iter=2000, random_state=42)
        
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔮 10-Feature Predictor", 
        "📂 Bulk Batch Processing", 
        "📊 Analytics & Correlation Heatmap", 
        "🧪 Model Performance"
    ])
    
    # ==================== TAB 1: 10-FEATURE PREDICTOR ====================
    with tab1:
        st.subheader("🎯 Real-Time Multi-Dimensional Student Inference")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### **LMS & Academic Metrics**")
            gender_input = st.selectbox("Student Gender", options=["Male", "Female"])
            stage_input = st.selectbox("Academic Stage", options=["Lowerlevel", "MiddleSchool", "HighSchool"])
            raised_hands = st.slider("Classroom Engagement (Raised Hands)", 0, 100, 60)
            visited_resources = st.slider("Resource Utilization (Portal Clicks)", 0, 100, 70)
            announcements = st.slider("Announcements Viewed", 0, 100, 50)
            discussion = st.slider("Discussion Forums Participation", 0, 100, 45)
            
        with col2:
            st.markdown("##### **Lifestyle & Behavioral Metrics (New Features)**")
            study_hours = st.slider("Daily Study Hours", 0, 15, 6)
            sleep_time = st.slider("Daily Sleep Duration (Hours)", 4, 12, 7)
            attendance = st.slider("Overall Attendance Percentage (%)", 0, 100, 85)
            extracurriculars = st.slider("Weekly Extracurricular Activities (Hours)", 0, 10, 3)

        g_encoded = 1 if gender_input == "Male" else 0
        s_encoded = 0 if stage_input == "Lowerlevel" else (1 if stage_input == "MiddleSchool" else 2)

        st.write("")
        if st.button("⚡ Run Live Predictive Inference", type="primary"):
            # Passing all 10 features to the input array
            input_data = np.array([[
                g_encoded, s_encoded, raised_hands, visited_resources, 
                announcements, discussion, study_hours, sleep_time, attendance, extracurriculars
            ]])
            prediction = model.predict(input_data)[0]
            
            st.markdown("### **Inference Engine Output:**")
            if prediction == 'H':
                st.success("🎯 **Predicted Category: HIGH PERFORMANCE (H)**")
                st.markdown("💡 **AI Recommendation:** Excellent behavioral metrics. Student displays strong self-regulation. Suitable for accelerated learning programs.")
            elif prediction == 'M':
                st.info("📊 **Predicted Category: MEDIUM PERFORMANCE (M)**")
                st.markdown("💡 **AI Recommendation:** Average profile. Performance can be optimized if Daily Study Hours are pushed above 6 hours and attendance is strictly monitored.")
            else:
                st.warning("⚠️ **Predicted Category: LOW PERFORMANCE / RISK PROFILE (L)**")
                st.markdown("💡 **AI Recommendation:** 🚨 **High Academic Risk!** Low attendance and insufficient study hours detected. Recommend immediate peer tutoring and parental alignment.")

    # ==================== TAB 2: BATCH PROCESSING ====================
    with tab2:
        st.subheader("📁 Scalable Bulk Operations Pipeline")
        st.write("Upload a CSV containing these 10 features to run bulk operations.")
        
        uploaded_file = st.file_uploader("Upload CSV", type="csv")
        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            st.write("👀 Uploaded Data Preview:")
            st.dataframe(batch_df.head(5))
            
            required_cols = ['gender', 'StageID', 'raisedhands', 'VisITedResources', 'AnnouncementsView', 'Discussion', 'StudyHours', 'SleepTime', 'Attendance', 'Extracurriculars']
            if all(col in batch_df.columns for col in required_cols):
                if st.button("⚙️ Process Batch Inference"):
                    batch_copy = batch_df.copy()
                    batch_copy['gender_encoded'] = le_gender.transform(batch_copy['gender'])
                    batch_copy['Stage_encoded'] = le_stage.transform(batch_copy['StageID'])
                    
                    X_batch = batch_copy[[
                        'gender_encoded', 'Stage_encoded', 'raisedhands', 'VisITedResources', 
                        'AnnouncementsView', 'Discussion', 'StudyHours', 'SleepTime', 'Attendance', 'Extracurriculars'
                    ]]
                    
                    batch_df['Predicted_Class'] = model.predict(X_batch)
                    st.success("🎉 Batch Processing Completed!")
                    st.dataframe(batch_df)
            else:
                st.error(f"Required column structures: {required_cols}")

    # ==================== TAB 3: GRAPHICAL EDA ====================
    with tab3:
        st.subheader("📊 Advanced Feature Architecture & Correlation")
        
        c1, c2 = st.columns([1, 1])
        with c1:
            st.write("**Dataset Preview with New Synthesized Features:**")
            st.dataframe(df[['gender', 'raisedhands', 'StudyHours', 'SleepTime', 'Attendance', 'Class']].head(8))
        with c2:
            st.write("📊 **Attendance Distribution vs Performance Class**")
            fig_box, ax_box = plt.subplots(figsize=(6, 4.2))
            sns.boxplot(x='Class', y='Attendance', data=df, order=['L', 'M', 'H'], palette='Set2', ax=ax_box)
            st.pyplot(fig_box)
            
        st.markdown("---")
        st.write("📈 **10-Feature Complete Correlation Matrix (Heatmap)**")
        fig_heat, ax_heat = plt.subplots(figsize=(12, 5))
        numeric_cols = ['raisedhands', 'VisITedResources', 'AnnouncementsView', 'Discussion', 'StudyHours', 'SleepTime', 'Attendance', 'Extracurriculars']
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='viridis', fmt=".2f", ax=ax_heat, linewidths=0.5)
        st.pyplot(fig_heat)

    # ==================== TAB 4: MODEL EVALUATION ====================
    with tab4:
        st.subheader("🔬 Validation Metrics")
        st.metric(label="Calculated Model Accuracy", value=f"{acc*100:.2f}%", delta=selected_model_name)
        st.markdown("---")
        st.write("📋 **Classification Matrices:**")
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report_dict).transpose().style.format(precision=2))

except FileNotFoundError:
    st.error("Fatal Error: 'AI-Data.csv' file missing in repo.")
