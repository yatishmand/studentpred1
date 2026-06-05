import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Page Config for Professional Look
st.set_page_config(page_title="ML Student Analytics", layout="wide", page_icon="🎓")

# Custom Styling
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; background-color: #4F46E5; color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("📊 Advanced Student Performance Analysis & Prediction")
st.markdown("---")

@st.cache_data
def load_data():
    return pd.read_csv('AI-Data.csv')

try:
    df = load_data()
    
    # Sidebar - Project Metadata
    st.sidebar.header("📁 Project Dashboard")
    st.sidebar.info("Model: **Random Forest Classifier**")
    st.sidebar.write(f"Total Student Records: `{len(df)}`")
    st.sidebar.write(f"Features Analyzed: `6`")
    
    # Tabs for better UI organization
    tab1, tab2, tab3 = st.tabs(["🔮 Live Predictor", "📈 Data Visualizations", "⚙️ Model Evaluation"])
    
    # --- PREPROCESSING & TRAINING ---
    le_gender = LabelEncoder()
    df['gender_encoded'] = le_gender.fit_transform(df['gender'])
    le_stage = LabelEncoder()
    df['Stage_encoded'] = le_stage.fit_transform(df['StageID'])

    X = df[['gender_encoded', 'Stage_encoded', 'raisedhands', 'VisITedResources', 'AnnouncementsView', 'Discussion']]
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # ==================== TAB 1: LIVE PREDICTOR ====================
    with tab1:
        st.subheader("👨‍💻 Enter Student Parameters for Live Inference")
        
        col1, col2 = st.columns(2)
        with col1:
            gender_input = st.selectbox("Gender", options=["Male", "Female"])
            stage_input = st.selectbox("School Stage ID", options=["Lowerlevel", "MiddleSchool", "HighSchool"])
            raised_hands = st.slider("Raised Hands (Class Participation)", 0, 100, 65)
        with col2:
            visited_resources = st.slider("Visited Resources (Online Material)", 0, 100, 75)
            announcements = st.slider("Announcements Viewed", 0, 100, 40)
            discussion = st.slider("Discussion Group Participation", 0, 100, 55)

        g_encoded = 1 if gender_input == "Male" else 0
        s_encoded = 0 if stage_input == "Lowerlevel" else (1 if stage_input == "MiddleSchool" else 2)

        st.write("")
        if st.button("🔮 Run Machine Learning Inference", type="primary"):
            input_data = np.array([[g_encoded, s_encoded, raised_hands, visited_resources, announcements, discussion]])
            prediction = model.predict(input_data)[0]
            
            st.markdown("### **Deployment Output:**")
            if prediction == 'H':
                st.success("🎯 **Predicted Category: HIGH (H)** \n\nThis student shows excellent engagement and is highly likely to secure top grades.")
            elif prediction == 'M':
                st.info("📊 **Predicted Category: MEDIUM (M)** \n\nThis student performs consistently but has room for improvement in class participation.")
            else:
                st.warning("⚠️ **Predicted Category: LOW (L)** \n\nWarning: High risk of poor academic performance. Needs immediate intervention.")

    # ==================== TAB 2: VISUALIZATIONS ====================
    with tab2:
        st.subheader("🔍 Dataset Overview & Visual Analytics")
        st.write("First 5 rows of the loaded `AI-Data.csv` file:")
        st.dataframe(df[['gender', 'StageID', 'raisedhands', 'VisITedResources', 'Class']].head())
        
        st.markdown("---")
        st.write("📊 **Feature Distribution vs Class Output**")
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        sns.boxplot(x='Class', y='raisedhands', data=df, order=['L', 'M', 'H'], ax=ax[0], palette='Set2')
        ax[0].set_title('Raised Hands vs Academic Class')
        
        sns.boxplot(x='Class', y='VisITedResources', data=df, order=['L', 'M', 'H'], ax=ax[1], palette='Pastel1')
        ax[1].set_title('Visited Resources vs Academic Class')
        
        st.pyplot(fig)

    # ==================== TAB 3: MODEL EVALUATION ====================
    with tab3:
        st.subheader("⚙️ Backend Model Performance Metrics")
        
        c1, c2 = st.columns(2)
        c1.metric(label="Model Accuracy", value=f"{acc*100:.2f}%", delta="Random Forest")
        c2.write("**Classification Report:**")
        
        # Displaying a clean text report
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format(precision=2))

except FileNotFoundError:
    st.error("Fatal Error: 'AI-Data.csv' database file not found in the repository root.")
