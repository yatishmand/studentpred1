import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Student Performance Predictor", layout="centered")
st.title("🎓 Student Performance Prediction System")
st.write("Enter student details below to predict their performance level.")

@st.cache_data
def load_data():
    return pd.read_csv('AI-Data.csv')

try:
    df = load_data()
    le_gender = LabelEncoder()
    df['gender'] = le_gender.fit_transform(df['gender'])
    le_stage = LabelEncoder()
    df['StageID'] = le_stage.fit_transform(df['StageID'])

    X = df[['gender', 'StageID', 'raisedhands', 'VisITedResources', 'AnnouncementsView', 'Discussion']]
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    col1, col2 = st.columns(2)
    with col1:
        gender_input = st.selectbox("Gender", options=["Male", "Female"])
        stage_input = st.selectbox("School Stage", options=["Lowerlevel", "MiddleSchool", "HighSchool"])
        raised_hands = st.slider("Raised Hands in Class", 0, 100, 50)
    with col2:
        visited_resources = st.slider("Visited Resources", 0, 100, 50)
        announcements = st.slider("Announcements Viewed", 0, 100, 50)
        discussion = st.slider("Discussion Participation", 0, 100, 50)

    g_encoded = 1 if gender_input == "Male" else 0
    s_encoded = 0 if stage_input == "Lowerlevel" else (1 if stage_input == "MiddleSchool" else 2)

    if st.button("🔮 Predict Performance", type="primary"):
        input_data = np.array([[g_encoded, s_encoded, raised_hands, visited_resources, announcements, discussion]])
        prediction = model.predict(input_data)[0]
        st.markdown("---")
        if prediction == 'H':
            st.success("### 🎉 Predicted Class: HIGH Performance!")
        elif prediction == 'M':
            st.info("### 📊 Predicted Class: MEDIUM Performance.")
        else:
            st.warning("### ⚠️ Predicted Class: LOW Performance. Needs Attention.")
except FileNotFoundError:
    st.error("Error: 'AI-Data.csv' nahi mili!")
