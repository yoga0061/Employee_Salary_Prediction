# app.py
import streamlit as st
import pandas as pd
import pickle
from PIL import Image

# Load the trained model
@st.cache_resource
def load_model():
    return pickle.load(open("model.pkl", "rb"))

model = load_model()

# Page configuration
st.set_page_config(
    page_title="AI Salary Insights",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
        border: none;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stSuccess {
        font-size: 18px !important;
    }
    .header {
        color: #2c3e50;
    }
    .sidebar .sidebar-content {
        background-color: #e8f4f8;
    }
    .stSelectbox, .stSlider, .stRadio, .stTextInput, .stNumberInput {
        margin-bottom: 1.5rem;
    }
    .prediction-box {
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Header section
col1, col2 = st.columns([1, 5])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/3132/3132693.png", width=80)
with col2:
    st.title("AI Salary Insights Dashboard")
    st.markdown("Predict salary ranges based on demographic and employment factors.")

# Information expander
with st.expander("ℹ️ About this dashboard", expanded=True):
    st.write("""
    This predictive tool estimates whether an individual's annual salary exceeds $50,000
    based on various factors including education, work experience, and demographic information.
    """)
    st.write("The model was trained on US Census Bureau data using machine learning algorithms.")

# Main form in two columns
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Personal Information")
        age = st.slider("Age", 17, 90, 30, help="Select the individual's age")
        gender = st.radio("Gender", options=["Female", "Male"], format_func=lambda x: x, help="Select gender identity")
        marital_status = st.selectbox("Marital Status", options=["Married", "Single", "Divorced", "Widowed", "Separated"], help="Current marital status")
        relationship = st.selectbox("Relationship Status", options=["Husband", "Wife", "Own-child", "Unmarried", "Other-relative"], help="Relationship status in household")
        race = st.selectbox("Race", options=["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"], help="Race/ethnicity")
    with col2:
        st.subheader("Employment Details")
        workclass = st.selectbox("Employment Sector", options=["Private", "Government", "Self-employed", "Non-profit", "Other"], help="Primary employment sector")
        occupation = st.selectbox("Occupation", options=["Tech", "Admin", "Services", "Professional", "Manual-labor", "Other"], help="Primary occupation category")
        education = st.selectbox("Highest Education", options=["HS-grad", "Bachelors", "Masters", "Doctorate", "Some-college", "Other"], help="Highest level of education completed")
        education_num = st.slider("Years of Education", 1, 20, 10, help="Total years of formal education")
        hours_per_week = st.slider("Weekly Work Hours", 10, 100, 40, help="Typical hours worked per week")
        native_country = st.selectbox("Country of Origin", options=["United-States", "Mexico", "India", "Philippines", "Germany", "Other"], help="Country of origin")
        capital_gain = st.number_input("Capital Gains", min_value=0, value=0, help="Capital gains")
        capital_loss = st.number_input("Capital Losses", min_value=0, value=0, help="Capital losses")
        fnlwgt = st.number_input("Final Weight", min_value=0, value=100000, help="Final weight")

    submitted = st.form_submit_button("Predict Salary Range")

# Prediction and results
if submitted:
    # Convert inputs to encoded values
    gender_encoded = 1 if gender == "Male" else 0

    # Create input data DataFrame with all expected features
    input_data = pd.DataFrame([[
        age, workclass, education, education_num, marital_status,
        occupation, relationship, race, gender_encoded, hours_per_week,
        native_country, capital_gain, capital_loss, fnlwgt
    ]], columns=[
        'age', 'workclass', 'education', 'educational-num', 'marital-status',
        'occupation', 'relationship', 'race', 'gender', 'hours-per-week',
        'native-country', 'capital-gain', 'capital-loss', 'fnlwgt'
    ])

    with st.spinner('Analyzing the data...'):
        try:
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]
            st.success("Prediction Complete!")
            st.balloons()
            if prediction == 1:
                st.markdown(f"""
                <div class="prediction-box" style='background-color:#e8f5e9;'>
                    <h3 style='color:#2e7d32'>💰 Prediction: >$50K/year</h3>
                    <p>Confidence: {probability*100:.1f}%</p>
                    <p>This individual is likely earning more than $50,000 annually based on the provided information.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-box" style='background-color:#ffebee;'>
                    <h3 style='color:#c62828'>💰 Prediction: ≤$50K/year</h3>
                    <p>Confidence: {(1-probability)*100:.1f}%</p>
                    <p>This individual is likely earning $50,000 or less annually based on the provided information.</p>
                </div>
                """, unsafe_allow_html=True)
        except ValueError as e:
            st.error(f"Error during prediction: {e}")
            st.write("Please ensure all input fields are correctly filled and match the expected format.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; font-size: 14px;'>
    <p>This tool provides estimates only and should not be used for official purposes.</p>
    <p>Model accuracy: 85% | Last updated: June 2023</p>
</div>
""", unsafe_allow_html=True)
