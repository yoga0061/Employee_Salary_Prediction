import streamlit as st
import pandas as pd
import pickle
from PIL import Image

# Load the model
@st.cache_data
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

# Page configuration
st.set_page_config(
    page_title="AI Salary Insights",
    page_icon="💼",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
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
    </style>
    """, unsafe_allow_html=True)

# Header
col1, col2 = st.columns([1, 4])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/3132/3132693.png", width=80)
with col2:
    st.title("🧠 AI Salary Insights Dashboard")
    st.markdown("Predict salary ranges based on demographic and employment factors")

# Information expander
with st.expander("ℹ️ About this dashboard"):
    st.write("""
    This predictive tool estimates whether an individual's annual salary exceeds $50,000 
    based on various factors including education, work experience, and demographic information.
    """)
    st.write("The model was trained on US Census Bureau data using machine learning algorithms.")

# Main form
st.header("📋 Enter Your Details")
with st.form("user_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Personal Information")
        age = st.slider("Age", 17, 90, 30, help="Select your current age")
        gender = st.selectbox("Gender", ["Male", "Female"], help="Select your gender identity")
        marital_status = st.selectbox("Marital Status", 
                                    ["Married", "Single", "Divorced", "Widowed"],
                                    help="Your current marital status")
        relationship = st.selectbox("Relationship Status", 
                                  ["Husband", "Wife", "Not-in-family", "Own-child", "Unmarried", "Other-relative"],
                                  help="Your relationship in household")
        race = st.selectbox("Race", 
                          ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"],
                          help="Your race/ethnicity")
    
    with col2:
        st.subheader("Employment Details")
        workclass = st.selectbox("Employment Sector", 
                               ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", 
                                "Local-gov", "State-gov", "Without-pay", "Never-worked"],
                               help="Your primary employment sector")
        occupation = st.selectbox("Occupation", 
                                ["Tech", "Exec-managerial", "Craft-repair", "Sales", 
                                 "Other-service", "Machine-op-inspct"],
                                help="Your primary occupation")
        education = st.selectbox("Highest Education", 
                               ["Bachelors", "HS-grad", "Some-college", "Masters", 
                                "Assoc-acdm", "Assoc-voc", "Doctorate"],
                               help="Your highest education level")
        education_num = st.slider("Years of Education", 1, 20, 10, 
                                help="Total years of formal education")
        hours_per_week = st.slider("Weekly Work Hours", 10, 100, 40, 
                                 help="Your typical work hours per week")
        native_country = st.selectbox("Country of Origin", 
                                    ["India", "United-States", "Philippines", 
                                     "Germany", "Canada", "Mexico", "Other"],
                                    help="Your country of origin")

    submitted = st.form_submit_button("Predict Income Level")

# Prediction and results
if submitted:
    # Encode categorical features (in a real app, use proper encoding)
    gender_encoded = 1 if gender == "Male" else 0
    
    input_data = pd.DataFrame([{
        'age': age,
        'workclass': workclass,
        'education': education,
        'education-num': education_num,
        'marital-status': marital_status,
        'occupation': occupation,
        'relationship': relationship,
        'race': race,
        'gender': gender_encoded,
        'hours-per-week': hours_per_week,
        'native-country': native_country
    }])

    with st.spinner('Analyzing your information...'):
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        
        st.success("")
        st.balloons()
        
        if prediction == 1:
            st.markdown(f"""
            <div style='background-color:#e8f5e9; padding:20px; border-radius:10px;'>
                <h3 style='color:#2e7d32'>✅ Predicted Income: >$50K/year</h3>
                <p>Confidence: {probability*100:.1f}%</p>
                <p>Based on your information, you're likely earning more than $50,000 annually.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='background-color:#ffebee; padding:20px; border-radius:10px;'>
                <h3 style='color:#c62828'>✅ Predicted Income: ≤$50K/year</h3>
                <p>Confidence: {(1-probability)*100:.1f}%</p>
                <p>Based on your information, you're likely earning $50,000 or less annually.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Insights
        with st.expander("📊 See what influenced this prediction"):
            st.write("Key factors in this prediction:")
            if age > 40:
                st.write("- Older age typically correlates with higher earnings")
            if education in ["Masters", "Doctorate"]:
                st.write("- Advanced degrees significantly increase earning potential")
            if hours_per_week > 45:
                st.write("- Longer work hours may indicate higher compensation")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; font-size: 14px;'>
    <p>This tool provides estimates only and should not be used for official purposes.</p>
    <p>Model accuracy: 85% | Last updated: June 2023</p>
</div>
""", unsafe_allow_html=True)
