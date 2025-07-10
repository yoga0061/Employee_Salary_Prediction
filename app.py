import streamlit as st
import pandas as pd
import pickle
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="AI Salary Insights",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
    <style>
        .stApp {
            background-color: #f5f5f5;
        }
        .stForm {
            background-color: white;
            border-radius: 10px;
            padding: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            padding: 0.75rem 1rem;
            width: 100%;
            font-size: 1rem;
            transition: all 0.3s;
        }
        .stButton>button:hover {
            background-color: #45a049;
            transform: translateY(-2px);
        }
        .success {
            font-size: 1.2rem !important;
            text-align: center;
            padding: 1rem;
            border-radius: 5px;
        }
        .title {
            color: #2c3e50;
            text-align: center;
        }
        .header {
            color: #3498db;
            border-bottom: 2px solid #3498db;
            padding-bottom: 0.5rem;
        }
        .sidebar .sidebar-content {
            background-color: #2c3e50;
            color: white;
        }
        .sidebar .sidebar-content p, .sidebar .sidebar-content li {
            color: white;
        }
        .prediction-card {
            background-color: #e8f5e9;
            border-radius: 10px;
            padding: 2rem;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin: 1rem auto;
        }
    </style>
""", unsafe_allow_html=True)

# Load the model
@st.cache_data
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

# Sidebar
with st.sidebar:
    st.markdown("## About")
    st.markdown("""
    This AI Salary Insights Dashboard predicts income levels based on demographic and employment factors.
    """)
    st.markdown("## How to Use")
    st.markdown("""
    1. Fill in your details
    2. Click 'Predict Income Level'
    3. View your predicted income
    """)
    st.markdown("---")
    st.markdown("Made with ❤️ by [Your Name]")

# Main content
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.title("🧠 AI Salary Insights Dashboard")
    st.markdown("### Predict your potential income based on key factors")

st.markdown("---")

# Input form
with st.form("user_form"):
    st.header("📋 Personal Information")
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 17, 90, 30, help="Select your current age")
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        marital_status = st.selectbox("Marital Status", ["Married", "Single", "Divorced", "Widowed"])
        relationship = st.selectbox("Relationship Status", ["Husband", "Wife", "Not-in-family", "Own-child", "Unmarried", "Other-relative"])
        race = st.selectbox("Race", ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"])

    with col2:
        workclass = st.selectbox("Employment Sector", ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"])
        occupation = st.selectbox("Occupation", ["Tech", "Exec-managerial", "Craft-repair", "Sales", "Other-service", "Machine-op-inspct"])
        education = st.selectbox("Highest Education", ["Bachelors", "HS-grad", "Some-college", "Masters", "Assoc-acdm", "Assoc-voc", "Doctorate"])
        education_num = st.slider("Years of Education", 1, 20, 12)
        hours_per_week = st.slider("Weekly Work Hours", 10, 100, 40)
        native_country = st.selectbox("Country of Origin", ["India", "United-States", "Philippines", "Germany", "Canada", "Mexico", "Other"])

    submitted = st.form_submit_button("🔮 Predict Income Level", help="Click to get your income prediction")

# Predict and display results
if submitted:
    st.balloons()

    # Match model input structure
    input_data = pd.DataFrame([{
        'age': age,
        'workclass': workclass,
        'education': education,
        'education-num': education_num,
        'marital-status': marital_status,
        'occupation': occupation,
        'relationship': relationship,
        'race': race,
        'sex': gender,
        'hours-per-week': hours_per_week,
        'native-country': native_country
    }])

    # Predict
    prediction = model.predict(input_data)[0]

    # Display results in a nice card
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown(f"""
        <div class="prediction-card">
            <h3 style='color: #2e7d32;'>Prediction Result</h3>
            <p style='font-size: 1.5rem;'>Your predicted income is:</p>
            <h2 style='color: #1b5e20;'>💰 {prediction} 💰</h2>
            <p style='font-size: 0.9rem; color: #666;'>This is an estimate based on the information provided.</p>
        </div>
        """, unsafe_allow_html=True)

    # Add some additional insights
    st.markdown("---")
    st.markdown("### 💡 Insights & Recommendations")

    if ">50K" in prediction:
        st.success("Based on your profile, you're likely in a higher income bracket. Consider investment opportunities to grow your wealth further.")
    else:
        st.info("Based on your profile, here are some ways to potentially increase your income:")
        st.markdown("""
        - Consider additional education or certifications in your field
        - Explore higher-paying industries or roles
        - Develop in-demand skills that command higher salaries
        - Negotiate your salary during performance reviews
        """)
