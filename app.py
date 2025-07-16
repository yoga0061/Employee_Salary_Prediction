import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Page configuration
st.set_page_config(
    page_title="Salary Predictor Pro",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        .stButton>button {
            background-color: #6c757d;
            color: white;
            padding: 10px 24px;
            border: none;
            border-radius: 4px;
        }
        .stButton>button:hover {
            background-color: #5a6268;
        }
        .stSelectbox, .stSlider, .stNumberInput, .stRadio, .stTextInput {
            margin-bottom: 1rem;
        }
        .prediction-box {
            border: 1px solid #dee2e6;
            border-radius: 5px;
            padding: 15px;
            margin: 10px 0;
            background-color: #ffffff;
        }
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #343a40;
            color: white;
            text-align: center;
            padding: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    return pickle.load(open("model.pkl", "rb"))

model = load_model()

# Exchange rate (example: 1 USD = 83 INR)
USD_TO_INR = 83

# Feature order and encoders
correct_feature_order = [
    'age', 'workclass', 'fnlwgt', 'education', 'educational-num',
    'marital-status', 'occupation', 'relationship', 'race', 'gender',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country'
]
label_encoders = {feature: LabelEncoder() for feature in correct_feature_order
                 if feature not in ['age', 'fnlwgt', 'educational-num', 'capital-gain',
                                  'capital-loss', 'hours-per-week']}

# Sidebar with additional info
with st.sidebar:
    st.markdown("## 💼 Salary Predictor Pro")
    st.markdown("Predict income levels using advanced machine learning.")

    st.markdown("### 🔍 Model Details")
    st.markdown("- Algorithm: Random Forest\n- Accuracy: 85%\n- Trained on US Census data")

    st.markdown("### 🛠️ How To Use")
    st.markdown("1. Fill in the form\n2. Click Predict\n3. View results")

    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit")

# Main content
st.title("💼 Salary Predictor Pro")
st.markdown("Predict income levels based on demographic and employment factors")

# Form in two columns with tabs
tab1, tab2 = st.tabs(["📝 Input Form", "📊 Model Info"])

with tab1:
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 👤 Personal Details")
            age = st.slider("Age", 17, 90, 30, help="Select the individual's age")
            gender = st.radio("Gender", options=["Female", "Male"], help="Select gender identity", horizontal=True)
            marital_status = st.selectbox("Marital Status", options=["Married", "Single", "Divorced", "Widowed", "Separated"])
            relationship = st.selectbox("Relationship Status", options=["Husband", "Wife", "Own-child", "Unmarried", "Other-relative"])
            race = st.selectbox("Race", options=["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"])

        with col2:
            st.markdown("### 💼 Employment Info")
            workclass = st.selectbox("Employment Sector", options=["Private", "Government", "Self-employed", "Non-profit", "Other"])
            occupation = st.selectbox("Occupation", options=["Tech", "Admin", "Services", "Professional", "Manual-labor", "Other"])
            education = st.selectbox("Highest Education", options=["HS-grad", "Bachelors", "Masters", "Doctorate", "Some-college", "Other"])
            education_num = st.slider("Years of Education", 1, 20, 10)
            hours_per_week = st.slider("Weekly Work Hours", 10, 100, 40)
            native_country = st.selectbox("Country of Origin", options=["United-States", "Mexico", "India", "Philippines", "Germany", "Other"])

            st.markdown("### 💰 Financial Data")
            capital_gain = st.number_input("Capital Gains ($)", min_value=0, value=0)
            capital_loss = st.number_input("Capital Losses ($)", min_value=0, value=0)
            fnlwgt = st.number_input("Final Weight", min_value=0, value=100000)

        submitted = st.form_submit_button("🔮 Predict Income", use_container_width=True)

# Prediction and results
if submitted:
    with st.spinner('Analyzing data...'):
        try:
            # Convert inputs to encoded values
            gender_encoded = 1 if gender == "Male" else 0
            # Encode categorical features
            categorical_features = {
                'workclass': workclass,
                'education': education,
                'marital-status': marital_status,
                'occupation': occupation,
                'relationship': relationship,
                'race': race,
                'native-country': native_country
            }
            for feature, value in categorical_features.items():
                label_encoders[feature].fit([value])
            # Create input data DataFrame
            input_data = pd.DataFrame([[
                age,
                label_encoders['workclass'].transform([workclass])[0],
                fnlwgt,
                label_encoders['education'].transform([education])[0],
                education_num,
                label_encoders['marital-status'].transform([marital_status])[0],
                label_encoders['occupation'].transform([occupation])[0],
                label_encoders['relationship'].transform([relationship])[0],
                label_encoders['race'].transform([race])[0],
                gender_encoded,
                capital_gain,
                capital_loss,
                hours_per_week,
                label_encoders['native-country'].transform([native_country])[0]
            ]], columns=correct_feature_order)
            # Make prediction
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]

            st.success("Analysis Complete!")
            st.balloons()

            # Display prediction with styling and INR conversion
            if prediction == 1:
                inr_amount = 50000 * USD_TO_INR
                st.markdown(f"""
                <div class="prediction-box">
                    <h2>💰 High Income Prediction</h2>
                    <p>This individual is likely earning <strong>>$50K/year (₹{inr_amount:,.0f}/year)</strong></p>
                    <p><strong>Confidence:</strong> {probability*100:.1f}%</p>
                    <p>Key contributing factors:</p>
                    <ul>
                        <li>Education level</li>
                        <li>Occupation type</li>
                        <li>Work experience</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                inr_amount = 50000 * USD_TO_INR
                st.markdown(f"""
                <div class="prediction-box">
                    <h2>💰 Moderate Income Prediction</h2>
                    <p>This individual is likely earning <strong>≤$50K/year (≤₹{inr_amount:,.0f}/year)</strong></p>
                    <p><strong>Confidence:</strong> {(1-probability)*100:.1f}%</p>
                    <p>Potential influencing factors:</p>
                    <ul>
                        <li>Education level</li>
                        <li>Work hours</li>
                        <li>Industry sector</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            # Add recommendations section
            st.markdown("### 📝 Recommendations")
            if prediction == 1:
                st.markdown("""
                <div class="prediction-box">
                    <h4>For High Earners:</h4>
                    <ul>
                        <li>Tax optimization strategies</li>
                        <li>Investment portfolio review</li>
                        <li>Professional development</li>
                        <li>Retirement planning</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="prediction-box">
                    <h4>For Income Growth:</h4>
                    <ul>
                        <li>Additional education/certifications</li>
                        <li>Higher-paying industry exploration</li>
                        <li>Skill development</li>
                        <li>Salary negotiation tactics</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            st.markdown("""
            <div class="prediction-box">
                <p>Please check your inputs and try again.</p>
                <p>Ensure all fields are filled correctly.</p>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class="footer">
        <p>© 2025 <strong>AI Salary Insights Dashboard</strong> | Built with ❤️ by <strong>Yoganandha</strong></p>
        <p><em>Disclaimer:</em> Predictions are AI-based estimates and not guaranteed.</p>
        <p><small>Exchange Rate (FYI): 1 USD ≈ 83 INR</small></p>
    </div>
""", unsafe_allow_html=True)
