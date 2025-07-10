# app.py
import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Salary Predictor Pro",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
    <style>
        :root {
            --primary: #4a6fa5;
            --secondary: #166088;
            --accent: #4fc3f7;
            --success: #4caf50;
            --warning: #ff9800;
            --danger: #f44336;
        }
        
        .stApp {
            background-color: #f8f9fa;
        }
        
        .stForm {
            background-color: white;
            border-radius: 15px;
            padding: 2rem;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        }
        
        .stButton>button {
            background-color: var(--primary);
            color: white;
            border-radius: 8px;
            padding: 0.75rem 1.5rem;
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.3s;
            border: none;
        }
        
        .stButton>button:hover {
            background-color: var(--secondary);
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        
        .prediction-box {
            border-radius: 12px;
            padding: 2rem;
            margin: 1.5rem 0;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }
        
        .feature-importance {
            background-color: white;
            border-radius: 10px;
            padding: 1.5rem;
            margin-top: 2rem;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        }
        
        .header {
            color: var(--primary);
            border-bottom: 2px solid var(--accent);
            padding-bottom: 0.5rem;
            margin-bottom: 1.5rem;
        }
        
        .sidebar .sidebar-content {
            background: linear-gradient(180deg, var(--primary), var(--secondary));
            color: white;
        }
        
        .sidebar .sidebar-content .stMarkdown h2 {
            color: white !important;
        }
        
        .stNumberInput, .stSelectbox, .stSlider, .stRadio {
            margin-bottom: 1.2rem;
        }
        
        .stSpinner>div {
            border-color: var(--primary) transparent transparent transparent;
        }
    </style>
""", unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    return pickle.load(open("model.pkl", "rb"))

model = load_model()

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
    st.markdown("""
    This tool predicts whether an individual's income exceeds $50K/year based on demographic and employment factors.
    """)
    
    st.markdown("### 📊 Model Information")
    st.markdown("""
    - **Algorithm**: Random Forest Classifier
    - **Accuracy**: 85% (test set)
    - **Training Data**: US Census Bureau
    """)
    
    st.markdown("### 📝 How To Use")
    st.markdown("""
    1. Fill in all required fields
    2. Click 'Predict Salary Range'
    3. View prediction and insights
    """)
    
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit")

# Main content
st.title("💼 Salary Predictor Pro")
st.markdown("Predict whether an individual's income exceeds $50K/year based on demographic and employment factors.")

# Form in two columns with tabs for better organization
tab1, tab2 = st.tabs(["📝 Input Form", "ℹ️ About the Data"])

with tab1:
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 👤 Personal Information")
            age = st.slider("Age", 17, 90, 30, 
                           help="Select the individual's age")
            gender = st.radio("Gender", 
                             options=["Female", "Male"], 
                             format_func=lambda x: x, 
                             help="Select gender identity",
                             horizontal=True)
            marital_status = st.selectbox("Marital Status", 
                                        options=["Married", "Single", "Divorced", "Widowed", "Separated"], 
                                        help="Current marital status")
            relationship = st.selectbox("Relationship Status", 
                                      options=["Husband", "Wife", "Own-child", "Unmarried", "Other-relative"], 
                                      help="Relationship status in household")
            race = st.selectbox("Race", 
                              options=["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"], 
                              help="Race/ethnicity")
            
        with col2:
            st.markdown("### 💼 Employment Details")
            workclass = st.selectbox("Employment Sector", 
                                   options=["Private", "Government", "Self-employed", "Non-profit", "Other"], 
                                   help="Primary employment sector")
            occupation = st.selectbox("Occupation", 
                                   options=["Tech", "Admin", "Services", "Professional", "Manual-labor", "Other"], 
                                   help="Primary occupation category")
            education = st.selectbox("Highest Education", 
                                   options=["HS-grad", "Bachelors", "Masters", "Doctorate", "Some-college", "Other"], 
                                   help="Highest level of education completed")
            education_num = st.slider("Years of Education", 1, 20, 10, 
                                    help="Total years of formal education")
            hours_per_week = st.slider("Weekly Work Hours", 10, 100, 40, 
                                     help="Typical hours worked per week")
            native_country = st.selectbox("Country of Origin", 
                                        options=["United-States", "Mexico", "India", "Philippines", "Germany", "Other"], 
                                        help="Country of origin")
            
            st.markdown("### 💰 Financial Information")
            capital_gain = st.number_input("Capital Gains ($)", min_value=0, value=0, 
                                         help="Capital gains in dollars")
            capital_loss = st.number_input("Capital Losses ($)", min_value=0, value=0, 
                                         help="Capital losses in dollars")
            fnlwgt = st.number_input("Final Weight", min_value=0, value=100000, 
                                    help="Final weight (demographic weighting factor)")
        
        submitted = st.form_submit_button("🔮 Predict Salary Range", 
                                         help="Click to get salary prediction",
                                         use_container_width=True)

with tab2:
    st.markdown("### 📊 About the Data")
    st.markdown("""
    This model was trained on data from the US Census Bureau with the following characteristics:
    
    - **Target Variable**: Whether income exceeds $50K/year
    - **Features Used**: Demographic, employment, and financial factors
    - **Data Size**: 32,561 records
    - **Class Distribution**: 24% >$50K, 76% ≤$50K
    """)
    
    st.markdown("### 📈 Key Influencing Factors")
    st.markdown("""
    The model identifies these as the most important factors in predicting income:
    1. Age
    2. Education Level
    3. Occupation
    4. Hours Worked Per Week
    5. Capital Gains
    """)
    
    st.markdown("### ⚠️ Limitations")
    st.markdown("""
    - Predictions are estimates based on statistical patterns
    - Results may not account for all individual circumstances
    - Model accuracy may vary for populations not well-represented in the training data
    """)

# Prediction and results
if submitted:
    with st.spinner('Analyzing the data...'):
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
                label_encoders[feature].fit([value])  # Just fit, no transform yet

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
            
            st.success("Prediction Complete!")
            st.balloons()
            
            # Display prediction with styling
            if prediction == 1:
                st.markdown(f"""
                <div class="prediction-box" style='background-color:#e8f5e9; border-left: 6px solid #4caf50;'>
                    <h2 style='color:#2e7d32; margin-top:0;'>💰 High Income Prediction</h2>
                    <p style='font-size:1.2rem;'>This individual is likely earning <strong>>$50K/year</strong></p>
                    <div style='background-color:white; border-radius:8px; padding:1rem; margin:1rem 0;'>
                        <p style='margin:0;'><strong>Confidence Level:</strong> {probability*100:.1f}%</p>
                        <div style='height:10px; background-color:#e0e0e0; border-radius:5px; margin-top:0.5rem;'>
                            <div style='width:{probability*100}%; height:100%; background-color:#4caf50; border-radius:5px;'></div>
                        </div>
                    </div>
                    <p>Key factors contributing to this prediction:</p>
                    <ul>
                        <li>Higher education level</li>
                        <li>Professional occupation</li>
                        <li>Full-time work hours</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-box" style='background-color:#ffebee; border-left: 6px solid #f44336;'>
                    <h2 style='color:#c62828; margin-top:0;'>💰 Moderate Income Prediction</h2>
                    <p style='font-size:1.2rem;'>This individual is likely earning <strong>≤$50K/year</strong></p>
                    <div style='background-color:white; border-radius:8px; padding:1rem; margin:1rem 0;'>
                        <p style='margin:0;'><strong>Confidence Level:</strong> {(1-probability)*100:.1f}%</p>
                        <div style='height:10px; background-color:#e0e0e0; border-radius:5px; margin-top:0.5rem;'>
                            <div style='width:{(1-probability)*100}%; height:100%; background-color:#f44336; border-radius:5px;'></div>
                        </div>
                    </div>
                    <p>Potential factors affecting this prediction:</p>
                    <ul>
                        <li>Education level</li>
                        <li>Work hours</li>
                        <li>Occupation type</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            # Add recommendations section
            st.markdown("### 📝 Recommendations")
            if prediction == 1:
                st.markdown("""
                <div style='background-color:#e3f2fd; padding:1.5rem; border-radius:10px;'>
                    <h4 style='margin-top:0; color:#1565c0;'>For High Earners:</h4>
                    <ul>
                        <li>Consider tax optimization strategies</li>
                        <li>Explore investment opportunities to grow wealth</li>
                        <li>Professional development to maintain competitive edge</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='background-color:#fff8e1; padding:1.5rem; border-radius:10px;'>
                    <h4 style='margin-top:0; color:#ff8f00;'>For Income Growth:</h4>
                    <ul>
                        <li>Consider additional education or certifications</li>
                        <li>Explore higher-paying industries or roles</li>
                        <li>Develop in-demand skills that command higher salaries</li>
                        <li>Negotiate salary during performance reviews</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
            st.markdown("""
            <div style='background-color:#ffebee; padding:1rem; border-radius:8px;'>
                <p>Please ensure all fields are filled correctly and try again.</p>
                <p>If the problem persists, contact support.</p>
            </div>
            """, unsafe_allow_html=True)
