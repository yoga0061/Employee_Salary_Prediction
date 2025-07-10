# app.py
import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Page configuration
st.set_page_config(
    page_title="AI-Salary Predictor ",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Dark Theme CSS
st.markdown("""
    <style>
        :root {
            --primary: #6c5ce7;
            --secondary: #a29bfe;
            --accent: #fd79a8;
            --dark-bg: #0f0e17;
            --dark-card: #1e1e2e;
            --dark-text: #fffffe;
            --dark-subtext: #a7a9be;
        }
        
        body {
            color: var(--dark-text);
        }
        
        .stApp {
            background-color: var(--dark-bg);
        }
        
        .stForm {
            background-color: var(--dark-card);
            border-radius: 15px;
            padding: 2rem;
            border: 1px solid #2e2e3a;
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
            box-shadow: 0 4px 12px rgba(108, 92, 231, 0.3);
        }
        
        .prediction-box {
            border-radius: 12px;
            padding: 2rem;
            margin: 1.5rem 0;
            background-color: var(--dark-card);
            border: 1px solid #2e2e3a;
        }
        
        .feature-importance {
            background-color: var(--dark-card);
            border-radius: 10px;
            padding: 1.5rem;
            margin-top: 2rem;
            border: 1px solid #2e2e3a;
        }
        
        .header {
            color: var(--primary);
            border-bottom: 2px solid var(--accent);
            padding-bottom: 0.5rem;
            margin-bottom: 1.5rem;
        }
        
        .sidebar .sidebar-content {
            background: linear-gradient(180deg, #121212, #1e1e1e);
            color: var(--dark-text);
        }
        
        .stSelectbox, .stRadio, .stSlider, .stNumberInput {
            background-color: var(--dark-card);
            border: 1px solid #2e2e3a;
            border-radius: 8px;
            padding: 8px 12px;
        }
        
        .stTextInput>div>div>input {
            background-color: var(--dark-card);
            color: var(--dark-text);
            border: 1px solid #2e2e3a;
        }
        
        .stSpinner>div {
            border-color: var(--primary) transparent transparent transparent;
        }
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: var(--dark-bg);
        }
        
        ::-webkit-scrollbar-thumb {
            background: var(--primary);
            border-radius: 4px;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: var(--dark-card);
            color: var(--dark-subtext);
            border-radius: 8px 8px 0 0;
            padding: 10px 20px;
            transition: all 0.3s;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: var(--primary);
            color: white;
        }
        
        /* Tooltip styling */
        .stTooltip {
            background-color: var(--dark-card) !important;
            color: var(--dark-text) !important;
            border: 1px solid #2e2e3a !important;
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
    <p style='color:var(--dark-subtext)'>
    Predict income levels using advanced machine learning
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🔍 Model Details")
    st.markdown("""
    <p style='color:var(--dark-subtext)'>
    - Algorithm: Random Forest<br>
    - Accuracy: 85%<br>
    - Trained on US Census data
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🛠️ How To Use")
    st.markdown("""
    <p style='color:var(--dark-subtext)'>
    1. Fill in the form<br>
    2. Click Predict<br>
    3. View results
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("<p style='color:var(--dark-subtext)'>Built with ❤️ using Streamlit</p>", unsafe_allow_html=True)

# Main content
st.title("💼 Salary Predictor Pro")
st.markdown("<p style='color:var(--dark-subtext)'>Predict income levels based on demographic and employment factors</p>", unsafe_allow_html=True)

# Form in two columns with tabs
tab1, tab2 = st.tabs(["📝 Input Form", "📊 Model Info"])

with tab1:
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 👤 Personal Details")
            age = st.slider("Age", 17, 90, 30, 
                           help="Select the individual's age")
            gender = st.radio("Gender", 
                             options=["Female", "Male"], 
                             help="Select gender identity",
                             horizontal=True)
            marital_status = st.selectbox("Marital Status", 
                                        options=["Married", "Single", "Divorced", "Widowed", "Separated"])
            relationship = st.selectbox("Relationship Status", 
                                      options=["Husband", "Wife", "Own-child", "Unmarried", "Other-relative"])
            race = st.selectbox("Race", 
                              options=["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"])
            
        with col2:
            st.markdown("### 💼 Employment Info")
            workclass = st.selectbox("Employment Sector", 
                                   options=["Private", "Government", "Self-employed", "Non-profit", "Other"])
            occupation = st.selectbox("Occupation", 
                                   options=["Tech", "Admin", "Services", "Professional", "Manual-labor", "Other"])
            education = st.selectbox("Highest Education", 
                                   options=["HS-grad", "Bachelors", "Masters", "Doctorate", "Some-college", "Other"])
            education_num = st.slider("Years of Education", 1, 20, 10)
            hours_per_week = st.slider("Weekly Work Hours", 10, 100, 40)
            native_country = st.selectbox("Country of Origin", 
                                        options=["United-States", "Mexico", "India", "Philippines", "Germany", "Other"])
            
            st.markdown("### 💰 Financial Data")
            capital_gain = st.number_input("Capital Gains ($)", min_value=0, value=0)
            capital_loss = st.number_input("Capital Losses ($)", min_value=0, value=0)
            fnlwgt = st.number_input("Final Weight", min_value=0, value=100000)
        
        submitted = st.form_submit_button("🔮 Predict Income", use_container_width=True)

with tab2:
    st.markdown("### 🧠 About the Model")
    st.markdown("""
    <p style='color:var(--dark-subtext)'>
    This machine learning model was trained on census data to predict whether an individual's income exceeds $50K/year.
    </p>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Key Features")
        st.markdown("""
        <p style='color:var(--dark-subtext)'>
        - Age<br>
        - Education Level<br>
        - Occupation<br>
        - Work Hours<br>
        - Capital Gains
        </p>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### ⚠️ Limitations")
        st.markdown("""
        <p style='color:var(--dark-subtext)'>
        - Statistical estimates only<br>
        - May not reflect all circumstances<br>
        - Training data limitations
        </p>
        """, unsafe_allow_html=True)
    
    st.markdown("### 📊 Performance Metrics")
    st.markdown("""
    ```python
    Accuracy: 85.2%
    Precision: 0.83
    Recall: 0.62
    F1 Score: 0.71
    ```
    """)

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
            
            # Display prediction with styling
            if prediction == 1:
                st.markdown(f"""
                <div class="prediction-box" style='border-left: 6px solid var(--primary);'>
                    <h2 style='color:var(--primary); margin-top:0;'>💰 High Income Prediction</h2>
                    <p style='font-size:1.2rem; color:var(--dark-text);'>This individual is likely earning <strong>>$50K/year</strong></p>
                    <div style='background-color:#2e2e3a; border-radius:8px; padding:1rem; margin:1rem 0;'>
                        <p style='margin:0; color:var(--dark-subtext);'><strong>Confidence:</strong> {probability*100:.1f}%</p>
                        <div style='height:10px; background-color:#1e1e2e; border-radius:5px; margin-top:0.5rem;'>
                            <div style='width:{probability*100}%; height:100%; background-color:var(--primary); border-radius:5px;'></div>
                        </div>
                    </div>
                    <p style='color:var(--dark-subtext);'>Key contributing factors:</p>
                    <ul style='color:var(--dark-subtext);'>
                        <li>Education level</li>
                        <li>Occupation type</li>
                        <li>Work experience</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-box" style='border-left: 6px solid #ff7675;'>
                    <h2 style='color:#ff7675; margin-top:0;'>💰 Moderate Income Prediction</h2>
                    <p style='font-size:1.2rem; color:var(--dark-text);'>This individual is likely earning <strong>≤$50K/year</strong></p>
                    <div style='background-color:#2e2e3a; border-radius:8px; padding:1rem; margin:1rem 0;'>
                        <p style='margin:0; color:var(--dark-subtext);'><strong>Confidence:</strong> {(1-probability)*100:.1f}%</p>
                        <div style='height:10px; background-color:#1e1e2e; border-radius:5px; margin-top:0.5rem;'>
                            <div style='width:{(1-probability)*100}%; height:100%; background-color:#ff7675; border-radius:5px;'></div>
                        </div>
                    </div>
                    <p style='color:var(--dark-subtext);'>Potential influencing factors:</p>
                    <ul style='color:var(--dark-subtext);'>
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
                <div style='background-color:#2d3436; padding:1.5rem; border-radius:10px; border-left: 4px solid var(--primary);'>
                    <h4 style='margin-top:0; color:var(--primary);'>For High Earners:</h4>
                    <ul style='color:var(--dark-subtext);'>
                        <li>Tax optimization strategies</li>
                        <li>Investment portfolio review</li>
                        <li>Professional development</li>
                        <li>Retirement planning</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='background-color:#2d3436; padding:1.5rem; border-radius:10px; border-left: 4px solid #ff7675;'>
                    <h4 style='margin-top:0; color:#ff7675;'>For Income Growth:</h4>
                    <ul style='color:var(--dark-subtext);'>
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
            <div style='background-color:#2d3436; padding:1rem; border-radius:8px; border-left: 4px solid #ff7675;'>
                <p style='color:var(--dark-text);'>Please check your inputs and try again.</p>
                <p style='color:var(--dark-subtext);'>Ensure all fields are filled correctly.</p>
            </div>
            """, unsafe_allow_html=True)
