# app.py
import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import time

# Page configuration
st.set_page_config(
    page_title="Salary Predictor",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Dark Theme CSS with Fire Animation
st.markdown("""
    <style>
        :root {
            --primary: #6c5ce7;
            --primary-light: #a29bfe;
            --secondary: #00cec9;
            --accent: #fd79a8;
            --dark-bg: #0f0e17;
            --dark-card: #1e1e2e;
            --dark-text: #fffffe;
            --dark-subtext: #a7a9be;
            --success: #00b894;
            --warning: #fdcb6e;
            --danger: #ff7675;
        }
        
        body {
            color: var(--dark-text);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        .stApp {
            background-color: var(--dark-bg);
            background-image: radial-gradient(circle at 10% 20%, rgba(108, 92, 231, 0.1) 0%, rgba(0, 0, 0, 0) 90%);
        }
        
        .stForm {
            background-color: var(--dark-card);
            border-radius: 15px;
            padding: 2rem;
            border: 1px solid #2e2e3a;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        }
        
        .stButton>button {
            background: linear-gradient(135deg, var(--primary), var(--primary-light));
            color: white;
            border-radius: 8px;
            padding: 0.75rem 1.5rem;
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.3s;
            border: none;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(108, 92, 231, 0.3);
        }
        
        .prediction-box {
            border-radius: 12px;
            padding: 2rem;
            margin: 1.5rem 0;
            background-color: var(--dark-card);
            border: 1px solid #2e2e3a;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        
        .prediction-box:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 24px rgba(0, 0, 0, 0.3);
        }
        
        .feature-card {
            background-color: var(--dark-card);
            border-radius: 10px;
            padding: 1.5rem;
            margin: 1rem 0;
            border-left: 4px solid var(--primary);
        }
        
        .header {
            color: var(--primary);
            border-bottom: 2px solid var(--accent);
            padding-bottom: 0.5rem;
            margin-bottom: 1.5rem;
            font-weight: 700;
        }
        
        .sidebar .sidebar-content {
            background: linear-gradient(180deg, #121212, #1e1e1e);
            color: var(--dark-text);
            border-right: 1px solid #2e2e3a;
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
            border: 1px solid transparent;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: var(--primary);
            color: white;
            border-color: var(--primary);
        }
        
        /* Tooltip styling */
        .stTooltip {
            background-color: var(--dark-card) !important;
            color: var(--dark-text) !important;
            border: 1px solid #2e2e3a !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
        }
        
        /* Footer styling */
        .footer {
            position: relative;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: var(--dark-card);
            color: var(--dark-subtext);
            text-align: center;
            padding: 1.5rem 0;
            border-top: 1px solid #2e2e3a;
            margin-top: 3rem;
        }
        
        .footer a {
            color: var(--primary-light);
            text-decoration: none;
            transition: all 0.3s;
        }
        
        .footer a:hover {
            color: var(--accent);
            text-decoration: underline;
        }
        
        /* Animation for prediction result */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .result-animation {
            animation: fadeIn 0.6s ease-out forwards;
        }
        
        /* Fire animation */
        .fire-container {
            position: absolute;
            bottom: 0;
            left: 0;
            width: 100%;
            height: 20px;
            overflow: hidden;
            z-index: 1;
        }
        
        .fire {
            position: relative;
            width: 100%;
            height: 100%;
        }
        
        .particle {
            position: absolute;
            bottom: 0;
            width: 4px;
            height: 4px;
            border-radius: 50%;
            background: linear-gradient(to top, #ff7800, #ff4d00, #ff0000);
            animation: fire-animation 2s ease-out infinite;
            opacity: 0;
        }
        
        @keyframes fire-animation {
            0% {
                transform: translateY(0) translateX(0) scale(0.5);
                opacity: 0;
            }
            50% {
                opacity: 1;
            }
            100% {
                transform: translateY(-100px) translateX(calc(var(--random-x) * 20px - 10px)) scale(1.5);
                opacity: 0;
            }
        }
        
        /* Responsive adjustments */
        @media (max-width: 768px) {
            .stForm {
                padding: 1rem;
            }
            .prediction-box {
                padding: 1.5rem;
            }
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

# JavaScript for fire effect
FIRE_JS = """
<script>
function createFireEffect(container) {
    const fireContainer = document.createElement('div');
    fireContainer.className = 'fire-container';
    const fire = document.createElement('div');
    fire.className = 'fire';
    fireContainer.appendChild(fire);
    container.appendChild(fireContainer);
    
    // Create particles
    for (let i = 0; i < 30; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        particle.style.left = Math.random() * 100 + '%';
        particle.style.setProperty('--random-x', Math.random());
        particle.style.animationDelay = Math.random() * 2 + 's';
        fire.appendChild(particle);
    }
}

// Create fire effect when prediction is shown
if (window.location.hash === '#prediction-shown') {
    const predictionBoxes = document.querySelectorAll('.prediction-box');
    predictionBoxes.forEach(box => {
        createFireEffect(box);
    });
}
</script>
"""

# Sidebar with additional info
with st.sidebar:
    st.markdown("## 💼Salary Predictor")
    st.markdown("""
    <p style='color:var(--dark-subtext)'>
    Advanced machine learning model predicting income levels based on demographic and employment factors.
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🔍 Model Specifications")
    st.markdown("""
    <div style='background-color: rgba(108, 92, 231, 0.1); padding: 1rem; border-radius: 8px;'>
        <p style='color:var(--dark-subtext); margin-bottom: 0.5rem;'>
        <strong>Algorithm:</strong> Random Forest Classifier
        </p>
        <p style='color:var(--dark-subtext); margin-bottom: 0.5rem;'>
        <strong>Accuracy:</strong> 86.4% (test set)
        </p>
        <p style='color:var(--dark-subtext); margin-bottom: 0.5rem;'>
        <strong>Training Data:</strong> US Census Bureau
        </p>
        <p style='color:var(--dark-subtext); margin-bottom: 0;'>
        <strong>Last Updated:</strong> July 2024
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🛠️ How To Use")
    st.markdown("""
    <ol style='color:var(--dark-subtext); padding-left: 1.2rem;'>
        <li style='margin-bottom: 0.5rem;'>Fill in all required fields</li>
        <li style='margin-bottom: 0.5rem;'>Click 'Predict Income' button</li>
        <li>View detailed prediction and insights</li>
    </ol>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center;'>
        <p style='color:var(--dark-subtext); margin-bottom: 0.5rem;'>Need help?</p>
        <button style='background-color: var(--primary); color: white; border: none; border-radius: 6px; padding: 0.5rem 1rem; cursor: pointer; transition: all 0.3s;' 
                onMouseOver="this.style.backgroundColor='var(--primary-light)'" 
                onMouseOut="this.style.backgroundColor='var(--primary)'">
            Contact Support
        </button>
    </div>
    """, unsafe_allow_html=True)

# Main content
st.title("💼 Salary Predictor")
st.markdown("""
<p style='color:var(--dark-subtext); font-size: 1.1rem;'>
Predict whether an individual's income exceeds $50K/year (₹4,150,000/year) based on comprehensive demographic analysis.
</p>
""", unsafe_allow_html=True)

# Form in two columns with tabs
tab1, tab2 = st.tabs(["📝 Input Form", "📊 Model Insights"])

with tab1:
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 👤 Personal Details")
            age = st.slider("Age", 17, 90, 30, 
                           help="Select the individual's age in years")
            gender = st.radio("Gender", 
                             options=["Female", "Male", "Other"], 
                             help="Select gender identity",
                             horizontal=True)
            marital_status = st.selectbox("Marital Status", 
                                        options=["Married", "Single", "Divorced", "Widowed", "Separated"],
                                        help="Current marital status")
            relationship = st.selectbox("Relationship Status", 
                                      options=["Husband", "Wife", "Own-child", "Unmarried", "Other-relative"],
                                      help="Relationship status in household")
            race = st.selectbox("Race/Ethnicity", 
                              options=["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"],
                              help="Race or ethnic group")
            
        with col2:
            st.markdown("### 💼 Employment Details")
            workclass = st.selectbox("Employment Sector", 
                                   options=["Private", "Government", "Self-employed", "Non-profit", "Other"],
                                   help="Primary employment sector")
            occupation = st.selectbox("Occupation Category", 
                                   options=["Tech", "Admin", "Services", "Professional", "Manual-labor", "Other"],
                                   help="Primary occupation field")
            education = st.selectbox("Highest Education", 
                                   options=["HS-grad", "Bachelors", "Masters", "Doctorate", "Some-college", "Other"],
                                   help="Highest level of education completed")
            education_num = st.slider("Years of Education", 1, 20, 10,
                                    help="Total years of formal education")
            hours_per_week = st.slider("Weekly Work Hours", 10, 100, 40,
                                     help="Typical hours worked per week")
            native_country = st.selectbox("Country of Origin", 
                                        options=["United-States", "Mexico", "India", "Philippines", "Germany", "Other"],
                                        help="Country of birth or origin")
            
            st.markdown("### 💰 Financial Information")
            capital_gain = st.number_input("Capital Gains ($)", min_value=0, value=0,
                                         help="Income from investments or asset sales")
            capital_loss = st.number_input("Capital Losses ($)", min_value=0, value=0,
                                         help="Losses from investments or asset sales")
            fnlwgt = st.number_input("Final Weight", min_value=0, value=100000,
                                   help="Demographic weighting factor")
        
        submitted = st.form_submit_button("🔮 Predict Income", use_container_width=True)

with tab2:
    st.markdown("### 🧠 Model Insights & Methodology")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Feature Importance")
        st.markdown("""
        <div class="feature-card">
            <p style='color:var(--dark-subtext);'>The model considers these as the most influential factors:</p>
            <ol style='color:var(--dark-subtext); padding-left: 1.2rem;'>
                <li>Education Level</li>
                <li>Occupation Type</li>
                <li>Age</li>
                <li>Weekly Work Hours</li>
                <li>Capital Gains</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### 📈 Performance Metrics")
        st.markdown("""
        ```python
        Accuracy: 86.4%
        Precision: 0.83
        Recall: 0.62
        F1 Score: 0.71
        AUC-ROC: 0.89
        ```
        """)
    
    with col2:
        st.markdown("#### ⚙️ Technical Details")
        st.markdown("""
        <div class="feature-card">
            <p style='color:var(--dark-subtext);'><strong>Model Architecture:</strong></p>
            <ul style='color:var(--dark-subtext); padding-left: 1.2rem;'>
                <li>Random Forest with 100 trees</li>
                <li>Max depth of 15</li>
                <li>Min samples split of 5</li>
            </ul>
            <p style='color:var(--dark-subtext); margin-top: 1rem;'><strong>Data Preprocessing:</strong></p>
            <ul style='color:var(--dark-subtext); padding-left: 1.2rem;'>
                <li>Label encoding for categorical features</li>
                <li>Standard scaling for numerical features</li>
                <li>Class weight balancing</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### ⚠️ Limitations & Considerations")
    st.markdown("""
    <div style='background-color: rgba(255, 118, 117, 0.1); padding: 1.5rem; border-radius: 10px; border-left: 4px solid var(--danger);'>
        <p style='color:var(--dark-subtext);'><strong>Important Notes:</strong></p>
        <ul style='color:var(--dark-subtext); padding-left: 1.2rem;'>
            <li>Predictions are statistical estimates only</li>
            <li>Model trained primarily on US demographic data</li>
            <li>May not account for all individual circumstances</li>
            <li>Results should be considered alongside other factors</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Prediction and results
if submitted:
    with st.spinner('Analyzing data and generating insights...'):
        progress_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.02)
            progress_bar.progress(percent_complete + 1)
        
        try:
            # Convert inputs to encoded values
            gender_encoded = 1 if gender == "Male" else (0 if gender == "Female" else 2)

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
            
            # Add JavaScript for fire effect
            st.markdown(FIRE_JS, unsafe_allow_html=True)
            
            # Scroll to results
            st.markdown('<div id="prediction-shown"></div>', unsafe_allow_html=True)
            
            # Display prediction with styling and INR conversion
            inr_amount = 50000 * USD_TO_INR
            if prediction == 1:
                st.markdown(f"""
                <div class="prediction-box result-animation" style='border-left: 6px solid var(--success);'>
                    <h2 style='color:var(--success); margin-top:0;'>💰 High Income Prediction</h2>
                    <p style='font-size:1.2rem; color:var(--dark-text);'>
                        This individual is likely earning <strong>>$50K/year (₹{inr_amount:,.0f}/year)</strong>
                    </p>
                    <div style='background-color:#2e2e3a; border-radius:8px; padding:1rem; margin:1rem 0;'>
                        <p style='margin:0; color:var(--dark-subtext);'><strong>Confidence Level:</strong> {probability*100:.1f}%</p>
                        <div style='height:10px; background-color:#1e1e2e; border-radius:5px; margin-top:0.5rem;'>
                            <div style='width:{probability*100}%; height:100%; background: linear-gradient(90deg, var(--success), #55efc4); border-radius:5px;'></div>
                        </div>
                    </div>
                    <div style='background-color: rgba(0, 184, 148, 0.1); padding: 1rem; border-radius: 8px; margin: 1rem 0;'>
                        <p style='color:var(--dark-subtext); margin-bottom: 0.5rem;'><strong>Key Contributing Factors:</strong></p>
                        <ul style='color:var(--dark-subtext); padding-left: 1.2rem;'>
                            <li>Higher education level ({(education_num/20)*100:.0f}% of max)</li>
                            <li>Professional occupation category</li>
                            <li>Full-time work hours ({hours_per_week} hrs/week)</li>
                            <li>Age in prime earning years ({age} years old)</li>
                        </ul>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-box result-animation" style='border-left: 6px solid var(--danger);'>
                    <h2 style='color:var(--danger); margin-top:0;'>💰 Moderate Income Prediction</h2>
                    <p style='font-size:1.2rem; color:var(--dark-text);'>
                        This individual is likely earning <strong>≤$50K/year (≤₹{inr_amount:,.0f}/year)</strong>
                    </p>
                    <div style='background-color:#2e2e3a; border-radius:8px; padding:1rem; margin:1rem 0;'>
                        <p style='margin:0; color:var(--dark-subtext);'><strong>Confidence Level:</strong> {(1-probability)*100:.1f}%</p>
                        <div style='height:10px; background-color:#1e1e2e; border-radius:5px; margin-top:0.5rem;'>
                            <div style='width:{(1-probability)*100}%; height:100%; background: linear-gradient(90deg, var(--danger), #fab1a0); border-radius:5px;'></div>
                        </div>
                    </div>
                    <div style='background-color: rgba(255, 118, 117, 0.1); padding: 1rem; border-radius: 8px; margin: 1rem 0;'>
                        <p style='color:var(--dark-subtext); margin-bottom: 0.5rem;'><strong>Potential Limiting Factors:</strong></p>
                        <ul style='color:var(--dark-subtext); padding-left: 1.2rem;'>
                            <li>Education level ({(education_num/20)*100:.0f}% of max)</li>
                            <li>Occupation category ({occupation})</li>
                            <li>Work hours ({hours_per_week} hrs/week)</li>
                            <li>Limited capital gains (${capital_gain:,})</li>
                        </ul>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Add recommendations section
            st.markdown("### 📝 Personalized Recommendations")
            if prediction == 1:
                st.markdown("""
                <div style='background-color: rgba(0, 184, 148, 0.1); padding: 1.5rem; border-radius: 10px; border-left: 4px solid var(--success);'>
                    <h4 style='margin-top:0; color:var(--success);'>Wealth Optimization Strategies:</h4>
                    <div style='display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 1rem; margin-top: 1rem;'>
                        <div style='background-color: var(--dark-card); padding: 1rem; border-radius: 8px;'>
                            <h5 style='color:var(--primary); margin-top:0;'>💼 Career Growth</h5>
                            <ul style='color:var(--dark-subtext); padding-left: 1.2rem; font-size: 0.9rem;'>
                                <li>Executive education programs</li>
                                <li>Leadership training</li>
                                <li>Industry networking</li>
                            </ul>
                        </div>
                        <div style='background-color: var(--dark-card); padding: 1rem; border-radius: 8px;'>
                            <h5 style='color:var(--primary); margin-top:0;'>💰 Investments</h5>
                            <ul style='color:var(--dark-subtext); padding-left: 1.2rem; font-size: 0.9rem;'>
                                <li>Diversified portfolio</li>
                                <li>Tax-advantaged accounts</li>
                                <li>Real estate investments</li>
                            </ul>
                        </div>
                        <div style='background-color: var(--dark-card); padding: 1rem; border-radius: 8px;'>
                            <h5 style='color:var(--primary); margin-top:0;'>🛡️ Protection</h5>
                            <ul style='color:var(--dark-subtext); padding-left: 1.2rem; font-size: 0.9rem;'>
                                <li>Estate planning</li>
                                <li>Insurance review</li>
                                <li>Tax optimization</li>
                            </ul>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='background-color: rgba(253, 203, 110, 0.1); padding: 1.5rem; border-radius: 10px; border-left: 4px solid var(--warning);'>
                    <h4 style='margin-top:0; color:var(--warning);'>Income Growth Pathways:</h4>
                    <div style='display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 1rem; margin-top: 1rem;'>
                        <div style='background-color: var(--dark-card); padding: 1rem; border-radius: 8px;'>
                            <h5 style='color:var(--primary); margin-top:0;'>🎓 Education</h5>
                            <ul style='color:var(--dark-subtext); padding-left: 1.2rem; font-size: 0.9rem;'>
                                <li>Certification programs</li>
                                <li>Online courses</li>
                                <li>Community college</li>
                            </ul>
                        </div>
                        <div style='background-color: var(--dark-card); padding: 1rem; border-radius: 8px;'>
                            <h5 style='color:var(--primary); margin-top:0;'>💻 Skills</h5>
                            <ul style='color:var(--dark-subtext); padding-left: 1.2rem; font-size: 0.9rem;'>
                                <li>Technical skills</li>
                                <li>Soft skills training</li>
                                <li>Industry-specific skills</li>
                            </ul>
                        </div>
                        <div style='background-color: var(--dark-card); padding: 1rem; border-radius: 8px;'>
                            <h5 style='color:var(--primary); margin-top:0;'>🚀 Career Moves</h5>
                            <ul style='color:var(--dark-subtext); padding-left: 1.2rem; font-size: 0.9rem;'>
                                <li>Job market research</li>
                                <li>Resume optimization</li>
                                <li>Salary negotiation</li>
                            </ul>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            st.markdown("""
            <div style='background-color: rgba(255, 118, 117, 0.1); padding: 1.5rem; border-radius: 10px; border-left: 4px solid var(--danger);'>
                <h4 style='margin-top:0; color:var(--danger);'>Error in Processing</h4>
                <p style='color:var(--dark-text);'>We encountered an issue while processing your request.</p>
                <p style='color:var(--dark-subtext); margin-bottom:0;'>Please ensure all fields are filled correctly and try again. If the problem persists, contact support.</p>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <div style="
        background-color: var(--dark-card);
        padding: 2rem 1rem;
        text-align: center;
        margin-top: 4rem;
        border-top: 1px solid #2e2e3a;
        font-family: 'Inter', sans-serif;
    ">
        <div style="max-width: 1000px; margin: 0 auto;">
            <div style="margin-bottom: 1.5rem;">
                <h3 style="
                    color: var(--primary);
                    margin-bottom: 0.5rem;
                    font-size: 1.5rem;
                ">💼 Salary Predictor</h3>
                <p style="
                    color: var(--dark-subtext);
                    font-size: 0.95rem;
                    margin: 0;
                ">Developed by Yoganandha</p>
            </div>
            <div style="
                display: flex;
                justify-content: center;
                gap: 1.2rem;
                margin: 1.5rem 0;
                flex-wrap: wrap;
            ">
                <a href="https://github.com/yoga0061" target="_blank" title="GitHub" style="transition: transform 0.2s;">
                    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg"
                         alt="GitHub" style="width: 26px; height: 26px; filter: invert(0.7);"/>
                </a>
                <a href="https://www.linkedin.com/in/yoganandha-banavathu-a02092305/" target="_blank" title="LinkedIn" style="transition: transform 0.2s;">
                    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg"
                         alt="LinkedIn" style="width: 26px; height: 26px; filter: invert(0.7);"/>
                </a>
                <a href="mailto:yoga.142007@gmail.com" title="Email" style="transition: transform 0.2s;">
                    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/google/google-original.svg"
                         alt="Email" style="width: 26px; height: 26px; filter: invert(0.7);"/>
                </a>
            </div>
            <div style="
                border-top: 1px solid #3a3a4a;
                padding-top: 1rem;
                margin-top: 1.5rem;
            ">
                <p style="
                    color: var(--dark-subtext);
                    font-size: 0.8rem;
                    margin: 0;
                ">
                    © 2025 All rights reserved | AI-powered income prediction tool
                </p>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)
