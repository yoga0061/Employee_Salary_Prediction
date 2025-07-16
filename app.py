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

# Custom Futuristic Dark Theme CSS
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Roboto:wght@300;400&display=swap');

        :root {
            --primary: #00ddeb; /* Neon Cyan */
            --secondary: #ff2e63; /* Neon Magenta */
            --bg: #0a0a1f; /* Deep Space Black */
            --card-bg: #1c1c3c; /* Dark Nebula */
            --text: #e0e0ff; /* Light Gray */
            --subtext: #8a8aff; /* Soft Blue */
            --glow: 0 0 10px var(--primary), 0 0 20px var(--primary);
        }

        body {
            font-family: 'Roboto', sans-serif;
            color: var(--text);
            background: linear-gradient(135deg, var(--bg), #141430);
        }

        .stApp {
            background: transparent;
        }

        .stForm {
            background: var(--card-bg);
            border-radius: 12px;
            padding: 2rem;
            box-shadow: var(--glow);
            transition: transform 0.3s ease;
        }

        .stForm:hover {
            transform: translateY(-5px);
        }

        .stButton>button {
            background: linear-gradient(45deg, var(--primary), var(--secondary));
            color: var(--text);
            border-radius: 12px;
            padding: 0.8rem 2rem;
            font-family: 'Orbitron', sans-serif;
            font-weight: 700;
            font-size: 1.1rem;
            border: none;
            box-shadow: var(--glow);
            transition: all 0.3s ease;
        }

        .stButton>button:hover {
            transform: scale(1.05) rotateX(5deg);
            box-shadow: 0 0 20px var(--primary), 0 0 30px var(--secondary);
        }

        .prediction-box {
            background: var(--card-bg);
            border-radius: 15px;
            padding: 2rem;
            margin: 1.5rem 0;
            box-shadow: var(--glow);
            border: 1px solid var(--primary);
            transition: transform 0.3s ease;
        }

        .prediction-box:hover {
            transform: translateY(-5px) rotateX(2deg);
        }

        .stSlider > div > div > div > div {
            background-color: var(--primary);
            box-shadow: var(--glow);
        }

        .stSelectbox > div > div > div {
            background-color: var(--card-bg);
            border: 1px solid var(--subtext);
            border-radius: 8px;
            color: var(--text);
        }

        .stRadio > label > div {
            color: var(--text);
            background: var(--card-bg);
            border-radius: 8px;
            padding: 0.5rem;
        }

        .stNumberInput > div > div > input {
            background-color: var(--card-bg);
            color: var(--text);
            border: 1px solid var(--subtext);
            border-radius: 8px;
        }

        .sidebar .sidebar-content {
            background: var(--card-bg);
            border-right: 1px solid var(--primary);
            box-shadow: var(--glow);
        }

        .sidebar h2 {
            font-family: 'Orbitron', sans-serif;
            color: var(--primary);
            text-shadow: var(--glow);
        }

        a {
            color: var(--primary);
            text-decoration: none;
            transition: color 0.3s ease;
        }

        a:hover {
            color: var(--secondary);
            text-shadow: var(--glow);
        }

        .footer {
            margin-top: 2rem;
            padding: 1.5rem;
            text-align: center;
            background: var(--card-bg);
            border-top: 1px solid var(--primary);
            color: var(--subtext);
            font-size: 0.9rem;
            box-shadow: var(--glow);
        }

        h1, h2, h3, h4 {
            font-family: 'Orbitron', sans-serif;
            color: var(--primary);
            text-shadow: var(--glow);
        }

        .input-card {
            background: var(--card-bg);
            padding: 1rem;
            border-radius: 10px;
            border: 1px solid var(--subtext);
            margin-bottom: 1rem;
            transition: all 0.3s ease;
        }

        .input-card:hover {
            box-shadow: var(--glow);
            transform: translateY(-3px);
        }
    </style>
""", unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    return pickle.load(open("model.pkl", "rb"))

model = load_model()

# Exchange rate
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

# Sidebar
with st.sidebar:
    st.markdown("<h2>🌌 Quantum Predictor</h2>", unsafe_allow_html=True)
    st.markdown("""
        <p style='color:var(--subtext); font-size:0.95rem;'>
        Harness AI to predict income with cosmic precision.
        </p>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🚀 Model Specs")
    st.markdown("""
        <p style='color:var(--subtext); font-size:0.9rem;'>
        • <strong>Core:</strong> Random Forest<br>
        • <strong>Precision:</strong> ~85%<br>
        • <strong>Data:</strong> US Census
        </p>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🛠️ Usage")
    st.markdown("""
        <p style='color:var(--subtext); font-size:0.9rem;'>
        1. Input data<br>
        2. Activate prediction<br>
        3. Analyze results
        </p>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("<p style='color:var(--subtext); font-size:0.85rem;'>Powered by Streamlit</p>", unsafe_allow_html=True)

# Main content
st.markdown("<h1 style='text-align:center;'>🌌 Quantum Salary Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:var(--subtext); font-size:1rem;'>Decode your income potential with cutting-edge AI</p>", unsafe_allow_html=True)

# Tabs
tab1, tab2 = st.tabs(["🖥️ Input Interface", "ℹ️ System Specs"])

with tab1:
    with st.form("prediction_form"):
        st.markdown("<h3>Data Input Matrix</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2, gap="medium")
        
        with col1:
            st.markdown("<h4 style='color:var(--secondary);'>Personal Data</h4>", unsafe_allow_html=True)
            with st.container():
                st.markdown("<div class='input-card'>", unsafe_allow_html=True)
                age = st.slider("Age", 17, 90, 30, help="Select age", key="age")
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("<div class='input-card'>", unsafe_allow_html=True)
                gender = st.radio("Gender", options=["Female", "Male"], horizontal=True, help="Select gender", key="gender")
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("<div class='input-card'>", unsafe_allow_html=True)
                marital_status = st.selectbox("Marital Status", 
                                            options=["Married", "Single", "Divorced", "Widowed", "Separated"],
                                            help="Select marital status", key="marital")
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("<div class='input-card'>", unsafe_allow_html=True)
                relationship = st.selectbox("Relationship Status", 
                                          options=["Husband", "Wife", "Own-child", "Unmarried", "Other-relative"],
                                          help="Select relationship status", key="relationship")
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("<div class='input-card'>", unsafe_allow_html=True)
                race = st.selectbox("Race", 
                                  options=["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"],
                                  help="Select race", key="race")
                st.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            st.markdown("<h4 style='color:var(--secondary);'>Professional Data</h4>", unsafe_allow_html=True)
            with st.container():
                st.markdown("<div class='input-card'>", unsafe_allow_html=True)
                workclass = st.selectbox("Employment Sector", 
                                       options=["Private", "Government", "Self-employed", "Non-profit", "Other"],
                                       help="Select employment sector", key="workclass")
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("<div class='input-card'>", unsafe_allow_html=True)
                occupation = st.selectbox("Occupation", 
                                       options=["Tech", "Admin", "Services", "Professional", "Manual-labor", "Other"],
                                       help="Select occupation", key="occupation")
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("<div class='input-card'>", unsafe_allow_html=True)
                education = st.selectbox("Highest Education", 
                                       options=["HS-grad", "Bachelors", "Masters", "Doctorate", "Some-college", "Other"],
                                       help="Select education level", key="education")
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("<div class='input-card'>", unsafe_allow_html=True)
                education_num = st.slider("Years of Education", 1, 20, 10, help="Select years of education", key="edu_num")
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("<div class='input-card'>", unsafe_allow_html=True)
                hours_per_week = st.slider("Weekly Work Hours", 10, 100, 40, help="Select weekly work hours", key="hours")
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("<div class='input-card'>", unsafe_allow_html=True)
                native_country = st.selectbox("Country of Origin", 
                                            options=["United-States", "Mexico", "India", "Philippines", "Germany", "Other"],
                                            help="Select country of origin", key="country")
                st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<h4 style='color:var(--secondary);'>Financial Data</h4>", unsafe_allow_html=True)
            with st.container():
                st.markdown("<div class='input-card'>", unsafe_allow_html=True)
                capital_gain = st.number_input("Capital Gains ($)", min_value=0, value=0, help="Enter capital gains", key="gain")
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("<div class='input-card'>", unsafe_allow_html=True)
                capital_loss = st.number_input("Capital Losses ($)", min_value=0, value=0, help="Enter capital losses", key="loss")
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("<div class='input-card'>", unsafe_allow_html=True)
                fnlwgt = st.number_input("Final Weight", min_value=0, value=100000, help="Enter final weight", key="fnlwgt")
                st.markdown("</div>", unsafe_allow_html=True)
        
        submitted = st.form_submit_button("⚡️ Compute Prediction", use_container_width=True)

# Prediction and results
if submitted:
    with st.spinner('Scanning data matrix...'):
        try:
            # Convert inputs to encoded values
            gender_encoded = 1 if gender == "Male" else 0
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
            
            st.success("Prediction Matrix Generated!")
            st.balloons()
            
            # Display prediction
            inr_amount = 50000 * USD_TO_INR
            if prediction == 1:
                st.markdown(f"""
                <div class="prediction-box" style='border-left: 5px solid var(--primary);'>
                    <h3>High Income Detected</h3>
                    <p style='font-size:1.1rem;'>
                        Projected: <strong>>$50K/year (₹{inr_amount:,.0f}/year)</strong>
                    </p>
                    <div style='background-color:#2a2a4a; border-radius:8px; padding:1rem;'>
                        <p style='margin:0; color:var(--subtext);'><strong>Confidence:</strong> {probability*100:.1f}%</p>
                        <div style='height:10px; background-color:#1c1c3c; border-radius:5px; margin-top:0.5rem;'>
                            <div style='width:{probability*100}%; height:100%; background: linear-gradient(45deg, var(--primary), var(--secondary)); border-radius:5px; transition: width 1s ease;'></div>
                        </div>
                    </div>
                    <p style='color:var(--subtext); margin-top:1rem;'>Key Drivers:</p>
                    <ul style='color:var(--subtext); font-size:0.9rem;'>
                        <li>Education Matrix</li>
                        <li>Occupational Vector</li>
                        <li>Experience Factor</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-box" style='border-left: 5px solid var(--secondary);'>
                    <h3>Moderate Income Detected</h3>
                    <p style='font-size:1.1rem;'>
                        Projected: <strong>≤$50K/year (≤₹{inr_amount:,.0f}/year)</strong>
                    </p>
                    <div style='background-color:#2a2a4a; border-radius:8px; padding:1rem;'>
                        <p style='margin:0; color:var(--subtext);'><strong>Confidence:</strong> {(1-probability)*100:.1f}%</p>
                        <div style='height:10px; background-color:#1c1c3c; border-radius:5px; margin-top:0.5rem;'>
                            <div style='width:{(1-probability)*100}%; height:100%; background: linear-gradient(45deg, var(--secondary), var(--primary)); border-radius:5px; transition: width 1s ease;'></div>
                        </div>
                    </div>
                    <p style='color:var(--subtext); margin-top:1rem;'>Influencing Factors:</p>
                    <ul style='color:var(--subtext); font-size:0.9rem;'>
                        <li>Education Matrix</li>
                        <li>Work Hours</li>
                        <li>Industry Sector</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            # Recommendations
            st.markdown("<h3 style='color:var(--secondary);'>Strategic Insights</h3>", unsafe_allow_html=True)
            if prediction == 1:
                st.markdown("""
                <div class='input-card' style='border-left: 4px solid var(--primary);'>
                    <h4 style='color:var(--primary); margin-top:0;'>For High Earners:</h4>
                    <ul style='color:var(--subtext); font-size:0.9rem;'>
                        <li>Optimize tax strategies</li>
                        <li>Enhance investment portfolios</li>
                        <li>Pursue advanced training</li>
                        <li>Plan for long-term wealth</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class='input-card' style='border-left: 4px solid var(--secondary);'>
                    <h4 style='color:var(--secondary); margin-top:0;'>For Income Growth:</h4>
                    <ul style='color:var(--subtext); font-size:0.9rem;'>
                        <li>Acquire new certifications</li>
                        <li>Explore high-demand sectors</li>
                        <li>Develop technical skills</li>
                        <li>Master negotiation techniques</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"System Error: {str(e)}")
            st.markdown("""
            <div class='input-card' style='border-left: 4px solid var(--secondary);'>
                <p style='color:var(--text);'>Invalid data input. Please verify entries.</p>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class="footer">
        <p>© 2025 Quantum Insights | <a href="https://www.linkedin.com/in/yoganandha-banavathu-a02092305/">Yoganandha</a></p>
        <p><a href="https://yoga0061.github.io/portfolio/">Portfolio</a> | Powered by Streamlit</p>
        <p style='font-size:0.8rem; color:var(--subtext);'>Exchange Rate: 1 USD ≈ 83 INR</p>
    </div>
""", unsafe_allow_html=True)
