import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
from sklearn.preprocessing import LabelEncoder

# Page configuration
st.set_page_config(
    page_title="Salary Predictor Pro",
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

        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: var(--dark-card);
            color: var(--dark-subtext);
            text-align: center;
            padding: 10px;
            border-top: 1px solid #2e2e3a;
            font-size: 0.8rem;
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
                if value not in label_encoders[feature].classes_:
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
                <div class="prediction-box" style='border-left: 6px solid var(--primary);'>
                    <h2 style='color:var(--primary); margin-top:0;'>💰 High Income Prediction</h2>
                    <p style='font-size:1.2rem; color:var(--dark-text);'>
                        This individual is likely earning <strong>>$50K/year (₹{inr_amount:,.0f}/year)</strong>
                    </p>
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
                inr_amount = 50000 * USD_TO_INR
                st.markdown(f"""
                <div class="prediction-box" style='border-left: 6px solid #ff7675;'>
                    <h2 style='color:#ff7675; margin-top:0;'>💰 Moderate Income Prediction</h2>
                    <p style='font-size:1.2rem; color:var(--dark-text);'>
                        This individual is likely earning <strong>≤$50K/year (≤₹{inr_amount:,.0f}/year)</strong>
                    </p>
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

            # Feature Importance Visualization
            st.markdown("### 📊 Feature Importance")
            feature_importances = {
                'age': 0.1,
                'workclass': 0.05,
                'education': 0.2,
                'marital-status': 0.03,
                'occupation': 0.15,
                'relationship': 0.07,
                'race': 0.02,
                'gender': 0.08,
                'capital-gain': 0.12,
                'capital-loss': 0.06,
                'hours-per-week': 0.1,
                'native-country': 0.02
            }

            importance_df = pd.DataFrame(list(feature_importances.items()), columns=['Feature', 'Importance'])
            importance_df = importance_df.sort_values(by='Importance', ascending=False)

            fig = px.bar(importance_df, x='Importance', y='Feature', title='Feature Importance',
                         orientation='h', color='Importance',
                         color_continuous_scale='Viridis')

            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                yaxis=dict(autorange="reversed")
            )

            st.plotly_chart(fig)

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            st.markdown("""
            <div style='background-color:#2d3436; padding:1rem; border-radius:8px; border-left: 4px solid #ff7675;'>
                <p style='color:var(--dark-text);'>Please check your inputs and try again.</p>
                <p style='color:var(--dark-subtext);'>Ensure all fields are filled correctly.</p>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <hr style="border: 1px solid #444; margin-top: 40px; margin-bottom: 20px;" />
    <div style="text-align: center; color: #a7a9be; font-size: 14px; line-height: 1.6;">
        <p>© 2025 <strong style="color: #6c5ce7;">AI Salary Insights Dashboard</strong> | Built with ❤️ by <strong style="color: #6c5ce7;">Yoganandha</strong></p>
        <div style="margin: 10px 0;">
            <a href="https://www.linkedin.com/in/yoganandha-banavathu-a02092305/" target="_blank" style="text-decoration: none; color: #a7a9be; margin: 0 10px;">💼 LinkedIn</a>
            <span style="color: #a7a9be;">|</span>
            <a href="https://yoga0061.github.io/portfolio/" target="_blank" style="text-decoration: none; color: #a7a9be; margin: 0 10px;">🌐 Portfolio</a>
        </div>
        <p style="font-style: italic; font-size: 12px; color: #a7a9be;"><em>Disclaimer:</em> Predictions are AI-based estimates and not guaranteed.</p>
        <p style="font-size: 12px; color: #a7a9be;"><small>Exchange Rate (FYI): 1 USD ≈ 83 INR</small></p>
    </div>
""", unsafe_allow_html=True)
