# app.py
import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the trained model
@st.cache_resource
def load_model():
    return pickle.load(open("model.pkl", "rb"))

model = load_model()

# Assuming you know the correct order of features from the training process
correct_feature_order = [
    'age', 'workclass', 'fnlwgt', 'education', 'educational-num',
    'marital-status', 'occupation', 'relationship', 'race', 'gender',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country'
]

# Initialize LabelEncoders for categorical features
label_encoders = {feature: LabelEncoder() for feature in correct_feature_order if feature not in ['age', 'fnlwgt', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']}

# Page configuration and other setup code remains the same

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

# Function to calculate net profit
def calculate_net_profit(capital_gain, capital_loss):
    return capital_gain - capital_loss

# Function to convert USD to INR
def convert_to_inr(amount_usd):
    # Assuming a conversion rate of 1 USD = 83 INR
    conversion_rate = 83
    return amount_usd * conversion_rate

# Prediction and results
if submitted:
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

    # Create input data DataFrame with all expected features in the correct order
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

    with st.spinner('Analyzing the data...'):
        try:
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]
            net_profit_usd = calculate_net_profit(capital_gain, capital_loss)
            net_profit_inr = convert_to_inr(net_profit_usd)

            st.success("Prediction Complete!")
            st.balloons()

            if prediction == 1:
                st.markdown(f"""
                <div class="prediction-box" style='background-color:#e8f5e9;'>
                    <h3 style='color:#2e7d32'>💰 Prediction: >$50K/year</h3>
                    <p>Confidence: {probability*100:.1f}%</p>
                    <p>This individual is likely earning more than $50,000 annually based on the provided information.</p>
                    <p>💸 Net Profit: ${net_profit_usd:,.2f} (₹{net_profit_inr:,.2f} INR)</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-box" style='background-color:#ffebee;'>
                    <h3 style='color:#c62828'>💰 Prediction: ≤$50K/year</h3>
                    <p>Confidence: {(1-probability)*100:.1f}%</p>
                    <p>This individual is likely earning $50,000 or less annually based on the provided information.</p>
                    <p>💸 Net Profit: ${net_profit_usd:,.2f} (₹{net_profit_inr:,.2f} INR)</p>
                </div>
                """, unsafe_allow_html=True)
        except ValueError as e:
            st.error(f"Error during prediction: {e}")
            st.write("Please ensure all input fields are correctly filled and match the expected format.")
