import streamlit as st
import pandas as pd
from src.predict import HypertensionPredictor

# Page Configuration
st.set_page_config(page_title="Hypertension Predictor", page_icon="🫀", layout="wide")

@st.cache_resource
def load_predictor():
    return HypertensionPredictor(model_dir="src/models")

predictor = load_predictor()

# --- Title and Disclaimer ---
st.title("🫀 Hypertension Stage Predictor")
st.markdown("""
> **⚠️ MEDICAL DISCLAIMER**: This application is a clinical decision-support tool designed for demonstration and educational purposes only. 
> It does **NOT** provide a medical diagnosis and should **NOT** replace professional medical advice, diagnosis, or treatment. 
> Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
""")

st.write("Enter the patient's clinical parameters and lifestyle data below to predict their hypertension stage and receive personalized recommendations.")

# --- Input Form ---
with st.form("patient_data_form"):
    st.subheader("Patient Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age (years)", min_value=18, max_value=120, value=45)
        gender = st.selectbox("Gender", ["Male", "Female"])
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
        
    with col2:
        systolic_bp = st.number_input("Systolic Blood Pressure (mmHg)", min_value=70, max_value=250, value=120)
        diastolic_bp = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=40, max_value=150, value=80)
        family_history = st.selectbox("Family History of Hypertension", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
        
    with col3:
        diabetes = st.selectbox("Diabetes", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
        ckd = st.selectbox("Chronic Kidney Disease", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
        
    st.subheader("Lifestyle Factors")
    col4, col5 = st.columns(2)
    
    with col4:
        smoking = st.selectbox("Smoking Status", [("Non-smoker", 0), ("Smoker", 1)], format_func=lambda x: x[0])[1]
        alcohol = st.selectbox("Alcohol Consumption", [("None", 0), ("Moderate", 1), ("Heavy", 2)], format_func=lambda x: x[0])[1]
        physical_activity = st.selectbox("Physical Activity", [("Low", 0), ("Moderate", 1), ("High", 2)], format_func=lambda x: x[0])[1]
        
    with col5:
        salt_intake = st.selectbox("Salt Intake", [("Low", 0), ("Moderate", 1), ("High", 2)], format_func=lambda x: x[0])[1]
        stress_level = st.selectbox("Stress Level", [("Low", 0), ("Moderate", 1), ("High", 2)], format_func=lambda x: x[0])[1]
        
    submit_button = st.form_submit_button("Predict Hypertension Stage")

# --- Processing and Output ---
if submit_button:
    patient_data = {
        'Age': age,
        'Gender': gender,
        'BMI': bmi,
        'Smoking': smoking,
        'Alcohol': alcohol,
        'Physical_Activity': physical_activity,
        'Salt_Intake': salt_intake,
        'Stress_Level': stress_level,
        'Family_History': family_history,
        'Diabetes': diabetes,
        'Chronic_Kidney_Disease': ckd,
        'Systolic_BP': systolic_bp,
        'Diastolic_BP': diastolic_bp
    }
    
    with st.spinner("Analyzing patient data..."):
        results = predictor.predict(patient_data)
        
    st.markdown("---")
    st.header("Results")
    
    # Display Stage and Risk Score
    stage = results['prediction']
    confidence = results['risk_score']
    
    # Determine color based on severity
    if stage == "Normal":
        color = "green"
    elif stage == "Elevated":
        color = "orange"
    elif stage == "Stage 1":
        color = "darkorange"
    elif stage == "Stage 2":
        color = "red"
    else: # Hypertensive Crisis
        color = "darkred"
        
    st.markdown(f"### Predicted Stage: <span style='color:{color}'>{stage}</span>", unsafe_allow_html=True)
    st.progress(confidence / 100)
    st.caption(f"Model Confidence: {confidence:.2f}%")
    
    # Recommendations
    st.subheader("Personalized Recommendations")
    for rec in results['recommendations']:
        st.info(rec)
        
    # Probabilities
    with st.expander("View Stage Probabilities"):
        probs_df = pd.DataFrame(list(results['probabilities'].items()), columns=['Stage', 'Probability'])
        probs_df['Probability'] = (probs_df['Probability'] * 100).round(2)
        st.bar_chart(probs_df.set_index('Stage'))
