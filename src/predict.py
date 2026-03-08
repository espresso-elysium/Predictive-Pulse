import os
import joblib
import pandas as pd
import numpy as np

class HypertensionPredictor:
    def __init__(self, model_dir="src/models"):
        self.model = joblib.load(os.path.join(model_dir, "rf_model.pkl"))
        self.scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
        self.le = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))
        self.feature_names = joblib.load(os.path.join(model_dir, "feature_names.pkl"))
        
    def preprocess_input(self, input_data):
        """
        Preprocesses a dictionary of patient data into the format expected by the model.
        """
        df = pd.DataFrame([input_data])
        
        # Handle Categorical mappings if passed as strings instead of exact format
        if 'Gender' in df.columns:
            # The model expects 'Gender_Male' (boolean/int)
            # If the original input is Gender: 'Male' or 'Female'
            df['Gender_Male'] = 1 if df['Gender'].iloc[0] == 'Male' else 0
            df = df.drop('Gender', axis=1)
            
        # Ensure all expected features are present in the correct order
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0 # Default if missing
                
        df = df[self.feature_names]
        
        # Scale
        scaled_data = self.scaler.transform(df)
        return scaled_data
        
    def predict(self, input_data):
        """
        Returns the prediction, probabilities, and recommendations.
        """
        X_scaled = self.preprocess_input(input_data)
        
        # Prediction
        pred_encoded = self.model.predict(X_scaled)[0]
        prediction = self.le.inverse_transform([pred_encoded])[0]
        
        # Probabilities for risk score
        probs = self.model.predict_proba(X_scaled)[0]
        prob_dict = {self.le.classes_[i]: float(probs[i]) for i in range(len(self.le.classes_))}
        
        # Generate recommendations based on the input features
        recommendations = self.generate_recommendations(input_data, prediction)
        
        return {
            "prediction": prediction,
            "probabilities": prob_dict,
            "recommendations": recommendations,
            "risk_score": max(probs) * 100 # Confidence of the top class
        }
        
    def generate_recommendations(self, data, stage):
        recs = []
        if stage == "Normal":
            recs.append("Maintain your current healthy lifestyle and continue annual checkups.")
        elif stage == "Elevated":
            recs.append("Implement lifestyle changes such as improved diet and exercise to prevent progression.")
        elif stage == "Stage 1":
            recs.append("Consult a healthcare provider. Lifestyle modifications are strongly recommended, and medication may be considered.")
        elif stage == "Stage 2":
            recs.append("Consult a healthcare provider promptly. Medication and strict lifestyle changes are usually required.")
        elif stage == "Hypertensive Crisis":
            recs.append("SEEK IMMEDIATE EMERGENCY MEDICAL HELP. This is a critical condition.")
            
        # Specific feature-based recommendations
        if data.get('Smoking') == 1:
            recs.append("Smoking cessation is critical for cardiovascular health.")
        if data.get('Stress_Level', 0) >= 1:
            recs.append("Consider stress management techniques (e.g., meditation, yoga).")
        if data.get('Salt_Intake', 0) >= 1:
            recs.append("Reduce dietary sodium intake to lower blood pressure.")
        if data.get('Physical_Activity', 1) == 0:
            recs.append("Aim for at least 30 minutes of moderate aerobic activity daily.")
        if data.get('BMI', 22) >= 25:
            recs.append("Weight reduction can significantly lower blood pressure.")
            
        return recs

if __name__ == "__main__":
    predictor = HypertensionPredictor(model_dir="src/models")
    sample_patient = {
        'Age': 45, 'Gender': 'Male', 'BMI': 28.5, 'Smoking': 1, 'Alcohol': 1,
        'Physical_Activity': 0, 'Salt_Intake': 2, 'Stress_Level': 2, 'Family_History': 1,
        'Diabetes': 0, 'Chronic_Kidney_Disease': 0, 'Systolic_BP': 145, 'Diastolic_BP': 92
    }
    result = predictor.predict(sample_patient)
    print("Prediction:", result)
