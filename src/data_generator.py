import pandas as pd
import numpy as np
import random
import os

def generate_hypertension_data(num_samples=5000, output_path="../data/raw/synthetic_hypertension_data.csv"):
    """
    Generates a synthetic dataset for hypertension stage prediction.
    Correlates features like age, BMI, lifestyle habits with BP readings.
    """
    np.random.seed(42)
    random.seed(42)
    
    # 1. Base Features
    age = np.random.randint(20, 90, num_samples)
    gender = np.random.choice(['Male', 'Female'], p=[0.5, 0.5], size=num_samples)
    
    # BMI (Normal: 18.5-24.9, Overweight: 25-29.9, Obese: 30+)
    # Correlate BMI slightly with age
    bmi_base = np.random.normal(loc=26, scale=5, size=num_samples)
    bmi = bmi_base + (age - 40) * 0.05
    bmi = np.clip(bmi, 16.0, 45.0)
    
    # Lifestyle Factors
    smoking = np.random.choice([0, 1], p=[0.75, 0.25], size=num_samples) # 0: No, 1: Yes
    alcohol = np.random.choice([0, 1, 2], p=[0.5, 0.3, 0.2], size=num_samples) # 0: None, 1: Moderate, 2: Heavy
    physical_activity = np.random.choice([0, 1, 2], p=[0.3, 0.5, 0.2], size=num_samples) # 0: Low, 1: Moderate, 2: High
    salt_intake = np.random.choice([0, 1, 2], p=[0.2, 0.5, 0.3], size=num_samples) # 0: Low, 1: Moderate, 2: High
    stress_level = np.random.choice([0, 1, 2], p=[0.3, 0.4, 0.3], size=num_samples) # 0: Low, 1: Moderate, 2: High
    
    # Medical History
    family_history = np.random.choice([0, 1], p=[0.6, 0.4], size=num_samples)
    diabetes = np.random.choice([0, 1], p=[0.85, 0.15], size=num_samples)
    ckd = np.random.choice([0, 1], p=[0.95, 0.05], size=num_samples) # Chronic Kidney Disease
    
    # 2. Generate Blood Pressure (Systolic and Diastolic) highly correlated with the features
    
    # Base BP for a healthy young adult
    base_sys = 110
    base_dia = 70
    
    # Additive effects
    sys_effect = (age - 30) * 0.4 + (bmi - 22) * 1.2 + smoking * 5 + alcohol * 3 + salt_intake * 4 + stress_level * 3 + family_history * 5 + diabetes * 6 + ckd * 10 - physical_activity * 4
    dia_effect = (age - 30) * 0.2 + (bmi - 22) * 0.8 + smoking * 3 + alcohol * 2 + salt_intake * 2 + stress_level * 2 + family_history * 3 + diabetes * 4 + ckd * 5 - physical_activity * 2
    
    # Add some noise
    sys_noise = np.random.normal(0, 8, num_samples)
    dia_noise = np.random.normal(0, 6, num_samples)
    
    systolic_bp = base_sys + sys_effect + sys_noise
    diastolic_bp = base_dia + dia_effect + dia_noise
    
    # Ensure realistic ranges
    systolic_bp = np.clip(np.round(systolic_bp), 90, 220)
    diastolic_bp = np.clip(np.round(diastolic_bp), 60, 140)
    
    # 3. Determine Hypertension Stage based on AHA/ACC Guidelines
    # Normal: Sys < 120 AND Dia < 80
    # Elevated: Sys 120-129 AND Dia < 80
    # Stage 1: Sys 130-139 OR Dia 80-89
    # Stage 2: Sys 140-180 OR Dia 90-120
    # Crisis: Sys > 180 OR Dia > 120
    
    stage = []
    for sys, dia in zip(systolic_bp, diastolic_bp):
        if sys > 180 or dia > 120:
            stage.append("Hypertensive Crisis")
        elif sys >= 140 or dia >= 90:
            stage.append("Stage 2")
        elif sys >= 130 or dia >= 80:
            stage.append("Stage 1")
        elif sys >= 120 and dia < 80:
            stage.append("Elevated")
        else:
            stage.append("Normal")
            
    # Assemble DataFrame
    df = pd.DataFrame({
        'Age': np.round(age).astype(int),
        'Gender': gender,
        'BMI': np.round(bmi, 1),
        'Smoking': smoking,
        'Alcohol': alcohol,
        'Physical_Activity': physical_activity,
        'Salt_Intake': salt_intake,
        'Stress_Level': stress_level,
        'Family_History': family_history,
        'Diabetes': diabetes,
        'Chronic_Kidney_Disease': ckd,
        'Systolic_BP': systolic_bp.astype(int),
        'Diastolic_BP': diastolic_bp.astype(int),
        'Hypertension_Stage': stage
    })
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Successfully generated {num_samples} samples and saved to {output_path}")
    print("\nStage Distribution:")
    print(df['Hypertension_Stage'].value_counts())
    
    return df

if __name__ == "__main__":
    generate_hypertension_data()
