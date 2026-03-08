import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess_data(filepath="data/raw/synthetic_hypertension_data.csv"):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Define features and target
    X = df.drop('Hypertension_Stage', axis=1)
    y = df['Hypertension_Stage']
    
    # Handle categorical variables (Gender)
    # Using pandas get_dummies for simplicity in mapping later
    X = pd.get_dummies(X, columns=['Gender'], drop_first=True)
    
    # Target Encoding
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Save the label mapping for inference
    label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print("Label Mapping:", label_mapping)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    # Scaling numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save feature names for the pipeline
    feature_names = list(X.columns)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, le, feature_names

def train_and_evaluate(X_train, X_test, y_train, y_test, le):
    print("\nTraining Random Forest model...")
    # Using RandomForest as it generally performs well without exhaustive tuning and provides feature importance
    rf = RandomForestClassifier(random_state=42, n_estimators=100)
    rf.fit(X_train, y_train)
    
    print("\nEvaluating model...")
    y_pred = rf.predict(X_test)
    
    print("\nClassification Report:")
    target_names = [str(c) for c in le.classes_]
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    
    return rf

def analyze_feature_importance(model, feature_names, output_dir="data/processed"):
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    print("\nFeature Ranking:")
    for f in range(len(feature_names)):
        print(f"{f + 1}. {feature_names[indices[f]]} ({importance[indices[f]]:.4f})")
    
    # Save sorted feature importance for reference
    os.makedirs(output_dir, exist_ok=True)
    fi_df = pd.DataFrame({
        'Feature': [feature_names[i] for i in indices],
        'Importance': [importance[i] for i in indices]
    })
    fi_df.to_csv(f"{output_dir}/feature_importance.csv", index=False)

def save_pipeline(model, scaler, le, feature_names, output_dir="src/models"):
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model, f"{output_dir}/rf_model.pkl")
    joblib.dump(scaler, f"{output_dir}/scaler.pkl")
    joblib.dump(le, f"{output_dir}/label_encoder.pkl")
    joblib.dump(feature_names, f"{output_dir}/feature_names.pkl")
    print(f"\nPipeline saved to {output_dir}/")

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler, le, feature_names = load_and_preprocess_data()
    model = train_and_evaluate(X_train, X_test, y_train, y_test, le)
    analyze_feature_importance(model, feature_names)
    save_pipeline(model, scaler, le, feature_names)
