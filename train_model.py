import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1. Load Data
csv_file = "concussion_training_data.csv"
try:
    df = pd.read_csv(csv_file)
except FileNotFoundError:
    print(f"Error: '{csv_file}' not found. Run generate_synthetic_data.py or collect data using app.py first.")
    exit()

# Define feature columns (excluding timestamp and label)
feature_cols = [
    "pupil_diameter", "pupil_deviation", "pupil_anomaly",
    "blink_rate", "blink_duration", "rms_jitter",
    "peak_drift", "pursuit_gain", "pursuit_error", "saccade_count"
]

X = df[feature_cols]
y = df["label"]

# 2. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# 3. Train Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluate Accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("==========================================")
print(f"MODEL EVALUATION RESULTS (Accuracy: {acc * 100:.2f}%)")
print("==========================================")
print(classification_report(y_test, y_pred))

# 5. Extract Feature Importance Weights
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance Weight': importances
}).sort_values(by='Importance Weight', ascending=False)

print("\nRANKED SYMPTOM IMPORTANCE WEIGHTS:")
print("------------------------------------------")
print(feature_importance_df.to_string(index=False))

# 6. Save Model for Future Raspberry Pi Deployment
model_filename = "concussion_detector_model.pkl"
joblib.dump(model, model_filename)
print(f"\n[INFO] Model successfully trained and saved to '{model_filename}'")