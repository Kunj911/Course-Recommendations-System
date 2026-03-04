# train_models.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error, r2_score
import pickle
import os
from feature_engineering import create_features, get_feature_columns

# Create models directory
os.makedirs('models/saved', exist_ok=True)

# Load and prepare data
print("Loading data...")
df = create_features()
feature_cols = get_feature_columns()

X = df[feature_cols]
y_classification = df['passed']  # Binary: 0 or 1
y_regression = df['grade']        # Continuous: 0.0 to 10.0

# Feature scaling (important for 0-10 scale to prevent dominance)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)

# Train-test split (80-20)
X_train, X_test, y_class_train, y_class_test = train_test_split(
    X_scaled, y_classification, test_size=0.2, random_state=42, stratify=y_classification
)

_, _, y_reg_train, y_reg_test = train_test_split(
    X_scaled, y_regression, test_size=0.2, random_state=42
)

# --- LOGISTIC REGRESSION (Pass/Fail Prediction) ---
print("\nTraining Logistic Regression...")
log_reg = LogisticRegression(max_iter=2000, random_state=42, solver='lbfgs', C=1.0)
log_reg.fit(X_train, y_class_train)

y_class_pred = log_reg.predict(X_test)
y_class_proba = log_reg.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_class_test, y_class_pred)
precision = precision_score(y_class_test, y_class_pred, zero_division=0)
recall = recall_score(y_class_test, y_class_pred, zero_division=0)

print(f"  Accuracy:  {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")

# Save model and scaler
with open('models/saved/logistic_model.pkl', 'wb') as f:
    pickle.dump(log_reg, f)

# --- LINEAR REGRESSION (Grade Prediction, 0-10 scale) ---
print("\nTraining Linear Regression...")
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_reg_train)

y_reg_pred = lin_reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
r2 = r2_score(y_reg_test, y_reg_pred)
mae = np.mean(np.abs(y_reg_test - y_reg_pred))

print(f"  RMSE: {rmse:.4f} (on 0-10 scale)")
print(f"  MAE:  {mae:.4f} (on 0-10 scale)")
print(f"  R²:   {r2:.4f}")

# Save model
with open('models/saved/linear_model.pkl', 'wb') as f:
    pickle.dump(lin_reg, f)

# Save scaler
with open('models/saved/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\n✓ Models and scaler saved successfully!")
print("✓ All predictions on 0-10 CGPA scale")