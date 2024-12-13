import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import joblib

# Load the dataset
file_path = "D:/ESD-USB/2024-25_Design Engineering/DESE71010_Foundational Transdisciplinary Research Methods/Coursework/97_ML/clean_data_binary.csv" 
df = pd.read_csv(file_path)

# Define categorical and numeric columns
categorical_cols = [
    'Age18To35', 'Age36Above', 'gender', 'high_school_or_lower', 
    'bachelor', 'master_or_higher', 'Is_student', 'Driving_License', 
    'Interest_in_tech', 'Regular_taxi_user', 'Previous_AV_Experience', 'Weather'
]
numeric_cols = ['waiting_time', 'Fare']

# Separate the features and target
X_categorical = df[categorical_cols]
X_numeric = df[numeric_cols]
y = df['Choice']

# Scale the continuous variables
scaler = StandardScaler()
X_numeric_scaled = scaler.fit_transform(X_numeric)

# Combine categorical and scaled numeric data
X = np.hstack([X_categorical, X_numeric_scaled])

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the model and scaler
joblib.dump(model, 'logistic_model.pkl')
joblib.dump(scaler, 'fare_waitingtime_scaler.pkl')
print("Model and scaler have been saved successfully.")