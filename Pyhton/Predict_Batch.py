import pandas as pd
import joblib
import os
import numpy as np

# Load the trained model and scaler
model = joblib.load('logistic_model.pkl')
scaler = joblib.load('fare_waitingtime_scaler.pkl')  # Updated scaler name to reflect both Fare and waiting_time

def load_and_preprocess(file_path):
    """
    Load the CSV file and preprocess the data.
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Check if the required columns are present
    required_columns = [
        'Age18To35', 'Age36Above', 'gender', 'high_school_or_lower', 
        'bachelor', 'master_or_higher', 'Is_student', 'Driving_License', 
        'Interest_in_tech', 'Regular_taxi_user', 'Previous_AV_Experience', 
        'Weather', 'waiting_time', 'Fare'
    ]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"The input CSV file must contain the following columns: {required_columns}")

    # Store the original Fare and waiting_time values in new columns
    df['Original_Fare'] = df['Fare']
    df['Original_waiting_time'] = df['waiting_time']

    # Scale the 'waiting_time' and 'Fare' columns for prediction
    df[['waiting_time', 'Fare']] = scaler.transform(df[['waiting_time', 'Fare']])

    return df

def predict_from_csv(file_path):
    """
    Predict whether users in the CSV file will choose an autonomous taxi.
    """
    try:
        # Load and preprocess the data
        data = load_and_preprocess(file_path)

        # Define the feature columns for prediction
        feature_columns = [
            'Age18To35', 'Age36Above', 'gender', 'high_school_or_lower', 
            'bachelor', 'master_or_higher', 'Is_student', 'Driving_License', 
            'Interest_in_tech', 'Regular_taxi_user', 'Previous_AV_Experience', 
            'Weather', 'waiting_time', 'Fare'
        ]

        # Make predictions
        predictions = model.predict(data[feature_columns])
        probabilities = model.predict_proba(data[feature_columns])

        # Add predictions and probabilities to the DataFrame
        data['Prediction'] = ['Will Choose AC' if pred == 1 else 'Will Not Choose AC' for pred in predictions]
        data['Probability_Choosing_AC'] = [np.round(prob[1], 3) for prob in probabilities]

        # Replace the scaled 'Fare' and 'waiting_time' with the original values
        data['Fare'] = data['Original_Fare']
        data['waiting_time'] = data['Original_waiting_time']
        data.drop(['Original_Fare', 'Original_waiting_time'], axis=1, inplace=True)

        # Save the updated DataFrame to a new file
        output_path = os.path.splitext(file_path)[0] + '_with_predictions.csv'
        data.to_csv(output_path, index=False)
        print(f"Predictions appended and saved to: {output_path}")

    except Exception as e:
        print(f"Error: {e}")

# Main block to prompt for file upload
if __name__ == '__main__':
    file_path = input("Enter the path to the CSV file containing demographic data: ").strip()
    predict_from_csv(file_path)