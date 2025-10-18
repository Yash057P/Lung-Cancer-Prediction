import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

DATASET_PATH = 'Your Dataset file path'
TARGET_COLUMN = 'LUNG_CANCER'

def load_and_preprocess_data(file_path):
    """
    Loads the dataset and converts categorical and 1/2 scale features
    into a numerical format suitable for machine learning.
    """
    print(f"Loading data from: {file_path}")
    df = pd.read_csv(file_path)

    le_gender = LabelEncoder()
    df['GENDER'] = le_gender.fit_transform(df['GENDER'])

    le_target = LabelEncoder()
    df[TARGET_COLUMN] = le_target.fit_transform(df[TARGET_COLUMN])
    
    symptom_cols = df.columns.drop(['GENDER', 'AGE', TARGET_COLUMN])
    df[symptom_cols] = df[symptom_cols].replace({1: 0, 2: 1})
    
    feature_mapping = {
        'le_gender': le_gender,
        'le_target': le_target,
        'symptom_cols': symptom_cols.tolist()
    }
    
    print("Data preprocessing complete.")
    return df, feature_mapping

def train_model(df, target_col):
    """
    Splits the data and trains a RandomForestClassifier model.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"\nModel training complete (Random Forest).")
    print(f"Test Accuracy: {accuracy:.2f} (This is how well the model performed on unseen data).")
    
    return model, X.columns

def get_user_input(feature_names, symptom_cols):
    """
    Collects feature data from the user interactively.
    """
    user_data = {}
    print("\n--- Enter Patient Data for Prediction ---")

    gender_input = input("Enter GENDER (M/F): ").strip().upper()
    user_data['GENDER'] = 1 if gender_input == 'M' else 0

    while True:
        try:
            age_input = int(input("Enter AGE: ").strip())
            user_data['AGE'] = age_input
            break
        except ValueError:
            print("Invalid input. Please enter a valid number for AGE.")

    for feature in symptom_cols:
        while True:
            val = input(f"Does the patient exhibit '{feature}'? (1=NO / 2=YES): ").strip()
            if val in ['1', '2']:
                user_data[feature] = 1 if val == '2' else 0
                break
            else:
                print("Invalid input. Please enter 1 (NO) or 2 (YES).")

    input_df = pd.DataFrame([user_data], columns=feature_names)
    return input_df

def predict_and_report(model, user_df, feature_mapping):
    """
    Makes a prediction and provides the result and confidence.
    """
    
    prediction_encoded = model.predict(user_df)[0]
    
    probabilities = model.predict_proba(user_df)[0]
    
    target_le = feature_mapping['le_target']
    prediction_label = target_le.inverse_transform([prediction_encoded])[0]
    
    if prediction_label == 'YES':
        confidence_score = probabilities[1]
    else:
        confidence_score = probabilities[0]
        
    print("\n==============================================")
    print("           ** PREDICTION RESULT ** ")
    print("==============================================")
    print(f"Prediction: LUNG CANCER **{prediction_label}**")
    print(f"Confidence Score: {confidence_score:.4f}")
    
    confidence_percent = confidence_score * 100
    if confidence_percent > 90:
        confidence_text = "The model is **highly certain** of this prediction."
    elif confidence_percent > 75:
        confidence_text = "The model is **quite confident** in this result."
    elif confidence_percent > 60:
        confidence_text = "The model has **moderate confidence** in this prediction, but it's worth reviewing."
    else:
        confidence_text = "The model is **not highly confident**, indicating the features are close to the decision boundary."
        
    print(f"Confidentiality/Model Certainty: {confidence_text}")
    print("==============================================")

if __name__ == "__main__":
    try:
        df_processed, feature_mapping = load_and_preprocess_data(DATASET_PATH)
        
        trained_model, feature_names = train_model(df_processed, TARGET_COLUMN)
        
        user_input_df = get_user_input(feature_names, feature_mapping['symptom_cols'])
        
        predict_and_report(trained_model, user_input_df, feature_mapping)
        
    except FileNotFoundError:
        print(f"\nERROR: The file '{DATASET_PATH}' was not found.")
        print("Please ensure the CSV file is in the same directory as this Python script.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

