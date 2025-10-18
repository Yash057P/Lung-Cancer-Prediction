
# Lung Cancer Prediction

🎯 Project Description:  

This repository contains an interactive machine learning framework built with Python and Scikit-learn to predict the likelihood of a patient having lung cancer. The model is trained on a synthetic survey dataset containing demographic information, lifestyle habits, and various symptoms.

The project currently supports two primary models: a Random Forest Classifier and a Logistic Regression model. Comparative analysis showed the Random Forest Classifier is the superior choice, as its non-linear structure correctly captured Age as the dominant risk factor, leading to more robust and clinically plausible predictions for high-risk patients.

🚀 Getting Started

Prerequisites:  
To run this project, you need Python 3.x installed on your system.

Installation:  
Clone this repository and install the required dependencies:

    # Clone the repository (if hosted on GitHub)
    # git clone <your-repo-link>
    # cd lung-cancer-prediction-model 
    # Install dependencies
    pip install pandas scikit-learn numpy matplotlib

Data Requirement:  
Ensure that the dataset file, survey lung cancer.csv, is placed in the same directory as the lung_cancer_predictor.py script.

🛠️ How to Run the Model:  
The prediction script is designed to run directly from your terminal:

For Random Forest Algorithm:

    python random_forest.py

Prediction Output:  
The output provides a clear result and an interpretation of the random forest model's certainty:

    =====================================================================================
                                  ** PREDICTION RESULT **
    =====================================================================================
    Prediction: LUNG CANCER **YES**
    Confidence Score: 0.9450
    Confidentiality/Model Certainty: The model is **highly  certain** of this prediction.
    =====================================================================================

For Logistic Regression Algorithm:

    python logistic_regression.py

Prediction Output:  
The output provides a clear result and an interpretation of the logistic regression model's certainty:

    =====================================================================================
                                  ** PREDICTION RESULT **
    =====================================================================================
    Prediction: LUNG CANCER **YES**
    Confidence Score: 0.9999
    Confidentiality/Model Certainty: The model is **highly  certain** of this prediction.
    =====================================================================================

💡 Future Enhancements  
Implement feature importance analysis (e.g., using feature_importances_) to determine which symptoms contribute most to the prediction.

Containerize the application using Docker.

Wrap the model in a web service (e.g., Flask or Streamlit) for a user-friendly GUI.
