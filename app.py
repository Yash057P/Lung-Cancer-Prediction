import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

@st.cache_data
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    le_gender = LabelEncoder()
    df['GENDER'] = le_gender.fit_transform(df['GENDER'])

    le_target = LabelEncoder()
    df['LUNG_CANCER'] = le_target.fit_transform(df['LUNG_CANCER'])

    symptom_cols = df.columns.drop(['GENDER', 'AGE', 'LUNG_CANCER'])
    df[symptom_cols] = df[symptom_cols].replace({1: 0, 2: 1})

    return df, le_gender, le_target, symptom_cols

@st.cache_resource
def train_models(df):
    X = df.drop(columns=['LUNG_CANCER'])
    y = df['LUNG_CANCER']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logreg_model = Pipeline([
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression(random_state=42, solver='liblinear'))
    ])
    logreg_model.fit(X_train, y_train)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    return logreg_model, rf_model, X.columns

def predict(model, user_df, le_target):
    pred_encoded = model.predict(user_df)[0]
    prob = model.predict_proba(user_df)[0][pred_encoded]

    label = le_target.inverse_transform([pred_encoded])[0]
    comment = (
        "✅ Highly certain prediction." if prob > 0.5 else
        "🟡 Moderate confidence — consider additional checks." if prob > 0.3 and prob < 0.5 else
        "⚠️ Low confidence — results are uncertain."
    )

    return label, prob, comment

st.set_page_config(page_title="Lung Cancer Prediction", page_icon="🫁", layout="centered")

st.title("🫁 Lung Cancer Prediction System")
st.write("Answer a few questions to predict the likelihood of lung cancer using ML models.")

data_path = "survey lung cancer.csv"
df, le_gender, le_target, symptom_cols = load_and_preprocess_data(data_path)

logreg_model, rf_model, feature_cols = train_models(df)

st.sidebar.header("🧩 Model Selection")
model_choice = st.sidebar.selectbox("Select Model:", ["Logistic Regression", "Random Forest"])

st.header("Patient Details")

gender = st.radio("Gender:", ["Male", "Female"])
age = st.number_input("Age:", min_value=1, max_value=120, value=40)

st.subheader("Symptoms (Select Yes/No):")
user_data = {"GENDER": 1 if gender == "Male" else 0, "AGE": age}

for symptom in symptom_cols:
    val = st.radio(f"{symptom}:", ["No", "Yes"], horizontal=True, key=symptom)
    user_data[symptom] = 1 if val == "Yes" else 0

user_df = pd.DataFrame([user_data])

if st.button("🔍 Predict"):
    if model_choice == "Logistic Regression":
        model = logreg_model
    else:
        model = rf_model

    label, prob, comment = predict(model, user_df, le_target)

    st.markdown("---")
    st.subheader("📊 Prediction Result")
    st.markdown(f"**Prediction:** {label}")
    st.markdown(f"**Confidence:** {prob*100:.2f}%")
    st.markdown(comment)
    st.markdown("---")
    st.success("Prediction complete ✅")

