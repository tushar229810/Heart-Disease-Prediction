import streamlit as st
import pandas as pd
import joblib

model = joblib.load("knn_heart_model.pkl")
scaler = joblib.load("heart_scaler.pkl")
expected_columns = joblib.load("heart_column.pkl")


st.title("Heart Stroke Prediction by Tushar")
st.markdown("Provide the following details")

age = st.slider("Age", 18,100,40)
sex = st.selectbox("SEX", ["M", "F"])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholestrol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
fating_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["0", "1"])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.slider("Max Heart Rate", 60, 220, 150)
excercise_angina = st.selectbox("Exercise Induced Angina", ["Y", "N"])
oldpeak = st.slider("Old Peak (ST depression)", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

if st.button("Predict"):
    raw_input = {
        "age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholestrol,
        "FastingBS": fating_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        "Sex" + sex: 1,
        "ChestPainType" + chest_pain: 1,
        "RestingECG" + resting_ecg: 1,
        "ExerciseAngina" + excercise_angina: 1,
        "ST_Slope" + st_slope: 1
       
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.error(" High Risk of Heart Disease")
    else:
        st.success(" Low Risk of Heart Disease")

