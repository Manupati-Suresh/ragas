
import streamlit as st
import pandas as pd
import pickle
import os

st.title("🩺 Diabetes Prediction")

# Check if model files exist, if not, train the model
if not os.path.exists("logistic_model.pkl") or not os.path.exists("scaler.pkl"):
    with st.spinner("Training model for first time... This may take a moment."):
        exec(open("train_model.py").read())
    st.success("Model trained successfully!")

# Load model and scaler
model = pickle.load(open("logistic_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Input fields
pregnancies = st.slider("Pregnancies", 0, 17, 3)
glucose = st.slider("Glucose", 0, 200, 120)
bp = st.slider("Blood Pressure", 0, 122, 70)
skin = st.slider("Skin Thickness", 0, 99, 20)
insulin = st.slider("Insulin", 0, 846, 79)
bmi = st.slider("BMI", 0.0, 67.1, 32.0)
pedigree = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.47)
age = st.slider("Age", 21, 90, 33)

input_df = pd.DataFrame([[pregnancies, glucose, bp, skin, insulin, bmi, pedigree, age]],
                         columns=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"])
input_scaled = scaler.transform(input_df)

if st.button("Predict"):
    pred = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0][1]
    label = "Diabetic" if pred == 1 else "Non‑Diabetic"
    st.success(f"Prediction: **{label}** (probability: {proba:.2f})")
