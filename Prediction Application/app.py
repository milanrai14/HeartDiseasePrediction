import streamlit as st
import pandas as pd
import joblib

model = joblib.load('LogisticRegression.pkl')
scaler = joblib.load('scaler.pkl')
columns = joblib.load('columns.pkl')

st.title("Heart Stroke Prediction")
st.markdown("Provide the following details")
age = st.slider("Age", 18,100,40)
sex = st.selectbox("SEX", ["M", "F"])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
resting_bp = st.number_input("Resting BP (mm Hg)", 80, 200)
cholesterol = st.number_iput("Cholesterol (mg/DL)", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mh/dL", [0,1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.slider("Max Hear Rate", 60, 220, 150)
excersise_angina = st.selectbox("Exercise-Induced Angina", ['Y', 'N'])
oldpeak =st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
st_slop = st.selectbox("ST Slope", ["Up", "Flat", "Down"])


if st.button("Predict"):
    raw_input = {
        
    }