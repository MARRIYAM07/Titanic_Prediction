import streamlit as st
import joblib
import os

# Build the model path relative to THIS file so it works on the cloud too
BASE = os.path.dirname(__file__)
model = joblib.load(os.path.join(BASE, 'heart_model.pkl'))

st.title("Heart Disease Prediction")

# One input widget per feature (same order as training!)
age = st.number_input("Age", min_value=1, max_value=120, value=50)
cp = st.selectbox("Chest pain type (1-4)", [1, 2, 3, 4])
chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=240)
thalach = st.number_input("Max heart rate achieved", min_value=60, max_value=220, value=150)

if st.button("Predict"):
    # Order must match: ['age', 'cp', 'chol', 'thalach']
    pred = model.predict([[age, cp, chol, thalach]])[0]
    prob = model.predict_proba([[age, cp, chol, thalach]])[0][1]
    if pred == 1:
        st.error(f"Likely heart disease (probability {prob:.0%})")
    else:
        st.success(f"Unlikely heart disease (probability {1 - prob:.0%})")
