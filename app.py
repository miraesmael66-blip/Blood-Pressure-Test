import streamlit as st
import numpy as np
import joblib

# تحميل الموديل والـ scaler
model = joblib.load("hypertension_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Hypertension Prediction System")

# إدخال بيانات المستخدم
age = st.number_input("Age", 1, 120)
gender = st.selectbox("Gender (0 = Male, 1 = Female)", [0,1])
height = st.number_input("Height (cm)")
weight = st.number_input("Weight (kg)")
bmi = st.number_input("BMI")
sys = st.number_input("Systolic Blood Pressure")
dia = st.number_input("Diastolic Blood Pressure")
hr = st.number_input("Heart Rate")

if st.button("Predict"):
    # إنشاء مصفوفة بنفس ترتيب الأعمدة في التدريب
    x = np.array([[age, gender, height, weight, bmi, sys, dia, hr]])
    x_scaled = scaler.transform(x)
    pred = model.predict(x_scaled)

    if pred[0] == 1:
        st.error("Hypertension Detected")
    else:
        st.success("Normal Blood Pressure")