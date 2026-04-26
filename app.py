import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Train model (dummy data)
X = np.array([
    [80,30,40,3000],
    [95,50,55,4000],
    [70,20,35,2500],
    [110,60,65,4500]
])
y = [0,1,0,1]

model = RandomForestClassifier()
model.fit(X,y)

st.title("Engine Health Prediction 🚗")

temp = st.number_input("Temperature")
vib = st.number_input("Vibration")
press = st.number_input("Pressure")
rpm = st.number_input("RPM")

if st.button("Predict"):
    result = model.predict([[temp, vib, press, rpm]])
    
    if result[0] == 1:
        st.error("Engine is Faulty ⚠️")
    else:
        st.success("Engine is Healthy ✅")