import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

st.title("Water Potability Prediction")
st.write("Enter the water quality metrics below to check if the water is potable.")

# Create input fields for the features used in the project
ph = st.number_input("pH level", min_value=0.0, max_value=14.0, value=7.0)
hardness = st.number_input("Hardness", value=200.0)
solids = st.number_input("Solids (ppm)", value=20000.0)
chloramines = st.number_input("Chloramines (ppm)", value=7.0)
sulfate = st.number_input("Sulfate (mg/L)", value=300.0)
conductivity = st.number_input("Conductivity (μS/cm)", value=400.0)
organic_carbon = st.number_input("Organic Carbon (ppm)", value=10.0)
trihalomethanes = st.number_input("Trihalomethanes (μg/L)", value=60.0)
turbidity = st.number_input("Turbidity (NTU)", value=4.0)

# Prediction logic
if st.button("Predict Potability"):
    features = np.array([[ph, hardness, solids, chloramines, sulfate, 
                          conductivity, organic_carbon, trihalomethanes, turbidity]])
    prediction = model.predict(features)
    
    if prediction[0] == 1:
        st.success("The water is Potable (Safe to drink).")
    else:
        st.error("The water is Not Potable (Unsafe to drink).")
