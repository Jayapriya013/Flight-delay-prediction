import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# -------- Load the model safely --------
model_path = os.path.join(os.path.dirname(__file__), "flight_delay_model.pkl")

try:
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error("❌ Model file 'flight_delay_model.pkl' not found. Please ensure it's in the same folder as this script.")
    st.stop()

# -------- Preprocessing and prediction logic --------
def preprocess_and_predict(input_data):
    # Sample preprocessing - update this according to your model requirements
    df = pd.DataFrame([input_data])
    
    # If you have encoding/scaling, add it here

    # Prediction
    prediction = model.predict(df)
    return prediction[0]

# -------- Streamlit UI --------
st.title("✈️ Flight Delay Prediction App")

# Example input fields - adjust these based on your model
airline = st.selectbox("Airline", ["AirlineA", "AirlineB", "AirlineC"])
source = st.selectbox("Source Airport", ["JFK", "LAX", "ATL"])
destination = st.selectbox("Destination Airport", ["ORD", "DFW", "DEN"])
departure_time = st.number_input("Scheduled Departure Hour (0–23)", 0, 23)
arrival_time = st.number_input("Scheduled Arrival Hour (0–23)", 0, 23)
day_of_week = st.selectbox("Day of Week", [1, 2, 3, 4, 5, 6, 7])  # 1 = Monday

# Collect data
input_data = {
    "Airline": airline,
    "Source": source,
    "Destination": destination,
    "Scheduled_Departure_Hour": departure_time,
    "Scheduled_Arrival_Hour": arrival_time,
    "Day_of_Week": day_of_week
}

# Prediction button
if st.button("Predict Delay Status"):
    result = preprocess_and_predict(input_data)
    st.success(f"✈️ Prediction: **{result}**")

