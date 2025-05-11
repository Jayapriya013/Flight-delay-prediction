# app.py

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib

# Load model and scaler
model = load_model("lstm_model.h5")
scaler = joblib.load("scaler.pkl")

# Encoding maps
origin_encoding = {"ATL": 0, "LAX": 1, "ORD": 2}
destination_encoding = {"SEA": 0, "JFK": 1, "SFO": 2}

def preprocess_input(input_data):
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    return input_scaled.reshape(1, 1, -1)  # For LSTM

# UI Setup
st.set_page_config(page_title="Flight Delay Prediction", layout="centered")

st.markdown("<h2 style='text-align: center;'>FLIGHT DELAY PREDICTION</h2>", unsafe_allow_html=True)

# Input Fields
flight_number = st.text_input("Enter the Flight Number", "1399")
month = st.text_input("Month", "1")
day_of_month = st.text_input("Day of Month", "1")
day_of_week = st.text_input("Day of Week", "5")

origin = st.selectbox("Origin", list(origin_encoding.keys()))
destination = st.selectbox("Destination", list(destination_encoding.keys()))

scheduled_departure = st.text_input("Scheduled Departure Time", "1905")
scheduled_arrival = st.text_input("Scheduled Arrival Time", "2143")
actual_departure = st.text_input("Actual Departure Time", "1901")

if st.button("Submit"):
    try:
        # Prepare input
        input_features = [

            int(airport_name),
            int(month),
            int(year),
            int(scheduled_departure),
            int(scheduled_arrival),
            int(actual_departure),
            origin_encoding[origin],
            destination_encoding[destination]
        ]

        processed = preprocess_input(input_features)
        prediction = model.predict(processed)[0][0]

        result = "Delayed" if prediction > 0.5 else "On Time"
        st.success(f"Prediction: **{result}**")

    except Exception as e:
        st.error(f"Error: {e}")
