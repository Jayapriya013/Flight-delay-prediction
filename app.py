# app.py

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# Load LSTM model and scaler
model = load_model("lstm_model.h5")
scaler = joblib.load("scaler.pkl")

# Categorical encoding maps
origin_encoding = {"ATL": 0, "LAX": 1, "ORD": 2}
destination_encoding = {"SEA": 0, "JFK": 1, "SFO": 2}

# Function to preprocess input for LSTM
def preprocess_input(input_data):
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    return input_scaled.reshape(1, 1, -1)  # LSTM expects 3D input

# Page configuration
st.set_page_config(page_title="Flight Delay Prediction", layout="centered")

# Title
st.markdown(
    """
    <h2 style='text-align: center; color: #003366;'>FLIGHT DELAY PREDICTION</h2>
    """,
    unsafe_allow_html=True
)

# Input fields
flight_number = st.text_input("Enter the Flight Number", "1399")
month = st.text_input("Month", "1")
day_of_month = st.text_input("Day of Month", "1")
day_of_week = st.text_input("Day of Week", "5")

origin = st.selectbox("Origin", list(origin_encoding.keys()))
destination = st.selectbox("Destination", list(destination_encoding.keys()))

scheduled_departure = st.text_input("Scheduled Departure Time (HHMM)", "1905")
scheduled_arrival = st.text_input("Scheduled Arrival Time (HHMM)", "2143")
actual_departure = st.text_input("Actual Departure Time (HHMM)", "1901")

# Submit button
if st.button("Submit"):
    try:
        # Convert and encode input
        input_features = [
            int(flight_number),
            int(month),
            int(day_of_month),
            int(day_of_week),
            int(scheduled_departure),
            int(scheduled_arrival),
            int(actual_departure),
            origin_encoding[origin],
            destination_encoding[destination]
        ]

        # Preprocess and predict
        processed = preprocess_input(input_features)
        prediction = model.predict(processed)[0][0]

        # Output result
        result = "Delayed" if prediction > 0.5 else "On Time"
        st.success(f"Prediction: **{result}** (Confidence: {prediction:.2f})")

    except Exception as e:
        st.error(f"Error: {e}")
