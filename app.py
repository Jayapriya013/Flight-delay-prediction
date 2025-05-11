# app.py

import streamlit as st
import numpy as np
import joblib
from keras.models import load_model  # Use keras (not tensorflow.keras) for compatibility

# Load model and scaler
@st.cache_resource
def load_lstm_model():
    return load_model("lstm_model.h5")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

model = load_lstm_model()
scaler = load_scaler()

# Encoding maps
origin_encoding = {"ATL": 0, "LAX": 1, "ORD": 2}
destination_encoding = {"SEA": 0, "JFK": 1, "SFO": 2}

# Input preprocessing
def preprocess_input(input_data):
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    return input_scaled.reshape(1, 1, -1)  # for LSTM

# Streamlit UI
st.set_page_config(page_title="Flight Delay Prediction", layout="centered")
st.markdown("<h2 style='text-align: center;'>FLIGHT DELAY PREDICTION</h2>", unsafe_allow_html=True)

# Input form
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

        processed_input = preprocess_input(input_features)
        prediction = model.predict(processed_input)[0][0]

        result = "Delayed" if prediction > 0.5 else "On Time"
        st.success(f"Prediction: **{result}** (Confidence: {prediction:.2f})")

    except Exception as e:
        st.error(f"Error processing input: {e}")
