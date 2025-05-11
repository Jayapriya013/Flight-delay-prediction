# app.py

import streamlit as st
import numpy as np
import joblib

# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load("random_forest_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model_and_scaler()

# Encodings for categorical values
origin_encoding = {"ATL": 0, "LAX": 1, "ORD": 2}
destination_encoding = {"SEA": 0, "JFK": 1, "SFO": 2}

# Preprocessing
def preprocess_input(input_data):
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    return input_scaled

# UI Design
st.set_page_config(page_title="Flight Delay Prediction", layout="centered")

st.markdown("""
    <h2 style='text-align: center;'>FLIGHT DELAY PREDICTION</h2>
    <style>
        .stButton>button {
            background-color: #003366;
            color: white;
            padding: 0.5em 2em;
            border-radius: 8px;
            font-size: 16px;
        }
    </style>
    """, unsafe_allow_html=True)

# Form Input
flight_number = st.text_input("Enter the Flight Number", "1399")
month = st.text_input("Month", "1")
day_of_month = st.text_input("Day of Month", "1")
day_of_week = st.text_input("Day of Week", "5")
origin = st.selectbox("Origin", list(origin_encoding.keys()))
destination = st.selectbox("Destination", list(destination_encoding.keys()))
scheduled_departure = st.text_input("Scheduled Departure Time", "1905")
scheduled_arrival = st.text_input("Scheduled Arrival Time", "2143")
actual_departure = st.text_input("Actual Departure Time", "1901")

# Submit
if st.button("Submit"):
    try:
        features = [
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

        processed_input = preprocess_input(features)
        prediction = model.predict(processed_input)[0]

        # Output
        if prediction == 1:
            st.success("✈️ Prediction: **Delayed**")
        else:
            st.success("🟢 Prediction: **On Time**")

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
