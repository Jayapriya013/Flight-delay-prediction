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
    st.error("❌ Model file 'flight_delay_model.pkl' not found. Please upload it to the same directory as app.py.")
    st.stop()

# -------- Prediction Function --------
def preprocess_and_predict(input_data):
    df = pd.DataFrame([input_data])

    # Placeholder for any preprocessing if your model needs it
    # For example: label encoding, scaling, feature engineering etc.

    prediction = model.predict(df)
    return prediction[0]

# -------- Streamlit UI --------
st.set_page_config(page_title="Flight Delay Predictor", layout="centered")
st.title("🛫 Flight Delay Prediction App")

# -------- Form Layout --------
with st.form("prediction_form"):
    airline = st.selectbox("Airline", ["AirlineA", "AirlineB", "AirlineC"])
    source = st.selectbox("Source Airport", ["JFK", "LAX", "ATL"])
    destination = st.selectbox("Destination Airport", ["ORD", "DFW", "DEN"])
    departure_time = st.number_input("Scheduled Departure Hour (0–23)", 0, 23)
    arrival_time = st.number_input("Scheduled Arrival Hour (0–23)", 0, 23)
    day_of_week = st.selectbox("Day of Week", [1, 2, 3, 4, 5, 6, 7])  # Monday = 1
    weather = st.selectbox("Weather Condition", ["Clear", "Rain", "Storm", "Snow"])

    submit = st.form_submit_button("Predict")

# -------- On Submit: Run Prediction --------
if submit:
    input_data = {
        "Airline": airline,
        "Source": source,
        "Destination": destination,
        "Scheduled_Departure_Hour": departure_time,
        "Scheduled_Arrival_Hour": arrival_time,
        "Day_of_Week": day_of_week,
        "Weather_Condition": weather
    }

    try:
        result = preprocess_and_predict(input_data)
        st.success(f"✈️ Prediction: **{result}**")
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")


