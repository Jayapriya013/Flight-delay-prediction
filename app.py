import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Load the model
model_path = os.path.join(os.path.dirname(__file__), "flight_delay_model.pkl")
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error("Model file 'flight_delay_model.pkl' not found.")
    st.stop()

# Manual encodings (must match model training)
airline_map = {"AirlineA": 0, "AirlineB": 1, "AirlineC": 2}
airport_map = {"JFK": 0, "LAX": 1, "ATL": 2, "ORD": 3, "DFW": 4, "DEN": 5}
weather_map = {"Clear": 0, "Rain": 1, "Storm": 2, "Snow": 3}

def preprocess_and_predict(input_data):
    # Encode categorical values
    encoded_data = {
        "Airline": airline_map.get(input_data["Airline"], -1),
        "Source": airport_map.get(input_data["Source"], -1),
        "Destination": airport_map.get(input_data["Destination"], -1),
        "Scheduled_Departure_Hour": input_data["Scheduled_Departure_Hour"],
        "Scheduled_Arrival_Hour": input_data["Scheduled_Arrival_Hour"],
        "Day_of_Week": input_data["Day_of_Week"],
        "Weather_Condition": weather_map.get(input_data["Weather_Condition"], -1)
    }

    df = pd.DataFrame([encoded_data])
    return model.predict(df)[0]

# Streamlit UI
st.title("✈️ Flight Delay Predictor")

with st.form("prediction_form"):
    airline = st.selectbox("Airline", list(airline_map.keys()))
    source = st.selectbox("Source Airport", ["JFK", "LAX", "ATL"])
    destination = st.selectbox("Destination Airport", ["ORD", "DFW", "DEN"])
    dep_hour = st.number_input("Scheduled Departure Hour (0–23)", 0, 23)
    arr_hour = st.number_input("Scheduled Arrival Hour (0–23)", 0, 23)
    day = st.selectbox("Day of Week", [1, 2, 3, 4, 5, 6, 7])
    weather = st.selectbox("Weather Condition", list(weather_map.keys()))

    submit = st.form_submit_button("Predict")

if submit:
    input_data = {
        "Airline": airline,
        "Source": source,
        "Destination": destination,
        "Scheduled_Departure_Hour": dep_hour,
        "Scheduled_Arrival_Hour": arr_hour,
        "Day_of_Week": day,
        "Weather_Condition": weather
    }

    try:
        prediction = preprocess_and_predict(input_data)
        st.success(f"✅ Prediction: **{prediction}**")
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")
