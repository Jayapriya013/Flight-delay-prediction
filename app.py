import streamlit as st
import pandas as pd
import joblib
import os

# Load trained model
model_path = os.path.join(os.path.dirname(__file__), "flight_delay_model.pkl")
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error("Model file not found.")
    st.stop()

# Manual encoding - match your training code
airport_map = {"JFK": 0, "LAX": 1, "ATL": 2, "ORD": 3, "DFW": 4, "DEN": 5}
weather_map = {"Clear": 0, "Rain": 1, "Storm": 2, "Snow": 3}

def preprocess_input(data):
    return pd.DataFrame([{
        "Source": airport_map.get(data["Source"], -1),
        "Destination": airport_map.get(data["Destination"], -1),
        "Scheduled_Departure_Hour": data["Scheduled_Departure_Hour"],
        "Day_of_Week": data["Day_of_Week"],
        "Weather_Condition": weather_map.get(data["Weather_Condition"], -1)
    }])

# Streamlit UI
st.title("✈️ Flight Delay Predictor")

with st.form("prediction_form"):
    source = st.selectbox("Source Airport", list(airport_map.keys()))
    destination = st.selectbox("Destination Airport", list(airport_map.keys()))
    dep_hour = st.number_input("Scheduled Departure Hour (0–23)", min_value=0, max_value=23)
    day_of_week = st.selectbox("Day of Week (1=Mon, 7=Sun)", list(range(1, 8)))
    weather = st.selectbox("Weather Condition", list(weather_map.keys()))

    submitted = st.form_submit_button("Predict")

if submitted:
    input_data = {
        "Source": source,
        "Destination": destination,
        "Scheduled_Departure_Hour": dep_hour,
        "Day_of_Week": day_of_week,
        "Weather_Condition": weather
    }

    try:
        features = preprocess_input(input_data)
        prediction = model.predict(features)[0]
        st.success(f"✅ Prediction: {'Delayed' if prediction == 1 else 'On Time'}")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

   
