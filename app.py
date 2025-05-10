import streamlit as st
import joblib
from datetime import datetime
import pandas as pd

# Load the model
model = joblib.load("flight_delay_model.pkl")

# Prediction logic
def preprocess_and_predict(data_dict):
    data_dict['Scheduled_Departure_Time'] = pd.to_datetime(data_dict['Scheduled_Departure_Time'])
    data_dict['Actual_Departure_Time'] = pd.to_datetime(data_dict['Actual_Departure_Time'])
    data_dict['Scheduled_Arrival_Time'] = pd.to_datetime(data_dict['Scheduled_Arrival_Time'])
    data_dict['Actual_Arrival_Time'] = pd.to_datetime(data_dict['Actual_Arrival_Time'])

    df = pd.DataFrame([data_dict])
    df['Dep_Hour'] = df['Scheduled_Departure_Time'].dt.hour
    df['Dep_DayOfWeek'] = df['Scheduled_Departure_Time'].dt.dayofweek
    df['Dep_Month'] = df['Scheduled_Departure_Time'].dt.month
    df['Flight_Duration_Min'] = (df['Scheduled_Arrival_Time'] - df['Scheduled_Departure_Time']).dt.total_seconds() / 60
    df['Dep_Delay_Min'] = (df['Actual_Departure_Time'] - df['Scheduled_Departure_Time']).dt.total_seconds() / 60
    df['Arr_Delay_Min'] = (df['Actual_Arrival_Time'] - df['Scheduled_Arrival_Time']).dt.total_seconds() / 60

    categorical_cols = ['Airline', 'Origin_Airport', 'Destination_Airport', 'Weather_Condition']
    df = pd.get_dummies(df, columns=categorical_cols)
    
    model_features = model.feature_names_in_
    for col in model_features:
        if col not in df.columns:
            df[col] = 0
    df = df[model_features]

    prediction = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]
    return ("Delayed" if prediction == 1 else "On-Time", round(prob * 100, 2))

# Streamlit UI
st.title("Flight Delay Prediction")

with st.form("flight_form"):
    airline = st.text_input("Airline")
    origin = st.text_input("Origin Airport")
    destination = st.text_input("Destination Airport")
    weather = st.selectbox("Weather Condition", ["Clear", "Rain", "Fog", "Storm"])
    sched_dep = st.datetime_input("Scheduled Departure Time")
    actual_dep = st.datetime_input("Actual Departure Time")
    sched_arr = st.datetime_input("Scheduled Arrival Time")
    actual_arr = st.datetime_input("Actual Arrival Time")

    submitted = st.form_submit_button("Predict")

    if submitted:
        data = {
            "Airline": airline,
            "Origin_Airport": origin,
            "Destination_Airport": destination,
            "Weather_Condition": weather,
            "Scheduled_Departure_Time": sched_dep,
            "Actual_Departure_Time": actual_dep,
            "Scheduled_Arrival_Time": sched_arr,
            "Actual_Arrival_Time": actual_arr
        }
        result, confidence = preprocess_and_predict(data)
        st.success(f"Prediction: {result} ({confidence}% confidence)")
