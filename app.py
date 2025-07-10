import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="🚦 Predict Urban Traffic Density", layout="wide")
st.title("🚦 Predict Urban Traffic Density")

# Load model and encoders
model = joblib.load("traffic_density_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

st.write("Provide the following inputs to estimate traffic congestion.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        city = st.selectbox("City", ['New York', 'Los Angeles', 'Chicago'])
        vehicle_type = st.selectbox("Vehicle Type", ['Car', 'Bus', 'Bike'])
        weather = st.selectbox("Weather Condition", ['Sunny', 'Rainy', 'Cloudy'])
        economy = st.selectbox("Economic Condition", ['Stable', 'Declining', 'Growing'])
        day_of_week = st.selectbox("Day of the Week", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

    with col2:
        hour = st.slider("Hour of Day", 0, 23, 8)
        speed = st.slider("Vehicle Speed (km/h)", 0, 120, 40)
        is_peak = st.selectbox("Is it Peak Hour?", [True, False])
        event = st.selectbox("Random Event Occurred?", [True, False])
        energy = st.slider("Energy Consumption (kWh)", 0.0, 100.0, 50.0)

    submit = st.form_submit_button("Predict")

if submit:
    input_data = pd.DataFrame({
        'City': [city],
        'Vehicle Type': [vehicle_type],
        'Weather': [weather],
        'Economic Condition': [economy],
        'Day Of Week': [day_of_week],
        'Hour Of Day': [hour],
        'Speed': [speed],
        'Is Peak Hour': [is_peak],
        'Random Event Occurred': [event],
        'Energy Consumption': [energy]
    })

    # Encode categorical features
    for col in label_encoders:
        input_data[col] = label_encoders[col].transform(input_data[col])

    # Predict
    prediction = model.predict(input_data)[0]
    st.success(f"🚗 Predicted Traffic Density: **{prediction}**")
