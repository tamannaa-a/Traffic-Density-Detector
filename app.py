import streamlit as st
import pandas as pd
import joblib

# Load model + encoders
model = joblib.load("traffic_density_model.pkl")
encoders = joblib.load("encoders.pkl")

st.title("🚦 Predict Urban Traffic Density")

# Input UI
city = st.selectbox("City", ["New York", "Los Angeles", "Chicago"])
vehicle_type = st.selectbox("Vehicle Type", ["Car", "Bus", "Truck", "SUV"])
weather = st.selectbox("Weather Condition", ["Sunny", "Rainy", "Cloudy", "Snowy"])
economic = st.selectbox("Economic Condition", ["Stable", "Declining", "Recession"])
day = st.selectbox("Day of the Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
hour = st.slider("Hour of Day", 0, 23, 8)
speed = st.slider("Vehicle Speed (km/h)", 0, 120, 40)
peak = st.selectbox("Is It Peak Hour?", ["True", "False"])
event = st.selectbox("Random Event Occurred?", ["True", "False"])
energy = st.slider("Energy Consumption (kWh)", 0.0, 100.0, 50.0)

if st.button("Predict"):
    input_data = pd.DataFrame([{
        "City": city,
        "Vehicle Type": vehicle_type,
        "Weather": weather,
        "Economic Condition": economic,
        "Day Of Week": day,
        "Hour Of Day": hour,
        "Speed": speed,
        "Is Peak Hour": peak,
        "Random Event Occurred": event,
        "Energy Consumption": energy
    }])

    # Encode using saved encoders
    for col in encoders:
        input_data[col] = encoders[col].transform(input_data[col])

    prediction = model.predict(input_data)[0]
    decoded_prediction = encoders['Traffic Density'].inverse_transform([prediction])[0]
    st.success(f"🚗 Predicted Traffic Density in {city}: **{decoded_prediction}**")
