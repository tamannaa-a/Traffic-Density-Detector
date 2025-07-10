import streamlit as st
import pandas as pd
import joblib
import folium
from streamlit_folium import folium_static

# Load trained model and encoders
model = joblib.load("traffic_density_model.pkl")
encoders = joblib.load("label_encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")

# Define coordinates for cities
city_coords = {
    "New York": (40.7128, -74.0060),
    "Los Angeles": (34.0522, -118.2437),
    "Chicago": (41.8781, -87.6298)
}

st.set_page_config(page_title="🚦 Traffic Density Predictor with Map", layout="wide")
st.title("📍 Predict Urban Traffic Density")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        city = st.selectbox("City", list(city_coords.keys()))
        vehicle = st.selectbox("Vehicle Type", ["Car", "Bus", "Truck"])
        weather = st.selectbox("Weather Condition", ["Sunny", "Rainy", "Cloudy"])
        economy = st.selectbox("Economic Condition", ["Stable", "Declining", "Growing"])
        day = st.selectbox("Day of the Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

    with col2:
        hour = st.slider("Hour of Day", 0, 23, 9)
        speed = st.slider("Vehicle Speed (km/h)", 0, 120, 40)
        peak = st.selectbox("Is it Peak Hour?", [True, False])
        event = st.selectbox("Random Event Occurred?", [True, False])
        energy = st.slider("Energy Consumption (kWh)", 0.0, 100.0, 50.0)

    submit = st.form_submit_button("Predict")

if submit:
    input_df = pd.DataFrame([{
        "City": city,
        "Vehicle Type": vehicle,
        "Weather": weather,
        "Economic Condition": economy,
        "Day Of Week": day,
        "Hour Of Day": hour,
        "Speed": speed,
        "Is Peak Hour": peak,
        "Random Event Occurred": event,
        "Energy Consumption": energy
    }])

    # Encode all features using saved encoders
    for col in encoders:
        if col in input_df.columns:
            input_df[col] = encoders[col].transform(input_df[col])

    # Drop 'City' before model prediction
    model_input = input_df.drop(columns=["City"])

    # Make prediction
    prediction_encoded = model.predict(model_input)[0]
    prediction = target_encoder.inverse_transform([prediction_encoded])[0]

    st.success(f"🚗 Predicted Traffic Density in **{city}**: **{prediction}**")

    # Show location on map
    lat, lon = city_coords[city]
    m = folium.Map(location=[lat, lon], zoom_start=10)

    color_map = {"Low": "green", "Medium": "orange", "High": "red"}
    folium.CircleMarker(
        location=(lat, lon),
        radius=12,
        color=color_map.get(prediction, "blue"),
        fill=True,
        fill_opacity=0.8,
        popup=f"{city} - Traffic: {prediction}"
    ).add_to(m)

    st.subheader("📍 Traffic Density Location")
    folium_static(m)
