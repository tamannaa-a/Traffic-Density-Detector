import streamlit as st
import pandas as pd
import joblib
import folium
from streamlit_folium import folium_static

# Set Streamlit page configuration
st.set_page_config(page_title="GridGaze – City Traffic Predictor", page_icon="🚦", layout="wide")

# Centered branding
st.markdown("""
<div style='text-align: center;'>
    <h1>🚦 <b>GridGaze</b></h1>
    <p style='font-size:16px;'><i>A new way to look at city congestion.</i></p>
</div>
""", unsafe_allow_html=True)

# Load trained model and encoders
model = joblib.load("traffic_density_model.pkl")
encoders = joblib.load("label_encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")

# Define coordinates for supported cities
city_coords = {
    "New York": (40.7128, -74.0060),
    "Los Angeles": (34.0522, -118.2437),
    "Chicago": (41.8781, -87.6298)
}

st.subheader("📍 Predict Urban Traffic Density")

# User input form
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

# If user submits form
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

    # Encode features
    for col in encoders:
        if col in input_df.columns:
            input_df[col] = encoders[col].transform(input_df[col])

    # Drop city for prediction
    model_input = input_df.drop(columns=["City"])

    # Predict traffic density
    prediction_encoded = model.predict(model_input)[0]
    prediction = target_encoder.inverse_transform([prediction_encoded])[0]

    st.success(f"🚗 Predicted Traffic Density in **{city}**: **{prediction}**")

    # Map output
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

    st.subheader("🗺️ Traffic Location")
    folium_static(m)
