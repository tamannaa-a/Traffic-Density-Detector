import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("traffic_density_model.pkl")
encoders = joblib.load("encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")

st.set_page_config(page_title="Traffic Density Predictor", layout="centered")
st.title("🚦 Predict Urban Traffic Density")

st.markdown("Provide the following inputs to estimate traffic congestion.")

with st.form("predict_form"):
    vehicle = st.selectbox("Vehicle Type", encoders['Vehicle Type'].classes_)
    weather = st.selectbox("Weather Condition", encoders['Weather'].classes_)
    economy = st.selectbox("Economic Condition", encoders['Economic Condition'].classes_)
    day = st.selectbox("Day of the Week", encoders['Day Of Week'].classes_)
    hour = st.slider("Hour of Day", 0, 23, 8)
    speed = st.slider("Vehicle Speed (km/h)", 0, 120, 40)
    peak_hour = st.selectbox("Is it Peak Hour?", encoders['Is Peak Hour'].classes_)
    event = st.selectbox("Random Event Occurred?", encoders['Random Event Occurred'].classes_)
    energy = st.slider("Energy Consumption (kWh)", 0.0, 100.0, 50.0)

    submitted = st.form_submit_button("Predict")

if submitted:
    input_data = {
        "Vehicle Type": encoders['Vehicle Type'].transform([vehicle])[0],
        "Weather": encoders['Weather'].transform([weather])[0],
        "Economic Condition": encoders['Economic Condition'].transform([economy])[0],
        "Day Of Week": encoders['Day Of Week'].transform([day])[0],
        "Hour Of Day": hour,
        "Speed": speed,
        "Is Peak Hour": encoders['Is Peak Hour'].transform([peak_hour])[0],
        "Random Event Occurred": encoders['Random Event Occurred'].transform([event])[0],
        "Energy Consumption": energy,
    }

    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    label = target_encoder.inverse_transform([prediction])[0]

    # Color-coded result
    color = {"Low": "green", "Medium": "orange", "High": "red"}.get(label, "gray")
    st.markdown(f"""
        <div style='padding:1rem;text-align:center;background-color:{color};color:white;border-radius:10px;font-size:24px'>
            Predicted Traffic Density: <b>{label}</b>
        </div>
    """, unsafe_allow_html=True)
