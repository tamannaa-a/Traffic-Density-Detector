import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("traffic_density_model.pkl")
encoders = joblib.load("encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")

st.header("🚗 Predict Traffic Density")
st.markdown("Enter traffic and environmental factors to predict traffic density.")

with st.form("prediction_form"):
    city = st.selectbox("City", ["New York", "Los Angeles", "Chicago"])  # Optional UI, not used for prediction
    vehicle = st.selectbox("Vehicle Type", encoders['Vehicle Type'].classes_)
    weather = st.selectbox("Weather", encoders['Weather'].classes_)
    economy = st.selectbox("Economic Condition", encoders['Economic Condition'].classes_)
    day = st.selectbox("Day of Week", encoders['Day Of Week'].classes_)
    hour = st.slider("Hour of Day", 0, 23, 8)
    speed = st.slider("Vehicle Speed (km/h)", 0, 120, 40)
    peak_hour = st.selectbox("Is Peak Hour?", encoders['Is Peak Hour'].classes_)
    event_occurred = st.selectbox("Random Event Occurred?", encoders['Random Event Occurred'].classes_)
    energy = st.slider("Energy Consumption", 0.0, 100.0, 50.0)

    submit = st.form_submit_button("Predict")

if submit:
    input_dict = {
        "Vehicle Type": encoders['Vehicle Type'].transform([vehicle])[0],
        "Weather": encoders['Weather'].transform([weather])[0],
        "Economic Condition": encoders['Economic Condition'].transform([economy])[0],
        "Day Of Week": encoders['Day Of Week'].transform([day])[0],
        "Hour Of Day": hour,
        "Speed": speed,
        "Is Peak Hour": encoders['Is Peak Hour'].transform([peak_hour])[0],
        "Random Event Occurred": encoders['Random Event Occurred'].transform([event_occurred])[0],
        "Energy Consumption": energy,
    }

    input_df = pd.DataFrame([input_dict])
    prediction = model.predict(input_df)[0]
    predicted_label = target_encoder.inverse_transform([prediction])[0]

    # Display result
    st.subheader("📊 Prediction Result:")
    color_map = {"Low": "green", "Medium": "orange", "High": "red"}
    st.markdown(f"""
        <div style='padding:10px;background-color:{color_map[predicted_label]};color:white;font-size:24px;border-radius:8px;text-align:center'>
            Predicted Traffic Density: <b>{predicted_label}</b>
        </div>
    """, unsafe_allow_html=True)
