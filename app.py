import streamlit as st
import pandas as pd
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import folium_static

st.set_page_config(page_title="📍 Traffic Density Map", layout="wide")
st.title("🚦 Traffic Density Mapping")

uploaded_file = st.file_uploader("📁 Upload your traffic dataset (CSV)", type=["csv"])

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)

    required_cols = [
        'City', 'Vehicle Type', 'Weather', 'Economic Condition', 'Day Of Week',
        'Hour Of Day', 'Speed', 'Is Peak Hour', 'Random Event Occurred',
        'Energy Consumption', 'Traffic Density'
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"❌ Your CSV is missing required columns: {missing_cols}")
        st.stop()

    df['City'] = df['City'].str.strip().str.title()

    city_coords = {
        'New York': (40.7128, -74.0060),
        'Los Angeles': (34.0522, -118.2437),
        'Chicago': (41.8781, -87.6298)
    }
    df['Latitude'] = df['City'].map(lambda x: city_coords.get(x, (None, None))[0])
    df['Longitude'] = df['City'].map(lambda x: city_coords.get(x, (None, None))[1])

    return df.dropna(subset=['Latitude', 'Longitude'])

if uploaded_file is not None:
    df = load_data(uploaded_file)

    # Sidebar filters
    with st.sidebar:
        st.header("🔍 Filter Traffic Data")

        selected_cities = st.multiselect("City", df["City"].unique(), default=df["City"].unique())
        selected_vehicles = st.multiselect("Vehicle Type", df["Vehicle Type"].unique(), default=df["Vehicle Type"].unique())
        selected_weather = st.multiselect("Weather", df["Weather"].unique(), default=df["Weather"].unique())
        selected_economy = st.multiselect("Economic Condition", df["Economic Condition"].unique(), default=df["Economic Condition"].unique())
        selected_day = st.multiselect("Day of Week", df["Day Of Week"].unique(), default=df["Day Of Week"].unique())
        hour_range = st.slider("Hour of Day", 0, 23, (0, 23))
        speed_range = st.slider("Speed Range", int(df["Speed"].min()), int(df["Speed"].max()), (int(df["Speed"].min()), int(df["Speed"].max())))
        selected_peak = st.selectbox("Is Peak Hour", options=["Both", "Yes", "No"])
        selected_event = st.selectbox("Random Event Occurred", options=["Both", "Yes", "No"])
        energy_range = st.slider("Energy Consumption", float(df["Energy Consumption"].min()), float(df["Energy Consumption"].max()), (float(df["Energy Consumption"].min()), float(df["Energy Consumption"].max())))
        selected_density = st.multiselect("Traffic Density", df["Traffic Density"].unique(), default=df["Traffic Density"].unique())

    # Filter the data based on sidebar inputs
    filtered_df = df[
        (df["City"].isin(selected_cities)) &
        (df["Vehicle Type"].isin(selected_vehicles)) &
        (df["Weather"].isin(selected_weather)) &
        (df["Economic Condition"].isin(selected_economy)) &
        (df["Day Of Week"].isin(selected_day)) &
        (df["Hour Of Day"].between(hour_range[0], hour_range[1])) &
        (df["Speed"].between(speed_range[0], speed_range[1])) &
        (df["Energy Consumption"].between(energy_range[0], energy_range[1])) &
        (df["Traffic Density"].isin(selected_density))
    ]

    if selected_peak != "Both":
        filtered_df = filtered_df[filtered_df["Is Peak Hour"] == selected_peak]

    if selected_event != "Both":
        filtered_df = filtered_df[filtered_df["Random Event Occurred"] == selected_event]

    # Visualization on map
    density_map = {'Low': 1, 'Medium': 2, 'High': 3}
    color_map = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}
    filtered_df['Weight'] = filtered_df['Traffic Density'].map(density_map).fillna(2)

    st.subheader("🗺️ Interactive Traffic Map")
    map_center = [filtered_df['Latitude'].mean(), filtered_df['Longitude'].mean()]
    m = folium.Map(location=map_center, zoom_start=5)

    heat_data = [[row['Latitude'], row['Longitude'], row['Weight']] for _, row in filtered_df.iterrows()]
    HeatMap(heat_data, radius=15, blur=10, min_opacity=0.3).add_to(m)

    for _, row in filtered_df.iterrows():
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=6,
            color=color_map.get(row['Traffic Density'], 'gray'),
            fill=True,
            fill_opacity=0.8,
            popup=f"{row['City']} - {row['Traffic Density']}"
        ).add_to(m)

    folium_static(m)
    st.download_button("⬇️ Download Filtered CSV", filtered_df.to_csv(index=False), file_name="filtered_traffic.csv")

else:
    st.info("Please upload a CSV file to visualize traffic density on the map.")
