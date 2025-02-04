import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import joblib
import numpy as np
from folium.plugins import HeatMap
import os
from geopy.geocoders import Nominatim

# --- Data and Model Loading ---
DATA_FILE = os.path.join("data", "kolkata_accidents.csv")
MODEL_FILE = os.path.join("models", "accident_model.pkl")
USER_REPORTS_FILE = os.path.join("data", "user_reports.csv")

try:
    data = pd.read_csv(DATA_FILE)
    model = joblib.load(MODEL_FILE)
    try:
        user_reports = pd.read_csv(USER_REPORTS_FILE)
    except FileNotFoundError:
        user_reports = pd.DataFrame(columns=data.columns)
except FileNotFoundError:
    st.error("Data or model files not found.")
    st.stop()

# --- Data Preprocessing ---
time_frame_map = {"8-11 AM": 0, "12-3 PM": 1, "5-8 PM": 2}
data['time_frame_num'] = data['time_frame'].map(time_frame_map)
severity_map = {"High": 0, "Medium": 1, "Low": 2}
data['severity_num'] = data['accident_severity'].map(severity_map)

# --- Geocoding ---
def geocode_location(location):
    geolocator = Nominatim(user_agent="accident_app")
    try:
        location_info = geolocator.geocode(location)
        if location_info:
            return location_info.latitude, location_info.longitude
        else:
            return None, None
    except Exception as e:
        st.error(f"Geocoding error: {e}")
        return None, None


# --- Streamlit App ---
st.title("Kolkata Accident Risk Assessment")

# --- Accident Hotspots ---
st.header("Accident Hotspots")
heatmap_map = folium.Map(location=[22.5726, 88.3639], zoom_start=12)
heat_data = [[row['latitude'], row['longitude']] for _, row in data.iterrows()]
HeatMap(heat_data, radius=8, blur=5).add_to(heatmap_map)
folium_static(heatmap_map)

# --- User Reports and Data Update ---
st.header("Report an Accident")

with st.form("report_form"):
    report_place = st.text_input("Location of Incident")
    report_time = st.selectbox("Time of Incident", ["8-11 AM", "12-3 PM", "5-8 PM"])
    report_severity = st.selectbox("Severity", ["High", "Medium", "Low"])
    submitted = st.form_submit_button("Submit Report")

if submitted:
    if not report_place:
        st.warning("Please provide a location.")
    else:
        lat, lon = geocode_location(report_place)
        if lat is not None and lon is not None:
            new_report = pd.DataFrame([{
                'place_name': report_place,
                'latitude': lat,
                'longitude': lon,
                'accident_severity': report_severity,
                'time_frame': report_time,
                'date': pd.Timestamp.now().strftime("%Y-%m-%d"),
                'time': pd.Timestamp.now().strftime("%I:%M %p"),
                'time_frame_num': time_frame_map[report_time],
                'severity_num': severity_map[report_severity]
            }])

            # Update data (and user reports)
            data = pd.concat([data, new_report], ignore_index=True)
            data.to_csv(DATA_FILE, index=False)

            user_reports = pd.concat([user_reports, new_report], ignore_index=True)
            user_reports.to_csv(USER_REPORTS_FILE, index=False)
            st.success("Report submitted!")

            # Update Heatmap
            heatmap_map = folium.Map(location=[22.5726, 88.3639], zoom_start=12)
            heat_data = [[row['latitude'], row['longitude']] for _, row in data.iterrows()]
            HeatMap(heat_data, radius=8, blur=5).add_to(heatmap_map)
            folium_static(heatmap_map)

        else:
            st.warning("Could not find coordinates. Please be more specific.")


# --- Accident Risk Assessment ---
st.header("Accident Risk Check")
place_name = st.text_input("Enter a Location")
time_slot = st.selectbox("Time of Day", ["8-11 AM", "12-3 PM", "5-8 PM"])

if place_name:
    filtered_data = data[data['place_name'].str.contains(place_name, case=False, na=False)]
    user_reports_filtered = user_reports[user_reports['place_name'].str.contains(place_name, case=False, na=False)] #filter user report

    if not filtered_data.empty:
        avg_latitude = filtered_data['latitude'].mean()
        avg_longitude = filtered_data['longitude'].mean()
        time_frame_num = time_frame_map[time_slot]

        try:
            prediction = model.predict([[avg_latitude, avg_longitude, time_frame_num]])
            risk_levels = {0: "High", 1: "Medium", 2: "Low"}
            predicted_risk = risk_levels.get(prediction[0], "Unknown")

            st.subheader(f"Accident Risk for {place_name} at {time_slot}: {predicted_risk}")

            # User Report Count
            high_reports = user_reports_filtered[user_reports_filtered['accident_severity'] == 'High'].shape[0]
            st.write(f"{high_reports} user(s) reported this area as HIGH risk.") #display user report count

            # Safety Tips
            if predicted_risk == "High":
                st.write("⚠️ **Safety Tips:** Be extra cautious...")
            elif predicted_risk == "Medium":
                st.write("⚠️ **Safety Tips:** Exercise caution...")
            elif predicted_risk == "Low":
                st.write("✅ **Safety Tips:** While the risk is lower...")

            st.write("Historical Accident Data for this location:")
            st.dataframe(filtered_data[['place_name', 'accident_severity', 'time_frame', 'date', 'time']])

        except ValueError:
            st.error("Prediction error. Check input or try again.")

    else:
        st.write(f"No accident data found for '{place_name}'.")