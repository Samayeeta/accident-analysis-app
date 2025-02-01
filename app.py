import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import joblib

# Load clustered data
data = pd.read_csv("data/kolkata_accidents.csv")

# Load trained model
model = joblib.load("models/accident_model.pkl")

# App title
st.title("Kolkata Accident-Prone Areas Analysis")

# Display the map
st.header("Accident-Prone Areas in Kolkata")
map = folium.Map(location=[22.5726, 88.3639], zoom_start=12)

# Add accident points to the map
for index, row in data.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=5,
        color='red' if row['accident_severity'] == 'High' else 'blue',
        fill=True,
        popup=f"Place: {row['place_name']}, Severity: {row['accident_severity']}, Time: {row['time_frame']}"
    ).add_to(map)

# Display the map in the app
folium_static(map)

# User input for place name and time
st.header("Check Accident Risk for a Location")
place_name = st.selectbox("Select a place", data['place_name'].unique())
time_slot = st.selectbox("What time is it approximately now?", ["8-11 AM", "12-3 PM", "5-8 PM"])

# Map time slot to numerical value
time_frame_map = {"8-11 AM": 0, "12-3 PM": 1, "5-8 PM": 2}
time_frame_num = time_frame_map[time_slot]

# Filter data for the selected place
filtered_data = data[data['place_name'] == place_name]

# Predict risk for the selected place and time
if not filtered_data.empty:
    avg_latitude = filtered_data['latitude'].mean()
    avg_longitude = filtered_data['longitude'].mean()
    prediction = model.predict([[avg_latitude, avg_longitude, time_frame_num]])  # Include time_frame_num

    # Map numerical predictions to risk levels
    risk_levels = {0: "High", 1: "Medium", 2: "Low"}
    predicted_risk = risk_levels.get(prediction[0], "Unknown")

    # Get typical severity and exact time from the database
    typical_data = filtered_data[filtered_data['time_frame'] == time_slot]
    if not typical_data.empty:
        typical_severity = typical_data['accident_severity'].mode()[0]  # Most common severity
        exact_time = typical_data['time'].values[0]  # Exact time from the database
    else:
        typical_severity = "Unknown"
        exact_time = "Unknown"

    # Display the predicted risk in a user-friendly way
    st.subheader(f"Predicted Accident Risk for {place_name} at {time_slot}")
    st.write(f"**Risk Level:** {predicted_risk}")
    st.write(f"**Details:** This location has a **{predicted_risk.lower()}** risk of accidents based on historical data.")
    st.write(f"**Typical Severity:** {typical_severity} during {exact_time}.")
else:
    st.subheader(f"No data found for {place_name} at {time_slot}.")
    st.write("However, based on our data, here are the accident severities for other time slots:")
    other_times = data[data['place_name'] == place_name][['time_frame', 'accident_severity']]
    st.write(other_times)