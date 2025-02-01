import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import joblib

data = pd.read_csv("data/clustered_accidents.csv")

model = joblib.load("models/accident_model.pkl")

st.title("Kolkata Accident-Prone Areas Analysis")

st.header("Accident-Prone Areas in Kolkata")
map = folium.Map(location=[22.5726, 88.3639], zoom_start=12)

for index, row in data.iterrows():
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],  
        radius=5,
        color='red' if row['cluster'] == 0 else 'blue',  
        fill=True,
        popup=f"Place: {row['Place Name']}, Severity: {row['Accident Severity']}"  
    ).add_to(map)


folium_static(map)

st.header("Check Accident Risk for a Location")

place_names = data['Place Name'].unique()  
selected_place = st.selectbox("Select a Place", place_names)

selected_data = data[data['Place Name'] == selected_place].iloc[0]
latitude = selected_data['Latitude'] 
longitude = selected_data['Longitude']  

st.write(f"Selected Place: {selected_place}")
st.write(f"Coordinates: {latitude}, {longitude}")

if st.button("Predict Accident Risk"):
    prediction = model.predict([[latitude, longitude]])
    cluster = prediction[0]

    risk_levels = {
        0: "High Risk",
        1: "Medium Risk",
        2: "Low Risk",
        3: "Very Low Risk",
        4: "No Risk"
    }

    risk_level = risk_levels.get(cluster, "Unknown Risk")
    
    st.write(f"**Prediction:** This location is classified as **{risk_level}**.")