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
        location=[row['latitude'], row['longitude']],
        radius=5,
        color='red' if row['cluster'] == 0 else 'blue',
        fill=True
    ).add_to(map)

folium_static(map)

st.header("Check Accident Risk for a Location")
latitude = st.number_input("Enter Latitude", value=22.5726)
longitude = st.number_input("Enter Longitude", value=88.3639)

if st.button("Predict"):
    prediction = model.predict([[latitude, longitude]])
    st.write(f"This location is in cluster {prediction[0]} (0 = high risk).")