import pandas as pd
import folium
from sklearn.cluster import KMeans

# Load data
data = pd.read_csv("data/kolkata_accidents.csv")

# Display first few rows
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Drop missing values
data = data.dropna()

# Cluster accident locations using KMeans
kmeans = KMeans(n_clusters=5)  # 5 clusters for high-risk zones
data['cluster'] = kmeans.fit_predict(data[['latitude', 'longitude']])  # Use lowercase column names

# Save clustered data
data.to_csv("data/clustered_accidents.csv", index=False)

# Create a map of accident-prone areas
map = folium.Map(location=[22.5726, 88.3639], zoom_start=12)  # Kolkata coordinates

# Add accident points to the map
for index, row in data.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=5,
        color='red' if row['cluster'] == 0 else 'blue',  # Highlight high-risk clusters
        fill=True,
        popup=f"Place: {row['place_name']}, Severity: {row['accident_severity']}"
    ).add_to(map)

# Save the map
map.save("accident_map.html")
print("Map saved as accident_map.html")