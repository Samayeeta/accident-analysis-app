import pandas as pd
import folium
from sklearn.cluster import KMeans

data = pd.read_csv("data/kolkata_accidents.csv")

print(data.head())

print(data.isnull().sum())

data = data.dropna()

kmeans = KMeans(n_clusters=5)  
data['cluster'] = kmeans.fit_predict(data[['latitude', 'longitude']])

data.to_csv("data/clustered_accidents.csv", index=False)

map = folium.Map(location=[22.5726, 88.3639], zoom_start=12) 

for index, row in data.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=5,
        color='red' if row['cluster'] == 0 else 'blue', 
        fill=True
    ).add_to(map)

map.save("accident_map.html")
print("Map saved as accident_map.html")