import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load data
data = pd.read_csv("data/kolkata_accidents.csv")

# Map accident severity to numerical values
severity_map = {"High": 0, "Medium": 1, "Low": 2}
data['severity_num'] = data['accident_severity'].map(severity_map)

# Convert time_frame to numerical values
time_frame_map = {"8-11 AM": 0, "12-3 PM": 1, "5-8 PM": 2}
data['time_frame_num'] = data['time_frame'].map(time_frame_map)

# Features and target
X = data[['latitude', 'longitude', 'time_frame_num']]  # Include time_frame as a feature
y = data['severity_num']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["High", "Medium", "Low"]))

# Save the model
joblib.dump(model, "models/accident_model.pkl")
print("Model trained and saved as accident_model.pkl")