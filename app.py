from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import requests
import os

# Load the saved model
with open("models/final_gradient_boosting_model.pkl", "rb") as model_file:
    gb_model = pickle.load(model_file)

# Load the saved K-Means model for clustering
with open("models/kmeans_model.pkl", "rb") as kmeans_file:
    kmeans = pickle.load(kmeans_file)

# Constants
TORONTO_CENTER_LAT = 43.6532
TORONTO_CENTER_LONG = -79.3832
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

app = Flask(__name__)

# Helper function to get coordinates from address
def get_coordinates(address):
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={address}&key={GOOGLE_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data['results']:
            location = data['results'][0]['geometry']['location']
            return location['lat'], location['lng']
    return None, None

# Helper function to calculate distance from Toronto center
def calculate_distance(lat, lon):
    """Calculate the distance from Toronto's center."""
    return np.sqrt((lat - TORONTO_CENTER_LAT)**2 + (lon - TORONTO_CENTER_LONG)**2)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """API Endpoint to predict severity based on input features."""
    try:
        # Get form data
        address = request.form.get("address")
        time = request.form.get("time")
        date = request.form.get("date")

        # Ensure all fields received
        if not (address and time and date):
            raise ValueError("Please provide all fields.")
        
        # Geocode the address
        latitude, longitude = get_coordinates(address)
        if latitude is None or longitude is None:
            return render_template("result.html", prediction="Error", probability="Invalid Address")

        # Extract features from time and date
        hour = int(time.split(":")[0])
        month = int(date.split("-")[1])

        # Calculate Distance_From_Center
        distance_from_center = calculate_distance(latitude, longitude)

        # Determine Cluster using K-Means
        cluster = kmeans.predict(np.array([[latitude, longitude]]))[0]

        # Prepare features for prediction  
        features = pd.DataFrame(
            [[distance_from_center, hour, month, cluster]],
            columns=['Distance_From_Center', 'Hour', 'Month', 'Cluster']
            )

        # Make prediction
        prediction = gb_model.predict(features)[0]
        probability = gb_model.predict_proba(features)[0]

        # Map prediction to severity labels
        severity_map = {0: "Property Damage Only", 1: "Fatal/Injury"}
        result = {
            "predicted_severity": severity_map[prediction],
            "confidence_scores": {
                "Property Damage Only": probability[0],
                "Fatal/Injury": probability[1]
            }
        }

        return render_template(
            "result.html",
            prediction=result["predicted_severity"],
            probability=f"{result["confidence_scores"]["Fatal/Injury"]:.2f}"
        )
    
    except Exception as e:
        print("Error in prediction:", str(e))
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
