from flask import Flask, request, jsonify
from flask_cors import CORS
import ee
import os
import json

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Authenticate with GEE
service_account_info = json.loads(os.environ['GEE_CREDENTIALS'])
credentials = ee.ServiceAccountCredentials(email=service_account_info["client_email"], key_data=service_account_info)
ee.Initialize(credentials)

@app.route("/api/gee_ndvi", methods=["POST"])
def get_ndvi():
    try:
        data = request.get_json()
        coords = data["coordinates"]
        start = data["startDate"]
        end = data["endDate"]

        geometry = ee.Geometry.Polygon(coords)
        collection = ee.ImageCollection("COPERNICUS/S2_SR") \
            .filterBounds(geometry) \
            .filterDate(start, end) \
            .median()

        ndvi = collection.normalizedDifference(["B8", "B4"]).rename("NDVI")
        mean_ndvi = ndvi.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            scale=10,
            maxPixels=1e9
        ).get("NDVI").getInfo()

        return jsonify({"success": True, "ndvi": mean_ndvi})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
