from flask import Flask, request, jsonify
import ee
import os
import json

app = Flask(__name__)

# Load credentials from environment variables
service_account_email = os.environ.get("GEE_SERVICE_ACCOUNT")
private_key = os.environ.get("GEE_PRIVATE_KEY")

# Build the credentials JSON as a string
credentials_json_str = json.dumps({
    "type": "service_account",
    "client_email": service_account_email,
    "private_key": private_key,
    "token_uri": "https://oauth2.googleapis.com/token"
})

# Authenticate with Earth Engine
try:
    credentials = ee.ServiceAccountCredentials(email=service_account_email, key_data=credentials_json_str)
    ee.Initialize(credentials)
except Exception as e:
    print("EE Auth failed:", str(e))

@app.route('/api/gee_ndvi', methods=['POST'])
def get_ndvi():
    try:
        data = request.get_json()
        coordinates = data['coordinates']
        start_date = data['startDate']
        end_date = data['endDate']

        geometry = ee.Geometry.Polygon(coordinates)

        # Get NDVI from Sentinel-2 (more common now than Landsat 8)
        collection = ee.ImageCollection("COPERNICUS/S2_SR") \
            .filterDate(start_date, end_date) \
            .filterBounds(geometry) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))

        ndvi = collection.map(lambda image: image.normalizedDifference(['B8', 'B4']).rename('NDVI')).mean()

        mean_ndvi = ndvi.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            scale=10
        ).get('NDVI').getInfo()

        return jsonify({"success": True, "mean_ndvi": mean_ndvi})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)), debug=True)
