from flask import Flask, request, jsonify
import os
import json
import ee

app = Flask(__name__)

# Get service account info from environment variables
service_account_info = {
    "type": "service_account",
    "project_id": "ee-kaymunyukwa",
    "private_key_id": os.environ.get("GEE_PRIVATE_KEY_ID"),
    "private_key": os.environ.get("GEE_PRIVATE_KEY").replace("\\n", "\n"),
    "client_email": os.environ.get("GEE_SERVICE_ACCOUNT"),
    "client_id": os.environ.get("GEE_CLIENT_ID"),
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": f"https://www.googleapis.com/robot/v1/metadata/x509/{os.environ.get('GEE_SERVICE_ACCOUNT').replace('@', '%40')}"
}

# Convert to JSON string
key_data_json = json.dumps(service_account_info)

# Authenticate with GEE using service account
credentials = ee.ServiceAccountCredentials(
    email=service_account_info["client_email"],
    key_data=key_data_json
)

ee.Initialize(credentials)

@app.route('/api/gee_ndvi', methods=['POST'])
def get_ndvi():
    data = request.json
    coordinates = data['coordinates']
    start_date = data['startDate']
    end_date = data['endDate']

    geometry = ee.Geometry.Polygon(coordinates)

    dataset = ee.ImageCollection('LANDSAT/LC08/C01/T1')         .filterBounds(geometry)         .filterDate(start_date, end_date)

    ndvi = dataset.map(lambda image: image.normalizedDifference(['B5', 'B4']).rename('NDVI')).mean()

    ndvi_value = ndvi.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geometry,
        scale=30
    ).get('NDVI').getInfo()

    return jsonify({'success': True, 'ndvi': ndvi_value})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(debug=True, host='0.0.0.0', port=port)
