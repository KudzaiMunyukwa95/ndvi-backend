
from flask import Flask, request, jsonify
import ee
import os

app = Flask(__name__)

# Load GEE credentials from environment
SERVICE_ACCOUNT = os.getenv('GEE_SERVICE_ACCOUNT')
PRIVATE_KEY = os.getenv('GEE_PRIVATE_KEY')

# Full credentials JSON for Earth Engine
service_account_info = {
    "type": "service_account",
    "project_id": "ee-kaymunyukwa",
    "private_key_id": "dummy_key_id",
    "private_key": PRIVATE_KEY,
    "client_email": SERVICE_ACCOUNT,
    "client_id": "dummy_client_id",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": f"https://www.googleapis.com/robot/v1/metadata/x509/{SERVICE_ACCOUNT}"
}

# Authenticate Earth Engine using service account
credentials = ee.ServiceAccountCredentials(SERVICE_ACCOUNT, service_account_info)
ee.Initialize(credentials)

@app.route('/api/gee_ndvi', methods=['POST'])
def get_ndvi():
    data = request.json
    coordinates = data['coordinates']
    start_date = data['startDate']
    end_date = data['endDate']

    # Convert coordinates to an ee.Geometry
    geometry = ee.Geometry.Polygon(coordinates)

    # Calculate NDVI using GEE
    dataset = ee.ImageCollection('LANDSAT/LC08/C01/T1') \
        .filterBounds(geometry) \
        .filterDate(start_date, end_date)

    ndvi = dataset.map(lambda image: image.normalizedDifference(['B5', 'B4']).rename('NDVI')).mean()

    # Get NDVI value as a single pixel value (average)
    ndvi_value = ndvi.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geometry,
        scale=30
    ).get('NDVI').getInfo()

    return jsonify({'success': True, 'ndvi': ndvi_value})

if __name__ == '__main__':
    app.run(debug=True)
