
from flask import Flask, request, jsonify
import ee
import os
import json

app = Flask(__name__)

# Load GEE credentials from environment
SERVICE_ACCOUNT = os.getenv('GEE_SERVICE_ACCOUNT')
PRIVATE_KEY = os.getenv('GEE_PRIVATE_KEY')

# Format private key for JSON
key_json = {
    "type": "service_account",
    "client_email": SERVICE_ACCOUNT,
    "private_key": PRIVATE_KEY,
    "token_uri": "https://oauth2.googleapis.com/token"
}

# Authenticate Earth Engine using service account
credentials = ee.ServiceAccountCredentials(SERVICE_ACCOUNT, key_data=key_json)
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
