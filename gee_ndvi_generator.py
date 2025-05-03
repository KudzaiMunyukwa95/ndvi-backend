from flask import Flask, request, jsonify
import ee
import os
import json

app = Flask(__name__)

# Load service account credentials from environment variables
SERVICE_ACCOUNT = os.getenv('GEE_SERVICE_ACCOUNT')
PRIVATE_KEY = os.getenv('GEE_PRIVATE_KEY').replace('\\n', '\n')

# Build the full credentials dictionary manually
key_data = {
    "type": "service_account",
    "client_email": SERVICE_ACCOUNT,
    "private_key": PRIVATE_KEY,
    "token_uri": "https://oauth2.googleapis.com/token"
}

# Authenticate with Earth Engine
credentials = ee.ServiceAccountCredentials(SERVICE_ACCOUNT, key_data)
ee.Initialize(credentials)

@app.route('/api/gee_ndvi', methods=['POST'])
def get_ndvi():
    data = request.json
    coordinates = data['coordinates']
    start_date = data['startDate']
    end_date = data['endDate']

    geometry = ee.Geometry.Polygon(coordinates)

    dataset = ee.ImageCollection('LANDSAT/LC08/C01/T1') \
        .filterBounds(geometry) \
        .filterDate(start_date, end_date)

    ndvi = dataset.map(lambda image: image.normalizedDifference(['B5', 'B4']).rename('NDVI')).mean()

    ndvi_value = ndvi.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geometry,
        scale=30
    ).get('NDVI').getInfo()

    return jsonify({'success': True, 'ndvi': ndvi_value})

if __name__ == '__main__':
    app.run(debug=True)
