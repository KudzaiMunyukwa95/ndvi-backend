
from flask import Flask, request, jsonify
import ee
import os
import json

app = Flask(__name__)

# Load raw JSON string from Render env variable
key_json_str = os.environ.get('GEE_CREDENTIALS_JSON')

# Initialize Earth Engine with raw JSON string
credentials = ee.ServiceAccountCredentials(
    email=json.loads(key_json_str)["client_email"],
    key_data=key_json_str  # <-- Must be string
)

ee.Initialize(credentials)

@app.route('/api/gee_ndvi', methods=['POST'])
def get_ndvi():
    try:
        data = request.get_json()
        coordinates = data['coordinates']
        start_date = data['startDate']
        end_date = data['endDate']

        geometry = ee.Geometry.Polygon([coordinates])

        collection = ee.ImageCollection('COPERNICUS/S2_SR') \
            .filterDate(start_date, end_date) \
            .filterBounds(geometry) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))

        ndvi = collection.map(lambda img: img.normalizedDifference(['B8', 'B4']).rename('NDVI')).mean()

        mean_ndvi = ndvi.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            scale=10,
            maxPixels=1e9
        ).get('NDVI').getInfo()

        return jsonify({'success': True, 'ndvi': mean_ndvi})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)
