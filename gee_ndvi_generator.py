
from flask import Flask, request, jsonify
import ee
import json
import os

app = Flask(__name__)

# Load credentials from environment variable
service_account_info = json.loads(os.environ.get('GEE_CREDENTIALS_JSON'))
credentials = ee.ServiceAccountCredentials(email=service_account_info["client_email"], key_data=json.dumps(service_account_info))
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
