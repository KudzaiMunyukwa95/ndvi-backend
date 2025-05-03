import os
import json
import ee
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load and fix credentials
service_account_info = json.loads(os.environ['GEE_CREDENTIALS'])
service_account_info['private_key'] = service_account_info['private_key'].replace('\\n', '\n')

# Authenticate
credentials = ee.ServiceAccountCredentials(
    email=service_account_info['client_email'],
    key_data=json.dumps(service_account_info)
)
ee.Initialize(credentials)

@app.route('/api/gee_ndvi', methods=['POST'])
def get_ndvi():
    try:
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

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
