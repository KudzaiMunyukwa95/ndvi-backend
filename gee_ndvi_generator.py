from flask import Flask, request, jsonify
import ee
import json
import os

app = Flask(__name__)

# Load GEE credentials from Render environment variable
service_account_info = json.loads(os.environ['GEE_CREDENTIALS'])

# Authenticate and initialize Earth Engine
credentials = ee.ServiceAccountCredentials(
    email=service_account_info['client_email'],
    key_data=json.dumps(service_account_info)
)
ee.Initialize(credentials)

@app.route('/api/gee_ndvi_rgb', methods=['POST'])
def get_ndvi_and_rgb():
    try:
        data = request.get_json()
        coordinates = data.get('coordinates')
        start_date = data.get('startDate')
        end_date = data.get('endDate')

        if not coordinates or not start_date or not end_date:
            return jsonify({'success': False, 'error': 'Missing parameters'}), 400

        geometry = ee.Geometry.Polygon(coordinates)

        collection = ee.ImageCollection('COPERNICUS/S2_SR') \
            .filterBounds(geometry) \
            .filterDate(start_date, end_date) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))

        # Calculate mean NDVI
        ndvi_image = collection.map(
            lambda img: img.normalizedDifference(['B8', 'B4']).rename('NDVI')
        ).mean()

        # Create NDVI visualization parameters
        ndvi_viz_params = {
            'min': 0.0,
            'max': 1.0,
            'palette': ['gray', 'red', 'orange', 'yellow', 'green']
        }

        # Create RGB visualization parameters
        rgb_image = collection.median().select(['B4', 'B3', 'B2'])  # Red, Green, Blue
        rgb_viz_params = {
            'min': 0,
            'max': 3000,
            'bands': ['B4', 'B3', 'B2']
        }

        # Generate map IDs and tokens
        map_id_ndvi = ee.data.getMapId({'image': ndvi_image.visualize(**ndvi_viz_params)})
        map_id_rgb = ee.data.getMapId({'image': rgb_image.visualize(**rgb_viz_params)})

        return jsonify({
            'success': True,
            'ndvi_tile_url': map_id_ndvi['tile_fetcher'].url_format,
            'rgb_tile_url': map_id_rgb['tile_fetcher'].url_format
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
