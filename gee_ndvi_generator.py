from flask import Flask, request, jsonify
import ee
import os
import json

app = Flask(__name__)

# Load the GEE credentials from environment variable
service_account_info = json.loads(os.environ['GEE_CREDENTIALS'])

# Initialize Earth Engine with service account
credentials = ee.ServiceAccountCredentials(
    email=service_account_info["client_email"],
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

        # Construct valid GeoJSON Polygon
        geometry = ee.Geometry({
            "type": "Polygon",
            "coordinates": coordinates
        })

        # Fetch LANDSAT-8 and calculate NDVI
        collection = ee.ImageCollection('LANDSAT/LC08/C01/T1') \
            .filterBounds(geometry) \
            .filterDate(start_date, end_date)

        ndvi = collection.map(
            lambda image: image.normalizedDifference(['B5', 'B4']).rename('NDVI')
        ).mean()

        mean_ndvi = ndvi.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            scale=30
        ).get('NDVI').getInfo()

        return jsonify({"success": True, "ndvi": mean_ndvi})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
