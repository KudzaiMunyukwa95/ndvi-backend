from flask import Flask, request, jsonify
import ee

app = Flask(__name__)

# Initialize the Earth Engine API
ee.Initialize()

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