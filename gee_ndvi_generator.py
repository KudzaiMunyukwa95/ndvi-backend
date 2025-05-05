import os
import json
import ee
from flask import Flask, request, jsonify
from flask_cors import CORS

# Load service account info from environment variable
service_account_info = json.loads(os.environ["GEE_CREDENTIALS"])

# Initialize Earth Engine using service account credentials
credentials = ee.ServiceAccountCredentials(
    email=service_account_info["client_email"],
    key_data=json.dumps(service_account_info)
)
ee.Initialize(credentials)

app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    return "NDVI & RGB backend is live!"

@app.route("/api/gee_ndvi", methods=["POST"])
def generate_ndvi():
    try:
        data = request.get_json()
        coords = data.get("coordinates")
        start = data.get("startDate")
        end = data.get("endDate")

        if not coords or not start or not end:
            return jsonify({"success": False, "error": "Missing input fields"}), 400

        polygon = ee.Geometry.Polygon(coords)

        collection = (
            ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
            .filterBounds(polygon)
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
            .sort('system:time_start', False)  # newest first
        )

        image = collection.first().clip(polygon)

        # Extract actual image date
        image_date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd')

        ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
        rgb = image.select(["B4", "B3", "B2"])

        # Visualization settings
        ndvi_vis = ndvi.visualize(min=0, max=1, palette=["red", "yellow", "green"])
        rgb_vis = rgb.visualize(min=0, max=3000)

        map_id_ndvi = ee.data.getMapId({"image": ndvi_vis})
        map_id_rgb = ee.data.getMapId({"image": rgb_vis})

        return jsonify({
            "success": True,
            "ndvi_tile_url": map_id_ndvi["tile_fetcher"].url_format,
            "rgb_tile_url": map_id_rgb["tile_fetcher"].url_format,
            "capture_date": image_date.getInfo()
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
