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
            return jsonify({"success": False, "message": "Missing coordinates or date range"}), 400

        polygon = ee.Geometry.Polygon(coords)

        # Load Sentinel-2 collection and apply filters
        collection = ee.ImageCollection("COPERNICUS/S2_HARMONIZED") \
            .filterBounds(polygon) \
            .filterDate(start, end) \
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20)) \
            .sort('system:time_start', False)  # latest first

        image = collection.first().clip(polygon)

        # NDVI calculation
        ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")

        # RGB visualization parameters
        rgb_vis = {
            "bands": ["B4", "B3", "B2"],
            "min": 0,
            "max": 3000
        }

        # NDVI visualization parameters
        ndvi_vis = {
            "min": 0,
            "max": 1,
            "palette": ["#d7191c", "#fdae61", "#ffffbf", "#a6d96a", "#1a9641"]
        }

        # Get tile URLs
        ndvi_map = ndvi.getMapId(ndvi_vis)
        rgb_map = image.getMapId(rgb_vis)

        # Get NDVI stats
        stats = ndvi.reduceRegion(
            reducer=ee.Reducer.minMax().combine(ee.Reducer.mean(), "", True),
            geometry=polygon,
            scale=10,
            maxPixels=1e9
        )

        ndvi_min = stats.get("NDVI_min").getInfo()
        ndvi_max = stats.get("NDVI_max").getInfo()
        ndvi_mean = stats.get("NDVI_mean").getInfo()

        return jsonify({
            "success": True,
            "ndvi_tile_url": f"https://earthengine.googleapis.com/map/{ndvi_map['mapid']}/{{z}}/{{x}}/{{y}}?token={ndvi_map['token']}",
            "rgb_tile_url": f"https://earthengine.googleapis.com/map/{rgb_map['mapid']}/{{z}}/{{x}}/{{y}}?token={rgb_map['token']}",
            "ndvi_min": ndvi_min,
            "ndvi_max": ndvi_max,
            "ndvi_mean": ndvi_mean
        })

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

# FINAL DEPLOYMENT FOOTER FOR RENDER
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
