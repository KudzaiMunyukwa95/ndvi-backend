import os
import json
import ee
import traceback
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Example emergence windows (this part of your file may be longer; keep as is)
EMERGENCE_WINDOWS = {
    "Maize": (6, 10),
    "Soyabeans": (8, 12),
    "Sorghum": (6, 10),
    "Cotton": (5, 9),
    "Groundnuts": (6, 10),
    "Barley": (4, 8),
    "Wheat": (7, 11),
    "Millet": (4, 8),
    "Tobacco": (7, 11)  # For nursery emergence
}

# Constants for emergence detection
EMERGENCE_THRESHOLD = 0.2
DEFAULT_EMERGENCE_WINDOW = (5, 10)  # Default for unknown crops
SIGNIFICANT_RAINFALL = 10  # mm, threshold for significant rainfall

@app.route("/")
def index():
    return "NDVI & RGB backend is live!"

# (Other routes and functions above generate_ndvi_timeseries remain unchanged)

@app.route("/api/gee_ndvi_timeseries", methods=["POST"])
def generate_ndvi_timeseries():
    try:
        # Initialize GEE with service account if not already initialized
        if not ee.data._initialized:
            try:
                service_account_info = json.loads(os.environ.get("GEE_CREDENTIALS", "{}"))
                credentials = ee.ServiceAccountCredentials(
                    email=service_account_info.get("client_email"),
                    key_data=json.dumps(service_account_info)
                )
                ee.Initialize(credentials)
            except Exception as e:
                return jsonify({
                    "success": False,
                    "error": f"GEE initialization error: {str(e)}",
                    "details": traceback.format_exc()
                }), 500

        # Parse request JSON
        data = request.get_json()
        coords = data.get("coordinates")
        start = data.get("startDate")
        end = data.get("endDate")

        # Validate inputs
        if not coords or not start or not end:
            return jsonify({"success": False, "error": "Missing input fields"}), 400
        if not isinstance(coords, list) or len(coords) == 0:
            return jsonify({"success": False, "error": "Invalid coordinates format"}), 400
        if len(coords[0]) < 3:
            return jsonify({"success": False, "error": "Invalid polygon: must have at least 3 points"}), 400

        print(f"Processing time series request: start={start}, end={end}, coords length={len(coords[0])}")
        polygon = ee.Geometry.Polygon(coords)

        # Progressive cloud filtering
        cloud_thresholds = [10, 20, 30, 50, 80]
        collection = None
        for threshold in cloud_thresholds:
            coll = (
                ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
                .filterBounds(polygon)
                .filterDate(start, end)
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", threshold))
            )
            size = coll.size().getInfo()
            if size > 0:
                print(f"Found {size} images with cloud threshold {threshold}%")
                collection = coll
                break

        # If still empty, return 404
        collection_size = collection.size().getInfo() if collection else 0
        if collection_size == 0:
            return jsonify({
                "success": False,
                "error": "No Sentinel-2 imagery found for the specified date range and location",
                "empty_collection": True
            }), 404

        # ========== REFRACTORED PART: build server-side NDVI‐time series ==========
        def per_image_to_feature(img):
            """EE function: for each image, compute mean NDVI, read date & cloud%."""
            # Compute NDVI band
            ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
            # Calculate mean NDVI over the polygon
            mean_dict = ndvi.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=polygon,
                scale=10,
                maxPixels=1e9
            )
            mean_ndvi = mean_dict.get("NDVI")
            # Get image date as formatted string
            date_str = img.date().format("YYYY-MM-dd")
            # Get cloud percentage
            cloud_pct = img.get("CLOUDY_PIXEL_PERCENTAGE")
            # Return a Feature with date, ndvi, cloud as properties
            return ee.Feature(
                None,
                {
                    "date": date_str,
                    "ndvi": mean_ndvi,
                    "cloud_percentage": cloud_pct
                }
            )

        # Map over entire collection (server‐side)
        fc = collection.map(per_image_to_feature)

        # Now fetch everything in one getInfo() call
        features = fc.getInfo().get("features", [])

        # Build a simple list of dicts: { "date": "...", "ndvi": x.xx, "cloud_percentage": yy }
        ndvi_time_series = []
        for f in features:
            props = f.get("properties", {})
            # Make sure NDVI really exists (could be null if masked out)
            ndvi_val = props.get("ndvi")
            if ndvi_val is not None:
                ndvi_time_series.append({
                    "date": props.get("date"),
                    "ndvi": ndvi_val,
                    "cloud_percentage": props.get("cloud_percentage")
                })

        # If we ended up with no valid NDVI points, return 404
        if len(ndvi_time_series) == 0:
            return jsonify({
                "success": False,
                "error": "No valid NDVI readings could be calculated for this field",
                "empty_time_series": True
            }), 404

        # Sort by date (just in case EE didn’t preserve chronological order)
        ndvi_time_series.sort(key=lambda x: x["date"])

        # Return response
        response = {
            "success": True,
            "time_series": ndvi_time_series,
            "collection_size": collection_size
        }
        print(f"Successfully processed NDVI time series request. {len(ndvi_time_series)} data points returned.")
        return jsonify(response)

    except Exception as e:
        error_message = str(e)
        stack_trace = traceback.format_exc()
        print(f"Error in GEE NDVI time series processing: {error_message}")
        print(f"Stack trace: {stack_trace}")
        return jsonify({
            "success": False,
            "error": error_message,
            "stack_trace": stack_trace
        }), 500

# (Other routes and functions below, unchanged)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
