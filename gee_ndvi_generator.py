import os
import json
import ee
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    return "NDVI & RGB backend is live!"

@app.route("/api/gee_ndvi", methods=["POST"])
def generate_ndvi():
    try:
        # Initialize GEE with service account if not already initialized
        if not ee.data._initialized:
            try:
                # Load service account info from environment variable
                service_account_info = json.loads(os.environ["GEE_CREDENTIALS"])
                
                # Initialize Earth Engine using service account credentials
                credentials = ee.ServiceAccountCredentials(
                    email=service_account_info["client_email"],
                    key_data=json.dumps(service_account_info)
                )
                ee.Initialize(credentials)
            except Exception as e:
                return jsonify({
                    "success": False, 
                    "error": f"GEE initialization error: {str(e)}",
                    "details": traceback.format_exc()
                }), 500
        
        # Parse request data
        data = request.get_json()
        coords = data.get("coordinates")
        start = data.get("startDate")
        end = data.get("endDate")
        
        # Validate inputs
        if not coords or not start or not end:
            return jsonify({"success": False, "error": "Missing input fields"}), 400
        
        # Log incoming request
        print(f"Processing request: start={start}, end={end}, coords length={len(coords)}")
        
        # Create Earth Engine geometry
        polygon = ee.Geometry.Polygon(coords)

        # ENHANCED: Progressive cloud filtering approach
        # Instead of using fixed thresholds, we'll try increasingly permissive filters
        # until we find enough imagery
        cloud_thresholds = [10, 20, 30, 50, 80]  # Progressive thresholds to try
        collection = None
        
        for threshold in cloud_thresholds:
            # Get Sentinel-2 collection with cloud filtering
            collection = (
                ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
                .filterBounds(polygon)
                .filterDate(start, end)
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", threshold))
                .sort("CLOUDY_PIXEL_PERCENTAGE")  # Sort by cloud percentage
            )
            
            # Limit collection size based on cloud threshold - fewer images for higher cloud coverage
            limit_size = 10 if threshold <= 20 else (5 if threshold <= 50 else 3)
            collection = collection.limit(limit_size)
            
            # Check if we found any images
            collection_size = collection.size().getInfo()
            if collection_size > 0:
                print(f"Found {collection_size} images with cloud threshold {threshold}%")
                break
        
        # Handle empty collection after all attempts
        collection_size = collection.size().getInfo() if collection else 0
        if collection_size == 0:
            return jsonify({
                "success": False, 
                "error": "No Sentinel-2 imagery found for the specified date range and location",
                "empty_collection": True
            }), 404
        
        # Get median image and clip to polygon
        image = collection.median().clip(polygon)
        
        # ENHANCED: Save the cloud percentage of the best image for reporting
        first_image = collection.first()
        cloud_percentage = first_image.get("CLOUDY_PIXEL_PERCENTAGE").getInfo()
        
        # Calculate NDVI
        ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
        rgb = image.select(["B4", "B3", "B2"])
        
        # ENHANCED: Additional quality filters for better results
        # Apply a mask to exclude very cloudy or low-quality pixels
        quality_bands = ["QA60"]
        if image.bandNames().contains("QA60").getInfo():
            # Extract quality band (QA60 contains cloud mask information)
            qa_band = image.select("QA60")
            # Apply basic cloud mask (bits 10 and 11 are opaque and cirrus clouds)
            cloud_bitmask = 1 << 10 | 1 << 11
            mask = qa_band.bitwiseAnd(cloud_bitmask).eq(0)
            
            # Apply the quality mask to the NDVI and RGB layers
            ndvi = ndvi.updateMask(mask)
            rgb = rgb.updateMask(mask)
        
        # Visualization settings
        ndvi_vis = ndvi.visualize(min=0, max=1, palette=["red", "yellow", "green"])
        rgb_vis = rgb.visualize(min=0, max=3000)
        
        # Get map IDs for tile URLs with timeout handling
        try:
            map_id_ndvi = ee.data.getMapId({"image": ndvi_vis})
            map_id_rgb = ee.data.getMapId({"image": rgb_vis})
        except Exception as e:
            print(f"Error getting map IDs: {e}")
            # Still return statistics even if visualization fails
            ndvi_stats = ndvi.reduceRegion(
                reducer=ee.Reducer.mean().combine(
                    ee.Reducer.minMax(), "", True
                ),
                geometry=polygon,
                scale=10,
                maxPixels=1e9
            ).getInfo()
            
            image_date = first_image.date().format("YYYY-MM-dd").getInfo()
            
            return jsonify({
                "success": True,
                "mean_ndvi": ndvi_stats.get("NDVI_mean"),
                "min_ndvi": ndvi_stats.get("NDVI_min"),
                "max_ndvi": ndvi_stats.get("NDVI_max"),
                "image_date": image_date,
                "collection_size": collection_size,
                "cloudy_pixel_percentage": cloud_percentage,
                "visualization_error": str(e)
            })
        
        # Calculate NDVI statistics
        ndvi_stats = ndvi.reduceRegion(
            reducer=ee.Reducer.mean().combine(
                ee.Reducer.minMax(), "", True
            ),
            geometry=polygon,
            scale=10,
            maxPixels=1e9
        ).getInfo()
        
        # Get image acquisition date from first image in collection
        image_date = first_image.date().format("YYYY-MM-dd").getInfo()
        
        # Return response with all data including cloud percentage
        response = {
            "success": True,
            "ndvi_tile_url": map_id_ndvi["tile_fetcher"].url_format,
            "rgb_tile_url": map_id_rgb["tile_fetcher"].url_format,
            "mean_ndvi": ndvi_stats.get("NDVI_mean"),
            "min_ndvi": ndvi_stats.get("NDVI_min"),
            "max_ndvi": ndvi_stats.get("NDVI_max"),
            "image_date": image_date,
            "collection_size": collection_size,
            "cloudy_pixel_percentage": cloud_percentage
        }
        
        print(f"Successfully processed NDVI request. Mean NDVI: {ndvi_stats.get('NDVI_mean')}, Cloud cover: {cloud_percentage}%")
        return jsonify(response)

    except Exception as e:
        error_message = str(e)
        stack_trace = traceback.format_exc()
        print(f"Error in GEE NDVI processing: {error_message}")
        print(f"Stack trace: {stack_trace}")
        
        return jsonify({
            "success": False, 
            "error": error_message,
            "stack_trace": stack_trace
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
