import os
import json
import ee
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

@app.route("/")
def index():
    return "NDVI & RGB backend is live!"

@app.route("/api/agronomic_insight", methods=["POST"])
def generate_agronomic_report():
    try:
        # Parse request data
        data = request.get_json()
        
        # Extract required fields
        field_name = data.get("field_name", "Unknown field")
        crop = data.get("crop", "Unknown crop")
        variety = data.get("variety", "Unknown variety")
        irrigated = "Yes" if data.get("irrigated", False) else "No"
        latitude = data.get("latitude", "Unknown")
        longitude = data.get("longitude", "Unknown")
        date_range = data.get("date_range", "Unknown period")
        ndvi_data = data.get("ndvi_data", [])
        rainfall_data = data.get("rainfall_data", [])
        estimated_planting_date = data.get("estimated_planting_date")
        
        # Calculate average cloud cover if available
        avg_cloud_cover = None
        if ndvi_data and all("cloud_percentage" in item for item in ndvi_data):
            cloud_percentages = [item["cloud_percentage"] for item in ndvi_data if item["cloud_percentage"] is not None]
            if cloud_percentages:
                avg_cloud_cover = sum(cloud_percentages) / len(cloud_percentages)
        
        # Format NDVI data
        ndvi_formatted = ", ".join([f"{item['date']}: {item['ndvi']:.2f}" for item in ndvi_data[:10]]) if ndvi_data else "No data"
        if len(ndvi_data) > 10:
            ndvi_formatted += f" (+ {len(ndvi_data) - 10} more readings)"
        
        # Process rainfall data into weekly totals if available
        weekly_rainfall = {}
        if rainfall_data:
            for item in rainfall_data:
                date = item.get('date')
                if date:
                    # Simple weekly grouping by taking the first 7 chars of date (YYYY-MM)
                    # and the week number within the month (rough approximation)
                    week_key = date[:7] + "-W" + str((int(date[8:10]) - 1) // 7 + 1)
                    if week_key not in weekly_rainfall:
                        weekly_rainfall[week_key] = 0
                    weekly_rainfall[week_key] += item.get('rainfall', 0)
            
            rainfall_formatted = ", ".join([f"{week}: {total:.1f}mm" for week, total in weekly_rainfall.items()])
        else:
            rainfall_formatted = "No data"
        
        # Prepare prompt for insight generation - REMOVED FARMER-DECLARED PLANTING DATE
        prompt = f"""You are an expert agricultural advisor working for Yieldera. Analyze field-level satellite and weather data to provide a concise, professional agronomic insight.

Field name: {field_name}
Crop: {crop}
Variety: {variety}
Irrigated: {irrigated}
Coordinates: ({latitude}, {longitude})
Monitoring period: {date_range}

NDVI values over time: {ndvi_formatted}
Rainfall totals per week (mm): {rainfall_formatted}

Your job:
1. Comment on crop status based on NDVI trends.
2. If an estimated planting date is provided ({estimated_planting_date if estimated_planting_date else 'None'}), evaluate if it aligns with the NDVI signature.
3. If no clear planting date is detected, simply describe the vegetation patterns without referencing planting dates.
4. Mention if rainfall patterns support rainfed planting OR if the crop is likely irrigated.
5. If crop was already established before the start date, explain that.
6. If no crop activity is detected, say so.
7. End with a confidence rating (High, Medium, Low).

Respond in 2-4 sentences. Be clear, professional, and sound like an agronomist advising an insurance underwriter.
DO NOT mention "AI", "GPT", or any other third-party tools. The insight should appear to come directly from Yieldera's internal analysis system."""

        # Call OpenAI API
        try:
            print(f"Sending request to generate insight for field: {field_name}")
            
            response = openai_client.chat.completions.create(
                model="gpt-4o",  # You can adjust the model as needed
                messages=[
                    {"role": "system", "content": "You are Yieldera's agricultural advisor providing concise field assessments. DO NOT mention AI, GPT, or any third-party tools."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=300
            )
            
            insight = response.choices[0].message.content.strip()
            
            # Extract confidence level from the insight
            confidence_level = "medium"  # Default
            if "High confidence" in insight or "Confidence: High" in insight or "high confidence" in insight.lower():
                confidence_level = "high"
            elif "Low confidence" in insight or "Confidence: Low" in insight or "low confidence" in insight.lower():
                confidence_level = "low"
            
            # IMPROVED CONFIDENCE LOGIC: Boost confidence based on data quality
            if confidence_level != "high" and ndvi_data and len(ndvi_data) >= 10:
                # Check if NDVI pattern is consistent
                ndvi_values = [item["ndvi"] for item in ndvi_data]
                ndvi_std_dev = calculate_std_dev(ndvi_values)
                
                # If we have low cloud cover and consistent NDVI pattern, boost confidence
                if avg_cloud_cover is not None and avg_cloud_cover < 20 and ndvi_std_dev < 0.15:
                    confidence_level = "high"
                    print(f"Boosted confidence to high based on data quality: {len(ndvi_data)} observations, {avg_cloud_cover:.1f}% cloud cover")
            
            return jsonify({
                "success": True,
                "insight": insight,
                "confidence_level": confidence_level
            })
            
        except Exception as e:
            print(f"OpenAI API error: {str(e)}")
            return jsonify({
                "success": False,
                "error": f"Insight generation error: {str(e)}",
                "fallback_insight": "Unable to generate agronomic insight due to service error. Please try again later."
            }), 500
            
    except Exception as e:
        error_message = str(e)
        stack_trace = traceback.format_exc()
        print(f"Error generating agronomic report: {error_message}")
        print(f"Stack trace: {stack_trace}")
        
        return jsonify({
            "success": False,
            "error": error_message,
            "stack_trace": stack_trace
        }), 500

# Helper function to calculate standard deviation
def calculate_std_dev(values):
    if not values:
        return 0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance ** 0.5

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

@app.route("/api/gee_ndvi_timeseries", methods=["POST"])
def generate_ndvi_timeseries():
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
        print(f"Processing time series request: start={start}, end={end}, coords length={len(coords)}")
        
        # Create Earth Engine geometry
        polygon = ee.Geometry.Polygon(coords)

        # Progressive cloud filtering approach
        cloud_thresholds = [10, 20, 30, 50, 80]  # Progressive thresholds to try
        collection = None
        
        for threshold in cloud_thresholds:
            # Get Sentinel-2 collection with cloud filtering
            collection = (
                ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
                .filterBounds(polygon)
                .filterDate(start, end)
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", threshold))
            )
            
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
        
        # Get list of all images in the collection
        image_list = collection.toList(collection.size())
        
        # Process each image to get NDVI time series
        ndvi_time_series = []
        
        for i in range(collection_size):
            # Get the image
            image = ee.Image(image_list.get(i))
            
            # Clip to polygon
            clipped_image = image.clip(polygon)
            
            # Calculate NDVI
            ndvi = clipped_image.normalizedDifference(["B8", "B4"]).rename("NDVI")
            
            # Calculate mean NDVI for the polygon
            ndvi_stats = ndvi.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=polygon,
                scale=10,
                maxPixels=1e9
            ).getInfo()
            
            # Get image acquisition date
            image_date = image.date().format("YYYY-MM-dd").getInfo()
            
            # Get cloud percentage
            cloud_percentage = image.get("CLOUDY_PIXEL_PERCENTAGE").getInfo()
            
            # Only add valid NDVI readings
            ndvi_value = ndvi_stats.get("NDVI")
            if ndvi_value is not None:
                ndvi_time_series.append({
                    "date": image_date,
                    "ndvi": ndvi_value,
                    "cloud_percentage": cloud_percentage
                })
        
        # Sort time series by date
        ndvi_time_series.sort(key=lambda x: x["date"])
        
        # Return response with time series data
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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
