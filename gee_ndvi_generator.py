import os
import json
import ee
import traceback
from datetime import datetime
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
        temperature_data = data.get("temperature_data", [])
        gdd_data = data.get("gdd_data", [])
        gdd_stats = data.get("gdd_stats", {})
        temperature_summary = data.get("temperature_summary", {})
        base_temperature = data.get("base_temperature", 10)
        
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
        if irrigated == "Yes":
            rainfall_formatted = "Not applicable for irrigated fields"
        elif rainfall_data:
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
        
        # Format temperature data
        temp_formatted = "No data"
        if temperature_data:
            avg_min = sum(item["min"] for item in temperature_data) / len(temperature_data)
            avg_max = sum(item["max"] for item in temperature_data) / len(temperature_data)
            temp_formatted = f"Avg min: {avg_min:.1f}°C, Avg max: {avg_max:.1f}°C, Range: {min(item['min'] for item in temperature_data):.1f}°C to {max(item['max'] for item in temperature_data):.1f}°C"
        
        # Format GDD data
        gdd_formatted = "No data"
        if gdd_stats:
            gdd_formatted = f"Cumulative GDD: {gdd_stats.get('total_gdd', 'N/A')}, Avg daily GDD: {gdd_stats.get('avg_daily_gdd', 'N/A')}, Base temp: {base_temperature}°C"
        
        # Calculate NDVI change rate if we have enough data points
        ndvi_change_rates = []
        if len(ndvi_data) > 1:
            # Sort data by date
            sorted_ndvi = sorted(ndvi_data, key=lambda x: x['date'])
            
            # Calculate change rates
            for i in range(1, len(sorted_ndvi)):
                try:
                    date1 = datetime.strptime(sorted_ndvi[i-1]['date'], '%Y-%m-%d')
                    date2 = datetime.strptime(sorted_ndvi[i]['date'], '%Y-%m-%d')
                    days_diff = (date2 - date1).days
                    
                    if days_diff > 0:  # Avoid division by zero
                        ndvi_diff = sorted_ndvi[i]['ndvi'] - sorted_ndvi[i-1]['ndvi']
                        change_rate = ndvi_diff / days_diff
                        ndvi_change_rates.append({
                            'start_date': sorted_ndvi[i-1]['date'],
                            'end_date': sorted_ndvi[i]['date'],
                            'days': days_diff,
                            'change_rate': change_rate,
                            'total_change': ndvi_diff
                        })
                except Exception as e:
                    print(f"Error calculating NDVI change rate: {e}")

        # Format NDVI change rate data for the prompt
        ndvi_change_formatted = "No data"
        if ndvi_change_rates:
            # Find the periods with significant changes
            significant_changes = [r for r in ndvi_change_rates if abs(r['change_rate']) > 0.005]  # 0.005 per day is significant
            
            if significant_changes:
                # Sort by absolute change rate
                significant_changes.sort(key=lambda x: abs(x['change_rate']), reverse=True)
                
                # Format top 3 changes
                top_changes = significant_changes[:3]
                ndvi_change_formatted = ", ".join([
                    f"{c['start_date']} to {c['end_date']}: {c['change_rate']*100:.2f}% per day ({c['total_change']:.2f} over {c['days']} days)"
                    for c in top_changes
                ])
        
        # Analyze NDVI patterns to detect possible tillage/replanting scenarios
        tillage_replanting_detected = False
        tillage_info = "No tillage or replanting pattern detected"
        
        if len(ndvi_data) >= 3:
            sorted_ndvi = sorted(ndvi_data, key=lambda x: x['date'])
            # Look for high -> low -> high pattern
            max_drop = 0
            drop_start_idx = -1
            drop_end_idx = -1
            
            for i in range(1, len(sorted_ndvi)):
                current_drop = sorted_ndvi[i-1]['ndvi'] - sorted_ndvi[i]['ndvi']
                if current_drop > max_drop and current_drop > 0.15 and sorted_ndvi[i-1]['ndvi'] > 0.3:
                    max_drop = current_drop
                    drop_start_idx = i-1
                    drop_end_idx = i
            
            # If we found a significant drop, check if NDVI rises again afterward
            if drop_start_idx >= 0 and drop_end_idx < len(sorted_ndvi) - 1:
                # Look for subsequent rise
                subsequent_rise = False
                for i in range(drop_end_idx + 1, len(sorted_ndvi)):
                    if sorted_ndvi[i]['ndvi'] > sorted_ndvi[drop_end_idx]['ndvi'] + 0.1:
                        subsequent_rise = True
                        break
                
                if subsequent_rise:
                    tillage_replanting_detected = True
                    tillage_info = (f"Potential tillage/replanting detected: NDVI dropped from {sorted_ndvi[drop_start_idx]['ndvi']:.2f} "
                                   f"to {sorted_ndvi[drop_end_idx]['ndvi']:.2f} around {sorted_ndvi[drop_end_idx]['date']}, "
                                   f"then rose again afterward")
        
        # Construct prompt with enhanced information including tillage detection, NDVI change rates
        prompt = f"""You are an intelligent agronomic assistant embedded inside the Yieldera platform. Your task is to generate insightful crop development commentary based on NDVI trends, {'rainfall data, ' if irrigated == 'No' else ''}temperature patterns, GDD information, field location, and known crop properties.

🌾 Background
Each analysis request includes:
- Crop Type: {crop}
- Variety: {variety}
- Irrigation Status: {'Irrigated' if irrigated == 'Yes' else 'Rainfed'}
- Latitude and Longitude: {latitude}, {longitude}
- NDVI Time Series: {ndvi_formatted}
- NDVI Change Rate Analysis: {ndvi_change_formatted}
- Tillage Pattern Analysis: {tillage_info}
{'' if irrigated == 'Yes' else f'- Rainfall Time Series: {rainfall_formatted}'}
- Temperature Data: {temp_formatted}
- Growing Degree Days: {gdd_formatted}
- Analysis Date Range: {date_range}

🎓 Agronomic Intelligence to Assume:
- **Maize (e.g. SC727):**
  - Rainfed planting window: Nov--Jan (Zimbabwe)
  - Early NDVI rise expected ~2 weeks after planting
  - Peak NDVI: ~60--80 days after planting
  - Senescence onset: NDVI may drop after ~100--120 days
  - GDD for emergence: 90-120, silking: 700-800, maturity: 1350-1450
- **Soybeans:**
  - Planting window: Late Nov--Dec
  - Shorter lifecycle (~110 days)
  - GDD for emergence: 70-90, flowering: 550-600, maturity: 1100-1200
- **Wheat:**
  - Winter wheat planted May--Jun (irrigated)
  - Spring wheat typically Nov--Dec (rainfed)
  - GDD for emergence: 80-100, heading: 400-500, maturity: 800-900
- If crop/variety is unknown or labeled 'testing', use general NDVI pattern logic and suggest entering known values for deeper insights.

📌 Special Case Guidance:
- If NDVI starts high but drops to bare soil levels (<0.2) and then rises again, assume a new planting occurred within the date range.
- Do not conclude that the field was already planted if a tillage-then-emergence pattern is detected.
- Pay attention to NDVI change rate/slope to distinguish between rapid emergence, flat periods, or potential stress.
- For irrigated fields, focus on NDVI patterns, temperature, and GDD rather than rainfall.
- Always attempt to infer a planting date based on all available data.

🧠 Your Analysis Must Include:
1. NDVI Pattern Interpretation (flat, rising, declining, or mixed patterns including tillage events)
2. Temperature & GDD Implications (if data available)
3. {'Rainfall Response (rainfed crop triggers, dry periods)' if irrigated == 'No' else 'Irrigation & Crop Management Implications'}
4. Crop Status Summary (bare soil, emergence, stress, maturity, potential replanting)
5. Planting Date Inference - estimate based on NDVI patterns, {'rainfall, ' if irrigated == 'No' else ''}temperature and GDD
6. Confidence Rating (High, Medium, Low)

🧭 Examples of Language to Use:
- "NDVI remained flat at ~0.18, indicating bare soil or no active vegetation."
- "The NDVI drop from 0.6 to 0.2 followed by a rise suggests tillage and replanting occurred in mid-December."
- "Rising temperatures and accumulated GDD of 120 indicate favorable conditions for emergence."
- "NDVI shows a rapid increase rate of 0.05 per day after Jan 15, indicating vigorous early growth."
- "NDVI increase after Dec 3 suggests planting occurred in late Nov."
{'' if irrigated == 'Yes' else '- "Rainfall was insufficient to support rainfed planting."'}
- "NDVI decline suggests senescence or water stress."
- "Crop variety is unrecognized -- general vegetation analysis applied."

🧵 Output Format:
Respond in 2--4 sentences as a trained agronomist advising a field agent or insurer. Always include your estimated planting date if you can determine one. Avoid referencing GPT, AI, or farmer-declared dates."""

        # Call OpenAI API
        try:
            print(f"Sending request to generate insight for field: {field_name}")
            
            response = openai_client.chat.completions.create(
                model="gpt-4o",  # You can adjust the model as needed
                messages=[
                    {"role": "system", "content": "You are Yieldera's agricultural advisor. Focus on planting date estimation. DO NOT mention AI, GPT, or any third-party tools."},
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
                "confidence_level": confidence_level,
                "tillage_detected": tillage_replanting_detected
            })
            
        except Exception as e:
            print(f"Insight generation error: {str(e)}")
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
        
        # Validate polygon geometry
        if not isinstance(coords, list) or len(coords) == 0:
            return jsonify({"success": False, "error": "Invalid coordinates format"}), 400
            
        # Ensure we have a valid polygon (at least 3 points)
        if len(coords[0]) < 3:
            return jsonify({"success": False, "error": "Invalid polygon: must have at least 3 points"}), 400
        
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
        
        # Validate polygon geometry
        if not isinstance(coords, list) or len(coords) == 0:
            return jsonify({"success": False, "error": "Invalid coordinates format"}), 400
            
        # Ensure we have a valid polygon (at least 3 points)
        if len(coords[0]) < 3:
            return jsonify({"success": False, "error": "Invalid polygon: must have at least 3 points"}), 400
        
        # Log incoming request
        print(f"Processing time series request: start={start}, end={end}, coords length={len(coords[0])}")
        
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
        
        # Verify we have sufficient data points
        if len(ndvi_time_series) == 0:
            return jsonify({
                "success": False, 
                "error": "No valid NDVI readings could be calculated for this field",
                "empty_time_series": True
            }), 404
        
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
