import os
import json
import ee
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

@app.route("/")
def index():
    return "NDVI & RGB backend is live!"

def detect_planting_pattern(ndvi_data):
    """
    Analyzes NDVI time series to detect the weed → tillage → crop emergence pattern.
    
    This function looks for:
    1. Initially high NDVI (weeds)
    2. Significant drop in NDVI (tillage)
    3. Consistent rise in NDVI afterward (crop emergence)
    
    Returns:
        dict: Contains detected pattern info, including:
            - pattern_detected: Boolean indicating if pattern was found
            - likely_planting_date: Estimated planting date if pattern was found
            - confidence: High, medium, or low
    """
    if not ndvi_data or len(ndvi_data) < 4:
        return {
            "pattern_detected": False,
            "likely_planting_date": None,
            "confidence": "low",
            "reason": "Insufficient data points"
        }
    
    # Sort data by date
    sorted_data = sorted(ndvi_data, key=lambda x: x["date"])
    
    # Find significant drops in NDVI (potential tillage events)
    potential_tillage = []
    
    for i in range(1, len(sorted_data)):
        current = sorted_data[i]
        previous = sorted_data[i-1]
        
        # Calculate the NDVI drop
        ndvi_drop = previous["ndvi"] - current["ndvi"]
        
        # Consider it a potential tillage if:
        # 1. Previous NDVI was relatively high (weeds)
        # 2. Current NDVI is low (bare soil)
        # 3. The drop is significant
        if (previous["ndvi"] > 0.3 and 
            current["ndvi"] < 0.2 and 
            ndvi_drop > 0.15):
            
            potential_tillage.append({
                "index": i,
                "date": current["date"],
                "ndvi_before": previous["ndvi"],
                "ndvi_after": current["ndvi"],
                "drop": ndvi_drop
            })
    
    # If no potential tillage events found, return negative result
    if not potential_tillage:
        return {
            "pattern_detected": False,
            "likely_planting_date": None,
            "confidence": "low",
            "reason": "No significant NDVI drops detected"
        }
    
    # For each potential tillage event, check if NDVI rises consistently afterward
    valid_patterns = []
    
    for event in potential_tillage:
        tillage_index = event["index"]
        
        # Need at least 2 points after tillage to confirm rise
        if tillage_index >= len(sorted_data) - 2:
            continue
        
        # Check if NDVI rises consistently after this tillage event
        consistent_rise = True
        rise_points = []
        
        for j in range(tillage_index, min(tillage_index + 4, len(sorted_data) - 1)):
            current = sorted_data[j]
            next_point = sorted_data[j+1]
            
            # If NDVI drops again or stays very low, it's not a consistent rise
            if next_point["ndvi"] <= current["ndvi"] or next_point["ndvi"] < 0.15:
                consistent_rise = False
                break
            
            rise_points.append({
                "date": next_point["date"],
                "ndvi": next_point["ndvi"]
            })
        
        # If we have a consistent rise and at least 2 rising points, consider this a valid pattern
        if consistent_rise and len(rise_points) >= 2:
            # Calculate rise rate to differentiate between natural vegetation recovery and crop emergence
            first_rise = rise_points[0]["ndvi"]
            last_rise = rise_points[-1]["ndvi"]
            days_between = (datetime.strptime(rise_points[-1]["date"], "%Y-%m-%d") - 
                           datetime.strptime(rise_points[0]["date"], "%Y-%m-%d")).days
            
            if days_between <= 0:  # Avoid division by zero
                days_between = 1
                
            rise_rate = (last_rise - first_rise) / days_between
            
            # Crop emergence typically shows a faster, more consistent rise than natural recovery
            if rise_rate > 0.01:  # Arbitrary threshold, may need adjustment
                valid_patterns.append({
                    "tillage_date": event["date"],
                    "rise_points": rise_points,
                    "rise_rate": rise_rate
                })
    
    # If no valid patterns found, return negative result
    if not valid_patterns:
        return {
            "pattern_detected": False,
            "likely_planting_date": None,
            "confidence": "low",
            "reason": "No consistent NDVI rise after drops"
        }
    
    # Sort valid patterns by rise rate (higher rise rate suggests crop emergence vs natural recovery)
    valid_patterns.sort(key=lambda x: x["rise_rate"], reverse=True)
    best_pattern = valid_patterns[0]
    
    # Estimate planting date (typically between tillage and first rise point)
    tillage_date = datetime.strptime(best_pattern["tillage_date"], "%Y-%m-%d")
    first_rise_date = datetime.strptime(best_pattern["rise_points"][0]["date"], "%Y-%m-%d")
    
    # Planting typically happens right after tillage, or a short time before first detected rise
    days_diff = (first_rise_date - tillage_date).days
    
    if days_diff <= 7:
        # If time between tillage and first rise is short, likely planted right after tillage
        planting_date = tillage_date + timedelta(days=1)  # Day after tillage
        confidence = "medium"
    else:
        # If longer gap, estimate planting about 7-10 days before first rise (typical emergence time)
        planting_date = first_rise_date - timedelta(days=7)  # ~7 days before first rise
        confidence = "medium"
    
    # Make sure estimated planting date isn't before tillage
    if planting_date < tillage_date:
        planting_date = tillage_date + timedelta(days=1)
    
    # Adjust confidence based on rise rate and pattern clarity
    if best_pattern["rise_rate"] > 0.015 and days_diff < 20:
        confidence = "high"
    
    return {
        "pattern_detected": True,
        "likely_planting_date": planting_date.strftime("%Y-%m-%d"),
        "confidence": confidence,
        "tillage_date": best_pattern["tillage_date"],
        "rise_rate": best_pattern["rise_rate"],
        "pattern_details": {
            "tillage_date": best_pattern["tillage_date"],
            "first_rise_date": best_pattern["rise_points"][0]["date"],
            "days_between": days_diff
        }
    }

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
        temperature_data = data.get("temperature_data", [])
        gdd_data = data.get("gdd_data", [])
        gdd_stats = data.get("gdd_stats", {})
        temperature_summary = data.get("temperature_summary", {})
        base_temperature = data.get("base_temperature", 10)
        
        # NEW: Detect the weed → tillage → crop emergence pattern
        planting_pattern = detect_planting_pattern(ndvi_data)
        smart_planting_date = None
        planting_pattern_detected = False
        
        # Use detected planting date if pattern was found
        if planting_pattern["pattern_detected"]:
            smart_planting_date = planting_pattern["likely_planting_date"]
            planting_pattern_detected = True
            print(f"Detected planting pattern! Likely planting date: {smart_planting_date}, Confidence: {planting_pattern['confidence']}")
        
        # Determine which planting date to use, prioritize detected pattern if confidence is high
        if smart_planting_date and planting_pattern["confidence"] != "low":
            if not estimated_planting_date:
                # No existing estimate, use detected date
                estimated_planting_date = smart_planting_date
                print(f"Using detected planting date: {estimated_planting_date}")
            elif planting_pattern["confidence"] == "high":
                # Overwrite existing estimate if detection confidence is high
                print(f"Overwriting existing planting date {estimated_planting_date} with high-confidence detection: {smart_planting_date}")
                estimated_planting_date = smart_planting_date
            else:
                # Both estimates exist, keep both for reference in the prompt
                print(f"Keeping both estimates: original {estimated_planting_date} and detected {smart_planting_date}")
        
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
        
        # Construct base prompt
        base_prompt = f"""You are an intelligent agronomic assistant embedded inside the Yieldera platform. Your task is to generate insightful crop development commentary based on NDVI trends, rainfall data, temperature patterns, GDD information, field location, and known crop properties.

🌾 Background
Each analysis request includes:
- Crop Type: {crop}
- Variety: {variety}
- Irrigation Status: {'Irrigated' if irrigated == 'Yes' else 'Rainfed'}
- Latitude and Longitude: {latitude}, {longitude}
- NDVI Time Series: {ndvi_formatted}
- Rainfall Time Series: {rainfall_formatted}
- Temperature Data: {temp_formatted}
- Growing Degree Days: {gdd_formatted}
- Analysis Date Range: {date_range}"""

        # Add planting date information to prompt, including pattern detection if found
        planting_date_section = """
🌱 Planting Date Information:"""

        if planting_pattern_detected:
            planting_date_section += f"""
- IMPORTANT: System detected a tillage → planting → emergence pattern in the NDVI data!
- Detected pattern: Initially high NDVI (weeds) → Sharp drop (tillage on {planting_pattern["tillage_date"]}) → Consistent rise (crop emergence)
- AI-estimated planting date: {smart_planting_date} (Confidence: {planting_pattern["confidence"]})"""
            
            if estimated_planting_date and estimated_planting_date != smart_planting_date:
                planting_date_section += f"""
- Previously estimated planting date: {estimated_planting_date}
- You should favor the AI-detected planting date since it accounts for the tillage pattern."""
            else:
                planting_date_section += """
- Use this detected planting date as your reference point for growth stage estimation."""
        else:
            if estimated_planting_date:
                planting_date_section += f"""
- Estimated planting date: {estimated_planting_date}
- No clear tillage → planting → emergence pattern was detected in the NDVI data."""
            else:
                planting_date_section += """
- No planting date information available
- Please infer planting date from NDVI trends, rainfall patterns, and temperature data."""

        # Add the agronomic intelligence section
        agronomic_section = """
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
- If crop/variety is unknown or labeled 'testing', use general NDVI + rainfall pattern logic and suggest entering known values for deeper insights."""

        # Add pattern detection guidance if relevant
        pattern_guidance = ""
        if planting_pattern_detected:
            pattern_guidance = """
🔍 NDVI Pattern Recognition:
- The system detected a clear weed → tillage → crop emergence pattern in this field.
- This is common in real agricultural fields and can help pinpoint actual planting date.
- High initial NDVI does NOT always indicate early crop establishment; it may represent weeds.
- Sharp drops in NDVI often indicate tillage operations, with planting shortly after.
- Focus on the consistent rise in NDVI AFTER the detected tillage date."""

        # Complete the prompt
        analysis_section = """
🧠 Your Analysis Must Include:
1. NDVI Pattern Interpretation (flat, rising, declining)
2. Temperature & GDD Implications (if data available)
3. Rainfall Response (rainfed crop triggers, dry periods)
4. Crop Status Summary (bare soil, emergence, stress, maturity)
5. Planting Date Inference (estimate based on NDVI/rainfall/GDD)
6. Confidence Rating (High, Medium, Low)

🧭 Examples of Language to Use:
- "NDVI remained flat at ~0.18, indicating bare soil or no active vegetation."
- "Rising temperatures and accumulated GDD of 120 indicate favorable conditions for emergence."
- "NDVI increase after Dec 3 suggests planting occurred in late Nov."
- "Rainfall was insufficient to support rainfed planting."
- "NDVI decline suggests senescence or water stress."
- "Crop variety is unrecognized -- general vegetation analysis applied."

🧵 Output Format:
Respond in 2--4 sentences as a trained agronomist advising a field agent or insurer. Avoid referencing GPT, AI, or farmer-declared dates."""

        # Combine all prompt sections
        prompt = base_prompt + planting_date_section + agronomic_section
        
        # Add pattern guidance only if pattern was detected
        if pattern_guidance:
            prompt += pattern_guidance
            
        prompt += analysis_section

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
            
            # Include planting pattern detection results in response
            response_data = {
                "success": True,
                "insight": insight,
                "confidence_level": confidence_level
            }
            
            # Add planting pattern information if detected
            if planting_pattern_detected:
                response_data["planting_pattern"] = {
                    "detected": True,
                    "likely_planting_date": smart_planting_date,
                    "confidence": planting_pattern["confidence"],
                    "tillage_date": planting_pattern["tillage_date"]
                }
            
            return jsonify(response_data)
            
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
