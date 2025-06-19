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

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Define crop-specific emergence windows (in days)
EMERGENCE_WINDOWS = {
    "Maize": (6, 10),
    "Soyabeans": (7, 11),
    "Sorghum": (6, 10),
    "Cotton": (5, 9),
    "Groundnuts": (6, 10),
    "Barley": (7, 11),
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

# PRIMARY FUNCTION: Detect initial emergence and estimate planting window
def detect_primary_emergence_and_planting(ndvi_data, crop_type, irrigated, rainfall_data=None):
    """
    Detects the FIRST emergence event and estimates the primary planting window.
    This should be the main focus of planting date estimation.
    
    Args:
        ndvi_data: List of NDVI readings with date and ndvi value
        crop_type: Type of crop (determines emergence window)
        irrigated: Whether the field is irrigated
        rainfall_data: List of rainfall readings (for rainfed fields)
        
    Returns:
        Dictionary with primary planting date estimation
    """
    # Sort NDVI data by date
    sorted_ndvi = sorted(ndvi_data, key=lambda x: x['date'])
    
    # Find the FIRST emergence date (when NDVI consistently rises above threshold)
    emergence_date = None
    emergence_index = -1
    
    # Look for the first time NDVI crosses the emergence threshold
    for i in range(len(sorted_ndvi) - 1):
        if sorted_ndvi[i]['ndvi'] < EMERGENCE_THRESHOLD and sorted_ndvi[i + 1]['ndvi'] >= EMERGENCE_THRESHOLD:
            emergence_date = sorted_ndvi[i + 1]['date']
            emergence_index = i + 1
            print(f"Primary emergence detected on {emergence_date} at index {emergence_index}")
            break
    
    # If no clear threshold crossing, look for significant NDVI rise from low values
    if not emergence_date:
        for i in range(len(sorted_ndvi) - 1):
            current_ndvi = sorted_ndvi[i]['ndvi']
            next_ndvi = sorted_ndvi[i + 1]['ndvi']
            
            # Look for significant rise from low values (indicating emergence from bare soil)
            if current_ndvi < 0.15 and next_ndvi > current_ndvi + 0.05:
                emergence_date = sorted_ndvi[i + 1]['date']
                emergence_index = i + 1
                print(f"Alternative emergence detection on {emergence_date} - significant rise from low NDVI")
                break
    
    # Check if crop was pre-established (all values already high)
    if not emergence_date and sorted_ndvi and sorted_ndvi[0]['ndvi'] >= EMERGENCE_THRESHOLD:
        # Check if most values are consistently high
        high_values = [item['ndvi'] for item in sorted_ndvi if item['ndvi'] >= EMERGENCE_THRESHOLD]
        if len(high_values) >= len(sorted_ndvi) * 0.8:  # 80% of values are high
            return {
                "emergenceDate": None,
                "plantingWindowStart": None,
                "plantingWindowEnd": None,
                "preEstablished": True,
                "confidence": "high",
                "message": "Crop was already established before the analysis period began.",
                "primary_emergence": False
            }
    
    # If still no emergence detected, check for rainfall without emergence (rainfed only)
    if not emergence_date:
        if irrigated == "No" and rainfall_data:
            rainfall_failure = detect_rainfall_without_emergence(ndvi_data, rainfall_data)
            if rainfall_failure and rainfall_failure['detected']:
                return {
                    "emergenceDate": None,
                    "plantingWindowStart": None,
                    "plantingWindowEnd": None,
                    "preEstablished": False,
                    "confidence": "low",
                    "message": rainfall_failure['message'],
                    "rainfall_without_emergence": True,
                    "rainfall_date": rainfall_failure['rainfall_date'],
                    "primary_emergence": False
                }
        
        return {
            "emergenceDate": None,
            "plantingWindowStart": None,
            "plantingWindowEnd": None,
            "preEstablished": False,
            "confidence": "low",
            "message": "No clear emergence pattern detected in NDVI data.",
            "primary_emergence": False
        }
    
    # Calculate planting window based on crop-specific emergence timing
    emergence_window = EMERGENCE_WINDOWS.get(crop_type, DEFAULT_EMERGENCE_WINDOW)
    
    # Calculate planting window by rolling back from emergence date
    emergence_date_obj = datetime.strptime(emergence_date, '%Y-%m-%d')
    planting_window_end = (emergence_date_obj - timedelta(days=emergence_window[0])).strftime('%Y-%m-%d')
    planting_window_start = (emergence_date_obj - timedelta(days=emergence_window[1])).strftime('%Y-%m-%d')
    
    print(f"Calculated planting window: {planting_window_start} to {planting_window_end}")
    
    # For rainfed fields, check for rainfall events in the planting window
    rainfall_adjusted_planting = None
    if irrigated == "No" and rainfall_data:
        significant_rainfall_events = []
        planting_start_obj = datetime.strptime(planting_window_start, '%Y-%m-%d')
        
        for event in rainfall_data:
            event_date = event.get('date')
            if not event_date:
                continue
            
            try:
                event_date_obj = datetime.strptime(event_date, '%Y-%m-%d')
                if (planting_start_obj <= event_date_obj < emergence_date_obj and
                    event.get('rainfall', 0) >= SIGNIFICANT_RAINFALL):
                    significant_rainfall_events.append(event)
            except Exception as e:
                print(f"Error processing rainfall event: {e}")
                continue
        
        if significant_rainfall_events:
            # Use the first significant rainfall as the likely planting date
            significant_rainfall_events.sort(key=lambda x: x['date'])
            rainfall_adjusted_planting = significant_rainfall_events[0]['date']
            print(f"Found rainfall-adjusted planting date: {rainfall_adjusted_planting}")
    
    # Determine confidence level
    confidence = "medium"
    
    # Higher confidence for good data quality and clear patterns
    if len(sorted_ndvi) >= 6 and emergence_index > 0 and emergence_index < len(sorted_ndvi) - 1:
        confidence = "high"
    
    # Lower confidence for sparse data or edge cases
    if len(sorted_ndvi) < 4 or emergence_index <= 1 or emergence_index >= len(sorted_ndvi) - 2:
        confidence = "low"
    
    # Create primary planting message
    emergence_display = format_date_for_display(emergence_date)
    planting_start_display = format_date_for_display(planting_window_start)
    planting_end_display = format_date_for_display(planting_window_end)
    
    if rainfall_adjusted_planting:
        rainfall_date_display = format_date_for_display(rainfall_adjusted_planting)
        message = f"Primary emergence detected around {emergence_display}, indicating planting likely occurred between {planting_start_display} and {planting_end_display}. Rainfall data suggests planting occurred around {rainfall_date_display}."
    else:
        message = f"Primary emergence detected around {emergence_display}, indicating planting likely occurred between {planting_start_display} and {planting_end_display}."
    
    return {
        "emergenceDate": emergence_date,
        "plantingWindowStart": planting_window_start,
        "plantingWindowEnd": planting_window_end,
        "rainfallAdjustedPlanting": rainfall_adjusted_planting,
        "preEstablished": False,
        "confidence": confidence,
        "message": message,
        "primary_emergence": True
    }

# SECONDARY FUNCTION: Detect tillage/replanting events
def detect_tillage_replanting_events(ndvi_data, primary_emergence_date=None):
    """
    Detects tillage or replanting events AFTER the primary emergence.
    This is secondary analysis to complement the primary planting date.
    
    Args:
        ndvi_data: List of NDVI readings with date and ndvi value
        primary_emergence_date: Date of primary emergence (to avoid double-counting)
        
    Returns:
        Dictionary with tillage/replanting information
    """
    if len(ndvi_data) < 4:
        return {"tillage_detected": False, "message": ""}
    
    sorted_ndvi = sorted(ndvi_data, key=lambda x: x['date'])
    tillage_events = []
    
    # Look for significant drops followed by recovery
    for i in range(1, len(sorted_ndvi) - 1):
        current_drop = sorted_ndvi[i-1]['ndvi'] - sorted_ndvi[i]['ndvi']
        
        # Criteria for tillage: significant drop from established vegetation
        if (current_drop > 0.15 and 
            sorted_ndvi[i-1]['ndvi'] > 0.3 and
            sorted_ndvi[i]['ndvi'] < 0.25):
            
            # Check for subsequent recovery
            recovery_found = False
            for j in range(i + 1, min(i + 4, len(sorted_ndvi))):  # Look 3 readings ahead
                if sorted_ndvi[j]['ndvi'] > sorted_ndvi[i]['ndvi'] + 0.1:
                    recovery_found = True
                    break
            
            if recovery_found:
                tillage_date = sorted_ndvi[i]['date']
                
                # Avoid counting tillage that's close to primary emergence
                if primary_emergence_date:
                    try:
                        primary_date_obj = datetime.strptime(primary_emergence_date, '%Y-%m-%d')
                        tillage_date_obj = datetime.strptime(tillage_date, '%Y-%m-%d')
                        days_diff = abs((tillage_date_obj - primary_date_obj).days)
                        
                        # Skip if tillage is within 14 days of primary emergence
                        if days_diff < 14:
                            continue
                    except:
                        pass
                
                tillage_events.append({
                    'date': tillage_date,
                    'ndvi_before': sorted_ndvi[i-1]['ndvi'],
                    'ndvi_after': sorted_ndvi[i]['ndvi'],
                    'drop_magnitude': current_drop
                })
    
    if tillage_events:
        # Use the most significant tillage event
        most_significant = max(tillage_events, key=lambda x: x['drop_magnitude'])
        tillage_date_display = format_date_for_display(most_significant['date'])
        
        return {
            "tillage_detected": True,
            "tillage_date": most_significant['date'],
            "message": f"Subsequently, a tillage or replanting event was detected around {tillage_date_display}, where NDVI dropped from {most_significant['ndvi_before']:.2f} to {most_significant['ndvi_after']:.2f}, followed by recovery."
        }
    
    return {"tillage_detected": False, "message": ""}

# FUNCTION: Detect rainfall without emergence (unchanged)
def detect_rainfall_without_emergence(ndvi_data, rainfall_data, min_rainfall_threshold=10, ndvi_threshold=0.2, response_window_days=14):
    """
    Detect significant rainfall events that aren't followed by crop emergence.
    """
    if not rainfall_data or not ndvi_data:
        return None
    
    sorted_ndvi = sorted(ndvi_data, key=lambda x: x['date'])
    sorted_rainfall = sorted(rainfall_data, key=lambda x: x['date'])
    
    significant_rainfall_events = []
    
    for event in sorted_rainfall:
        try:
            if event.get('rainfall', 0) >= min_rainfall_threshold:
                significant_rainfall_events.append({
                    'date': event['date'],
                    'rainfall': event['rainfall']
                })
        except (KeyError, TypeError) as e:
            print(f"Error processing rainfall event: {e}")
            continue
    
    if not significant_rainfall_events:
        return None
    
    failure_events = []
    
    for rain_event in significant_rainfall_events:
        rain_date = datetime.strptime(rain_event['date'], '%Y-%m-%d')
        rain_date_str = rain_event['date']
        
        response_end_date = rain_date + timedelta(days=response_window_days)
        
        window_ndvi_readings = []
        for ndvi_point in sorted_ndvi:
            try:
                ndvi_date = datetime.strptime(ndvi_point['date'], '%Y-%m-%d')
                if rain_date <= ndvi_date <= response_end_date:
                    window_ndvi_readings.append(ndvi_point)
            except (ValueError, KeyError) as e:
                print(f"Error processing NDVI date: {e}")
                continue
        
        if window_ndvi_readings and len(window_ndvi_readings) >= 2:
            all_below_threshold = all(reading['ndvi'] < ndvi_threshold for reading in window_ndvi_readings)
            
            first_date = datetime.strptime(window_ndvi_readings[0]['date'], '%Y-%m-%d')
            last_date = datetime.strptime(window_ndvi_readings[-1]['date'], '%Y-%m-%d')
            days_span = (last_date - first_date).days
            
            if all_below_threshold and days_span >= 7:
                failure_events.append({
                    'rainfall_date': rain_date_str,
                    'rainfall_amount': rain_event['rainfall'],
                    'ndvi_readings': len(window_ndvi_readings),
                    'max_ndvi': max(reading['ndvi'] for reading in window_ndvi_readings),
                    'days_monitored': days_span
                })
    
    if failure_events:
        failure_events.sort(key=lambda x: x['rainfall_date'], reverse=True)
        selected_event = failure_events[0]
        rainfall_date = format_date_for_display(selected_event['rainfall_date'])
        
        return {
            'detected': True,
            'message': f"Significant rainfall occurred around {rainfall_date} ({selected_event['rainfall_amount']:.1f}mm), which may have provided a planting opportunity. However, no NDVI response was observed in the following {selected_event['days_monitored']} days, suggesting either planting did not occur or the crop failed to emerge.",
            'confidence': "low",
            'rainfall_date': selected_event['rainfall_date'],
            'rainfall_amount': selected_event['rainfall_amount'],
            'max_ndvi': selected_event['max_ndvi']
        }
    
    return None

# Format date for display (Month Day format)
def format_date_for_display(date_str):
    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        return date_obj.strftime('%B %d')
    except Exception:
        return date_str

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
        
        # Process rainfall data
        weekly_rainfall = {}
        if irrigated == "Yes":
            rainfall_formatted = "Not applicable for irrigated fields"
        elif rainfall_data:
            for item in rainfall_data:
                date = item.get('date')
                if date:
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
        
        # Calculate NDVI change rates
        ndvi_change_rates = []
        if len(ndvi_data) > 1:
            sorted_ndvi = sorted(ndvi_data, key=lambda x: x['date'])
            
            for i in range(1, len(sorted_ndvi)):
                try:
                    date1 = datetime.strptime(sorted_ndvi[i-1]['date'], '%Y-%m-%d')
                    date2 = datetime.strptime(sorted_ndvi[i]['date'], '%Y-%m-%d')
                    days_diff = (date2 - date1).days
                    
                    if days_diff > 0:
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

        # Format NDVI change rate data
        ndvi_change_formatted = "No data"
        if ndvi_change_rates:
            significant_changes = [r for r in ndvi_change_rates if abs(r['change_rate']) > 0.005]
            
            if significant_changes:
                significant_changes.sort(key=lambda x: abs(x['change_rate']), reverse=True)
                top_changes = significant_changes[:3]
                ndvi_change_formatted = ", ".join([
                    f"{c['start_date']} to {c['end_date']}: {c['change_rate']*100:.2f}% per day ({c['total_change']:.2f} over {c['days']} days)"
                    for c in top_changes
                ])
        
        # FIXED ANALYSIS FLOW: Primary emergence first, then tillage detection
        print("=== STARTING PRIMARY EMERGENCE ANALYSIS ===")
        
        # Step 1: Detect PRIMARY emergence and calculate planting window
        primary_results = detect_primary_emergence_and_planting(
            ndvi_data=ndvi_data,
            crop_type=crop,
            irrigated=irrigated,
            rainfall_data=rainfall_data if irrigated == "No" else None
        )
        
        print(f"Primary emergence results: {primary_results}")
        
        # Step 2: Detect SECONDARY tillage/replanting events
        tillage_results = detect_tillage_replanting_events(
            ndvi_data=ndvi_data,
            primary_emergence_date=primary_results.get("emergenceDate")
        )
        
        print(f"Tillage detection results: {tillage_results}")
        
        # Step 3: Combine primary and secondary analysis for final message
        planting_window_text = primary_results["message"]
        
        # Add tillage information if detected
        if tillage_results["tillage_detected"]:
            planting_window_text += " " + tillage_results["message"]
        
        # Determine overall confidence
        confidence_level = primary_results.get("confidence", "medium")
        
        # Boost confidence based on data quality
        if confidence_level != "high" and ndvi_data and len(ndvi_data) >= 10:
            ndvi_values = [item["ndvi"] for item in ndvi_data]
            ndvi_std_dev = calculate_std_dev(ndvi_values)
            
            if avg_cloud_cover is not None and avg_cloud_cover < 20 and ndvi_std_dev < 0.15:
                confidence_level = "high"
                print(f"Boosted confidence to high based on data quality")
        
        # Create instructions for AI prompt
        primary_instruction = f"""
IMPORTANT: PRIMARY PLANTING DATE ANALYSIS:
{planting_window_text}

YOU MUST USE THIS EXACT PLANTING ANALYSIS as the foundation of your response.
DO NOT modify it, rephrase it, or use different dates for the primary planting window.
"""
        
        tillage_instruction = ""
        if tillage_results["tillage_detected"]:
            tillage_instruction = f"""
SECONDARY INFORMATION: A tillage/replanting event was also detected around {format_date_for_display(tillage_results['tillage_date'])}.
This is ADDITIONAL context, not the primary planting date.
"""
        
        # Create optimized prompt
        prompt = f"""You are Yieldera's agronomic AI. Generate concise crop insight based on NDVI trends, {'rainfall, ' if irrigated == 'No' else ''}temperature, and GDD data.

{primary_instruction}
{tillage_instruction}

📊 Data:
- Crop: {crop} ({variety}) - {'Irrigated' if irrigated == 'Yes' else 'Rainfed'}
- Location: {latitude}, {longitude}
- NDVI: {ndvi_formatted}
- NDVI Changes: {ndvi_change_formatted}
{'' if irrigated == 'Yes' else f'- Rainfall: {rainfall_formatted}'}
- Temperature: {temp_formatted}
- GDD: {gdd_formatted}
- Period: {date_range}

🎯 Required:
1. NDVI Pattern (focus on primary emergence, mention tillage as secondary)
2. Temperature/GDD implications
3. {'Rainfall response' if irrigated == 'No' else 'Irrigation management'}
4. Crop status & exact planting statement above
5. Confidence (High/Medium/Low)

Respond in 2-3 sentences as field advisor. Lead with the primary planting analysis."""

        # Call OpenAI API
        try:
            print(f"Sending request to generate insight for field: {field_name}")
            
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are Yieldera's agricultural advisor. Focus on PRIMARY planting date estimation first, then mention any secondary events. DO NOT mention AI, GPT, or any third-party tools."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=250
            )
            
            insight = response.choices[0].message.content.strip()
            
            # Build comprehensive response
            response_data = {
                "success": True,
                "insight": insight,
                "confidence_level": confidence_level,
                "tillage_detected": tillage_results["tillage_detected"],
                "primary_emergence_detected": primary_results.get("primary_emergence", False)
            }
            
            # Add detailed planting date estimation
            if primary_results:
                response_data["planting_date_estimation"] = {
                    "emergence_date": primary_results.get("emergenceDate"),
                    "planting_window_start": primary_results.get("plantingWindowStart"),
                    "planting_window_end": primary_results.get("plantingWindowEnd"),
                    "rainfall_adjusted_planting": primary_results.get("rainfallAdjustedPlanting"),
                    "pre_established": primary_results.get("preEstablished", False),
                    "confidence": primary_results.get("confidence"),
                    "message": primary_results.get("message"),
                    "formatted_planting_window": planting_window_text,
                    "rainfall_without_emergence": primary_results.get("rainfall_without_emergence", False),
                    "primary_emergence": primary_results.get("primary_emergence", False)
                }
                
                # Add tillage information if detected
                if tillage_results["tillage_detected"]:
                    response_data["planting_date_estimation"]["tillage_event"] = {
                        "detected": True,
                        "date": tillage_results["tillage_date"],
                        "message": tillage_results["message"]
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

        # Progressive cloud filtering approach
        cloud_thresholds = [10, 20, 30, 50, 80]
        collection = None
        collection_size = 0
        
        for threshold in cloud_thresholds:
            # Get Sentinel-2 collection with cloud filtering
            collection = (
                ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
                .filterBounds(polygon)
                .filterDate(start, end)
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", threshold))
                .sort("CLOUDY_PIXEL_PERCENTAGE")  # Sort by cloud percentage
            )
            
            # Limit collection size based on cloud threshold
            limit_size = 10 if threshold <= 20 else (5 if threshold <= 50 else 3)
            collection = collection.limit(limit_size)
            
            # Check if we found any images
            collection_size = collection.size()
            if collection_size.getInfo() > 0:
                print(f"Found {collection_size.getInfo()} images with cloud threshold {threshold}%")
                break
        
        # Handle empty collection after all attempts
        if collection_size.getInfo() == 0:
            return jsonify({
                "success": False, 
                "error": "No Sentinel-2 imagery found for the specified date range and location",
                "empty_collection": True
            }), 404
        
        # Get median image and clip to polygon
        image = collection.median().clip(polygon)
        
        # Calculate NDVI
        ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
        rgb = image.select(["B4", "B3", "B2"])
        
        # Visualization settings
        ndvi_vis = ndvi.visualize(min=-0.5, max=1, palette=["blue", "red", "yellow", "green"])
        rgb_vis = rgb.visualize(min=0, max=3000)
        
        # Combine all metadata extraction into a single operation
        first_image = collection.first()
        
        # Create a combined operation to get NDVI stats, image date, and cloud percentage
        combined_data = ee.Dictionary({
            'ndvi_stats': ndvi.reduceRegion(
                reducer=ee.Reducer.mean().combine(
                    ee.Reducer.minMax(), "", True
                ),
                geometry=polygon,
                scale=10,
                maxPixels=1e9
            ),
            'image_date': first_image.date().format("YYYY-MM-dd"),
            'cloud_percentage': first_image.get("CLOUDY_PIXEL_PERCENTAGE"),
            'collection_size': collection_size
        })
        
        # Get map IDs for tile URLs with timeout handling
        try:
            map_id_ndvi = ee.data.getMapId({"image": ndvi_vis})
            map_id_rgb = ee.data.getMapId({"image": rgb_vis})
            
            # Single .getInfo() call to get all data
            all_data = combined_data.getInfo()
            ndvi_stats = all_data['ndvi_stats']
            image_date = all_data['image_date']
            cloud_percentage = all_data['cloud_percentage']
            collection_size_val = all_data['collection_size']
            
            # Return response with all data including cloud percentage
            response = {
                "success": True,
                "ndvi_tile_url": map_id_ndvi["tile_fetcher"].url_format,
                "rgb_tile_url": map_id_rgb["tile_fetcher"].url_format,
                "mean_ndvi": ndvi_stats.get("NDVI_mean"),
                "min_ndvi": ndvi_stats.get("NDVI_min"),
                "max_ndvi": ndvi_stats.get("NDVI_max"),
                "image_date": image_date,
                "collection_size": collection_size_val,
                "cloudy_pixel_percentage": cloud_percentage
            }
            
            print(f"Successfully processed NDVI request. Mean NDVI: {ndvi_stats.get('NDVI_mean')}, Cloud cover: {cloud_percentage}%")
            return jsonify(response)
            
        except Exception as e:
            print(f"Error getting map IDs: {e}")
            # Still return statistics even if visualization fails
            all_data = combined_data.getInfo()
            ndvi_stats = all_data['ndvi_stats']
            image_date = all_data['image_date']
            cloud_percentage = all_data['cloud_percentage']
            collection_size_val = all_data['collection_size']
            
            return jsonify({
                "success": True,
                "mean_ndvi": ndvi_stats.get("NDVI_mean"),
                "min_ndvi": ndvi_stats.get("NDVI_min"),
                "max_ndvi": ndvi_stats.get("NDVI_max"),
                "image_date": image_date,
                "collection_size": collection_size_val,
                "cloudy_pixel_percentage": cloud_percentage,
                "visualization_error": str(e)
            })

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
        cloud_thresholds = [10, 20, 30, 50, 80]
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
        
        # Use collection.map() to process all images on server side
        def process_image(image):
            """Process each image to extract NDVI, date, and cloud percentage"""
            # Clip to polygon
            clipped = image.clip(polygon)
            
            # Calculate NDVI
            ndvi = clipped.normalizedDifference(["B8", "B4"])
            
            # Calculate mean NDVI for the polygon
            ndvi_mean = ndvi.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=polygon,
                scale=10,
                maxPixels=1e9
            ).get('nd')
            
            # Create feature with all required data
            return ee.Feature(None, {
                'date': image.date().format('YYYY-MM-dd'),
                'ndvi': ndvi_mean,
                'cloud_percentage': image.get('CLOUDY_PIXEL_PERCENTAGE')
            })
        
        # Map the processing function over the entire collection
        processed_collection = collection.map(process_image)
        
        # Convert to FeatureCollection and get all data in a single call
        feature_collection = ee.FeatureCollection(processed_collection)
        
        # Single .getInfo() call to get all time series data
        features = feature_collection.getInfo()['features']
        
        # Extract NDVI time series data
        ndvi_time_series = []
        
        for feature in features:
            properties = feature['properties']
            ndvi_value = properties.get('ndvi')
            
            # Only add valid NDVI readings
            if ndvi_value is not None:
                ndvi_time_series.append({
                    "date": properties['date'],
                    "ndvi": ndvi_value,
                    "cloud_percentage": properties['cloud_percentage']
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
