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

# NEW FUNCTION: Detect rainfall trigger events without NDVI response
def detect_rainfall_without_emergence(ndvi_data, rainfall_data, min_rainfall_threshold=10, ndvi_threshold=0.2, response_window_days=14):
    """
    Detect significant rainfall events that aren't followed by crop emergence.
    
    Args:
        ndvi_data: List of NDVI readings with date and ndvi value
        rainfall_data: List of rainfall readings with date and rainfall value
        min_rainfall_threshold: Minimum rainfall to consider significant (mm)
        ndvi_threshold: NDVI threshold below which we consider no emergence
        response_window_days: Number of days to check for NDVI response after rainfall
    
    Returns:
        Dictionary with information about potential planting failure events
    """
    if not rainfall_data or not ndvi_data:
        return None
    
    # Sort data by date
    sorted_ndvi = sorted(ndvi_data, key=lambda x: x['date'])
    sorted_rainfall = sorted(rainfall_data, key=lambda x: x['date'])
    
    # Detect significant rainfall events (>10mm)
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
    
    # If no significant rainfall events found, return None
    if not significant_rainfall_events:
        return None
    
    # Check each significant rainfall event
    failure_events = []
    
    for rain_event in significant_rainfall_events:
        rain_date = datetime.strptime(rain_event['date'], '%Y-%m-%d')
        rain_date_str = rain_event['date']
        
        # Define the response window period
        response_end_date = rain_date + timedelta(days=response_window_days)
        
        # Find NDVI readings in the response window period
        window_ndvi_readings = []
        for ndvi_point in sorted_ndvi:
            try:
                ndvi_date = datetime.strptime(ndvi_point['date'], '%Y-%m-%d')
                if rain_date <= ndvi_date <= response_end_date:
                    window_ndvi_readings.append(ndvi_point)
            except (ValueError, KeyError) as e:
                print(f"Error processing NDVI date: {e}")
                continue
        
        # Analyze NDVI readings in response window
        if window_ndvi_readings:
            # If we have at least 2 readings in the window
            if len(window_ndvi_readings) >= 2:
                # Check if all NDVI readings remain below threshold
                all_below_threshold = all(reading['ndvi'] < ndvi_threshold for reading in window_ndvi_readings)
                
                # Calculate the time span of the readings
                first_date = datetime.strptime(window_ndvi_readings[0]['date'], '%Y-%m-%d')
                last_date = datetime.strptime(window_ndvi_readings[-1]['date'], '%Y-%m-%d')
                days_span = (last_date - first_date).days
                
                # Only consider it a failure if we have at least 7 days of low NDVI readings
                if all_below_threshold and days_span >= 7:
                    failure_events.append({
                        'rainfall_date': rain_date_str,
                        'rainfall_amount': rain_event['rainfall'],
                        'ndvi_readings': len(window_ndvi_readings),
                        'max_ndvi': max(reading['ndvi'] for reading in window_ndvi_readings),
                        'days_monitored': days_span
                    })
    
    # Return the most recent failure event if any found
    if failure_events:
        # Sort by rainfall date, most recent first
        failure_events.sort(key=lambda x: x['rainfall_date'], reverse=True)
        
        selected_event = failure_events[0]
        rainfall_date = format_date_for_display(selected_event['rainfall_date'])
        
        return {
            'detected': True,
            'message': f"Significant rainfall occurred around {rainfall_date} ({selected_event['rainfall_amount']:.1f}mm), which may have provided a planting opportunity. However, no NDVI response was observed in the following {selected_event['days_monitored']} days, suggesting either planting did not occur or the crop failed to emerge. This may indicate poor germination conditions or unsuccessful crop establishment.",
            'confidence': "low",
            'rainfall_date': selected_event['rainfall_date'],
            'rainfall_amount': selected_event['rainfall_amount'],
            'max_ndvi': selected_event['max_ndvi']
        }
    
    return None

# NEW FUNCTION: Detect post-tillage emergence and estimate planting window
def detect_post_tillage_emergence(ndvi_data, crop_type, tillage_date, irrigated, rainfall_data=None):
    """
    Detects emergence after a tillage event and estimates planting window.
    
    Args:
        ndvi_data: List of NDVI readings with date and ndvi value
        crop_type: Type of crop (determines emergence window)
        tillage_date: Date of the tillage event (when NDVI was at its lowest)
        irrigated: Whether the field is irrigated
        rainfall_data: List of rainfall readings (for rainfed fields)
        
    Returns:
        Dictionary with information about the post-tillage emergence and planting window
    """
    # Sort NDVI data by date
    sorted_ndvi = sorted(ndvi_data, key=lambda x: x['date'])
    
    # Convert tillage_date to datetime object
    tillage_date_obj = datetime.strptime(tillage_date, '%Y-%m-%d')
    
    # Filter NDVI data to only include readings after the tillage date
    post_tillage_ndvi = [point for point in sorted_ndvi if datetime.strptime(point['date'], '%Y-%m-%d') >= tillage_date_obj]
    
    # If no post-tillage data, we can't estimate emergence
    if not post_tillage_ndvi or len(post_tillage_ndvi) < 2:
        return {
            "emergenceDate": None,
            "plantingWindowStart": None,
            "plantingWindowEnd": None,
            "preEstablished": False,
            "confidence": "low",
            "message": "Insufficient data after tillage event to detect emergence.",
            "tillage_date": tillage_date
        }
    
    # Find the emergence date after tillage (when NDVI rises above threshold)
    emergence_date = None
    emergence_index = -1
    
    # Check for clear emergence point (crossing EMERGENCE_THRESHOLD)
    for i in range(len(post_tillage_ndvi) - 1):
        if post_tillage_ndvi[i]['ndvi'] < EMERGENCE_THRESHOLD and post_tillage_ndvi[i + 1]['ndvi'] >= EMERGENCE_THRESHOLD:
            emergence_date = post_tillage_ndvi[i + 1]['date']
            emergence_index = i + 1
            break
    
    # If no clear emergence, look for significant rise in NDVI
    if not emergence_date:
        # Look for any significant rise in NDVI (even if below threshold)
        for i in range(len(post_tillage_ndvi) - 1):
            ndvi_increase = post_tillage_ndvi[i + 1]['ndvi'] - post_tillage_ndvi[i]['ndvi']
            if ndvi_increase > 0.05:  # Significant rise threshold
                emergence_date = post_tillage_ndvi[i + 1]['date']
                emergence_index = i + 1
                break
    
    # If still no emergence detected
    if not emergence_date:
        return {
            "emergenceDate": None,
            "plantingWindowStart": None,
            "plantingWindowEnd": None,
            "preEstablished": False,
            "confidence": "low",
            "message": "No clear emergence pattern detected after tillage event.",
            "tillage_date": tillage_date
        }
    
    # Get emergence window for the crop type
    emergence_window = EMERGENCE_WINDOWS.get(crop_type, DEFAULT_EMERGENCE_WINDOW)
    
    # Calculate planting window by rolling back from emergence date
    emergence_date_obj = datetime.strptime(emergence_date, '%Y-%m-%d')
    planting_window_end = (emergence_date_obj - timedelta(days=emergence_window[0])).strftime('%Y-%m-%d')
    planting_window_start = (emergence_date_obj - timedelta(days=emergence_window[1])).strftime('%Y-%m-%d')
    
    # If field is rainfed, check for significant rainfall events before emergence but after tillage
    rainfall_adjusted_planting = None
    if irrigated == "No" and rainfall_data:
        # Filter rainfall events between tillage and emergence
        significant_rainfall_events = []
        for event in rainfall_data:
            event_date = event.get('date')
            if not event_date:
                continue
            
            try:
                event_date_obj = datetime.strptime(event_date, '%Y-%m-%d')
                if (tillage_date_obj <= event_date_obj < emergence_date_obj and
                    event.get('rainfall', 0) >= SIGNIFICANT_RAINFALL):
                    significant_rainfall_events.append(event)
            except Exception as e:
                print(f"Error processing rainfall event: {e}")
                continue
        
        # If significant rainfall events found, adjust estimate
        if significant_rainfall_events:
            # Sort rainfall events by date
            significant_rainfall_events.sort(key=lambda x: x['date'])
            
            # Use the first significant rainfall as the likely planting date
            rainfall_adjusted_planting = significant_rainfall_events[0]['date']
            print(f"Found rainfall-adjusted planting date after tillage: {rainfall_adjusted_planting}")
    
    # Determine confidence level
    confidence = "medium"
    
    # Higher confidence if we have good NDVI data and clear pattern
    if len(post_tillage_ndvi) >= 4 and emergence_index > 0 and emergence_index < len(post_tillage_ndvi) - 1:
        confidence = "high"
    
    # Lower confidence if sparse data or emergence is at the edge of the dataset
    if len(post_tillage_ndvi) < 3 or emergence_index == 0 or emergence_index == len(post_tillage_ndvi) - 1:
        confidence = "low"
    
    # Format dates for display
    tillage_display = format_date_for_display(tillage_date)
    emergence_display = format_date_for_display(emergence_date)
    planting_start_display = format_date_for_display(planting_window_start)
    planting_end_display = format_date_for_display(planting_window_end)
    
    # Create message about tillage and subsequent planting
    if rainfall_adjusted_planting:
        rainfall_date_display = format_date_for_display(rainfall_adjusted_planting)
        message = f"The crop appears to have been initially established before the analysis period. However, a sharp NDVI decline around {tillage_display} followed by a steady increase suggests a tillage or replanting event, with new emergence detected around {emergence_display}. Rainfall data suggests replanting likely occurred around {rainfall_date_display}."
    else:
        message = f"The crop appears to have been initially established before the analysis period. However, a sharp NDVI decline around {tillage_display} followed by a steady increase suggests a tillage or replanting event, with new emergence likely occurring around {emergence_display}. This indicates a likely replanting window between {planting_start_display} and {planting_end_display}."
    
    return {
        "emergenceDate": emergence_date,
        "plantingWindowStart": planting_window_start,
        "plantingWindowEnd": planting_window_end,
        "rainfallAdjustedPlanting": rainfall_adjusted_planting,
        "preEstablished": False,
        "confidence": confidence,
        "message": message,
        "tillage_date": tillage_date,
        "tillage_replanting_detected": True
    }

# Function to detect emergence and estimate planting window
def detect_emergence_and_estimate_planting(ndvi_data, crop_type, irrigated, rainfall_data=None):
    # Sort NDVI data by date
    sorted_ndvi = sorted(ndvi_data, key=lambda x: x['date'])
    
    # Find the emergence date (when NDVI consistently rises above threshold)
    emergence_date = None
    emergence_index = -1
    
    # Find consecutive readings above threshold
    for i in range(len(sorted_ndvi) - 1):
        if sorted_ndvi[i]['ndvi'] < EMERGENCE_THRESHOLD and sorted_ndvi[i + 1]['ndvi'] >= EMERGENCE_THRESHOLD:
            # Found potential emergence point
            emergence_date = sorted_ndvi[i + 1]['date']
            emergence_index = i + 1
            break
    
    # If no emergence detected, check if all values are above threshold (pre-established crop)
    if not emergence_date and sorted_ndvi[0]['ndvi'] >= EMERGENCE_THRESHOLD:
        return {
            "emergenceDate": None,
            "plantingWindowStart": None,
            "plantingWindowEnd": None,
            "preEstablished": True,
            "confidence": "high",
            "message": "Crop was already established before the analysis period began."
        }
    
    # If still no emergence detected
    if not emergence_date:
        # Look for any significant rise in NDVI (even if below threshold)
        for i in range(len(sorted_ndvi) - 1):
            if sorted_ndvi[i + 1]['ndvi'] - sorted_ndvi[i]['ndvi'] > 0.05:
                emergence_date = sorted_ndvi[i + 1]['date']
                emergence_index = i + 1
                break
        
        # If still nothing, we can't determine emergence
        if not emergence_date:
            # NEW: For rainfed fields, check if there was rainfall without emergence
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
                        "rainfall_date": rainfall_failure['rainfall_date']
                    }
            
            return {
                "emergenceDate": None,
                "plantingWindowStart": None,
                "plantingWindowEnd": None,
                "preEstablished": False,
                "confidence": "low",
                "message": "No clear emergence pattern detected in NDVI data."
            }
    
    # Get emergence window for the crop type
    emergence_window = EMERGENCE_WINDOWS.get(crop_type, DEFAULT_EMERGENCE_WINDOW)
    
    # Calculate planting window by rolling back from emergence date
    emergence_date_obj = datetime.strptime(emergence_date, '%Y-%m-%d')
    planting_window_end = (emergence_date_obj - timedelta(days=emergence_window[0])).strftime('%Y-%m-%d')
    planting_window_start = (emergence_date_obj - timedelta(days=emergence_window[1])).strftime('%Y-%m-%d')
    
    # If field is rainfed, check for significant rainfall events before emergence
    rainfall_adjusted_planting = None
    if irrigated == "No" and rainfall_data:
        # Filter rainfall events before emergence but within the estimated planting window
        significant_rainfall_events = []
        for event in rainfall_data:
            event_date = event.get('date')
            if not event_date:
                continue
            
            try:
                event_date_obj = datetime.strptime(event_date, '%Y-%m-%d')
                if (event_date_obj < emergence_date_obj and 
                    event_date_obj >= datetime.strptime(planting_window_start, '%Y-%m-%d') and
                    event.get('rainfall', 0) >= SIGNIFICANT_RAINFALL):
                    significant_rainfall_events.append(event)
            except Exception as e:
                print(f"Error processing rainfall event: {e}")
                continue
        
        # If significant rainfall events found, adjust estimate
        if significant_rainfall_events:
            # Sort rainfall events by date
            significant_rainfall_events.sort(key=lambda x: x['date'])
            
            # Use the first significant rainfall as the likely planting date
            rainfall_adjusted_planting = significant_rainfall_events[0]['date']
            print(f"Found rainfall-adjusted planting date: {rainfall_adjusted_planting}")
    
    # Determine confidence level
    confidence = "medium"
    
    # Higher confidence if we have good NDVI data and clear pattern
    if len(sorted_ndvi) >= 6 and emergence_index > 0 and emergence_index < len(sorted_ndvi) - 1:
        confidence = "high"
    
    # Lower confidence if sparse data or emergence is at the edge of the dataset
    if len(sorted_ndvi) < 4 or emergence_index == 0 or emergence_index == len(sorted_ndvi) - 1:
        confidence = "low"
    
    return {
        "emergenceDate": emergence_date,
        "plantingWindowStart": planting_window_start,
        "plantingWindowEnd": planting_window_end,
        "rainfallAdjustedPlanting": rainfall_adjusted_planting,
        "preEstablished": False,
        "confidence": confidence,
        "message": "Emergence detected with clear pattern."
    }

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
        tillage_date = None
        drop_start_idx = -1
        drop_end_idx = -1
        
        if len(ndvi_data) >= 3:
            sorted_ndvi = sorted(ndvi_data, key=lambda x: x['date'])
            # Look for high -> low -> high pattern
            max_drop = 0
            
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
                    tillage_date = sorted_ndvi[drop_end_idx]['date']
                    tillage_info = (f"Potential tillage/replanting detected: NDVI dropped from {sorted_ndvi[drop_start_idx]['ndvi']:.2f} "
                                   f"to {sorted_ndvi[drop_end_idx]['ndvi']:.2f} around {sorted_ndvi[drop_end_idx]['date']}, "
                                   f"then rose again afterward")
                    print(f"Tillage/replanting detected on {tillage_date}")
        
        # Check for consistently high NDVI values - ONLY if no tillage/replanting pattern detected
        consistently_high_ndvi = False
        if ndvi_data and not tillage_replanting_detected:
            sorted_ndvi = sorted(ndvi_data, key=lambda x: x['date'])
            ndvi_values = [item['ndvi'] for item in sorted_ndvi]
            # Consider consistently high if all values are above 0.4
            if all(val > 0.4 for val in ndvi_values):
                consistently_high_ndvi = True
                high_ndvi_min = min(ndvi_values)
                high_ndvi_max = max(ndvi_values)
                high_ndvi_avg = sum(ndvi_values) / len(ndvi_values)
                print(f"Detected consistently high NDVI: min={high_ndvi_min:.2f}, max={high_ndvi_max:.2f}, avg={high_ndvi_avg:.2f}")
        
        # Smart planting date estimation
        planting_date_results = None
        planting_window_text = None
        
        # If tillage/replanting detected, use specialized function
        if tillage_replanting_detected and tillage_date:
            try:
                print(f"Using post-tillage emergence detection for tillage date: {tillage_date}")
                planting_date_results = detect_post_tillage_emergence(
                    ndvi_data=ndvi_data,
                    crop_type=crop,
                    tillage_date=tillage_date,
                    irrigated=irrigated,
                    rainfall_data=rainfall_data if irrigated == "No" else None
                )
                
                # Use the message from the post-tillage function
                planting_window_text = planting_date_results["message"]
                print(f"Post-tillage planting window: {planting_window_text}")
                
            except Exception as e:
                print(f"Error estimating post-tillage planting window: {e}")
                planting_window_text = "Error estimating planting window after tillage event."
        
        # Otherwise use regular planting date estimation
        elif not consistently_high_ndvi and ndvi_data and len(ndvi_data) > 1:
            try:
                planting_date_results = detect_emergence_and_estimate_planting(
                    ndvi_data=ndvi_data, 
                    crop_type=crop, 
                    irrigated=irrigated,
                    rainfall_data=rainfall_data if irrigated == "No" else None
                )
                
                # Format planting window for the prompt
                if planting_date_results["preEstablished"]:
                    planting_window_text = "Crop was already established before the analysis period began."
                elif "rainfall_without_emergence" in planting_date_results and planting_date_results["rainfall_without_emergence"]:
                    # Handle the new case of rainfall without emergence
                    planting_window_text = planting_date_results["message"]
                elif planting_date_results["plantingWindowStart"] and planting_date_results["plantingWindowEnd"]:
                    start_formatted = format_date_for_display(planting_date_results["plantingWindowStart"])
                    end_formatted = format_date_for_display(planting_date_results["plantingWindowEnd"])
                    
                    # Use rainfall-adjusted date if available for rainfed fields
                    if irrigated == "No" and planting_date_results["rainfallAdjustedPlanting"]:
                        rainfall_date = format_date_for_display(planting_date_results["rainfallAdjustedPlanting"])
                        planting_window_text = f"Likely planted between {start_formatted} and {end_formatted}, with rainfall data suggesting planting occurred around {rainfall_date}."
                    else:
                        planting_window_text = f"Likely planted between {start_formatted} and {end_formatted}."
                else:
                    planting_window_text = "Unable to determine a clear planting window from the available data."
                
                print(f"Planting window estimate: {planting_window_text}")
                
            except Exception as e:
                print(f"Error estimating planting window: {e}")
                planting_window_text = "Error estimating planting window."
        
        # Create special instruction for consistently high NDVI (only if no tillage/replanting)
        high_ndvi_instruction = ""
        if consistently_high_ndvi and not tillage_replanting_detected:
            high_ndvi_instruction = """
IMPORTANT: The NDVI data shows consistently high values (>0.4) throughout the entire analysis period. 
This indicates the crop was already well-established before the analysis period began.
DO NOT attempt to estimate a planting date. Instead, clearly state that the crop was already established
before the beginning of the analysis period.
"""
        
        # Create special instruction for tillage/replanting
        tillage_instruction = ""
        if tillage_replanting_detected:
            tillage_instruction = f"""
IMPORTANT: The NDVI data shows a clear tillage or replanting pattern with initial high values 
followed by a sharp decline around {format_date_for_display(tillage_date)} and then a steady rise afterward.
This pattern indicates an initial crop was established, then the field was tilled or harvested,
followed by new crop emergence. 

DO NOT conclude that "the crop was already established before the analysis period began" - 
this would ignore the second crop cycle. Instead, highlight the tillage/replanting event and subsequent regrowth.
"""
        
        # Create planting date instruction
        planting_date_instruction = ""
        if planting_window_text:
            planting_date_instruction = f"""
IMPORTANT: Based on crop-specific emergence windows and NDVI patterns, our system has determined:
{planting_window_text}

YOU MUST USE THIS EXACT PLANTING WINDOW STATEMENT in your response. 
DO NOT modify it, rephrase it, or use different dates.
"""
        
        # Construct prompt with enhanced information including tillage detection, NDVI change rates, and smart planting date
        prompt = f"""You are an intelligent agronomic assistant embedded inside the Yieldera platform. Your task is to generate insightful crop development commentary based on NDVI trends, {'rainfall data, ' if irrigated == 'No' else ''}temperature patterns, GDD information, field location, and known crop properties.

{high_ndvi_instruction}
{tillage_instruction}
{planting_date_instruction}

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
- Always use the exact planting date statement provided above. Do not create your own planting date estimate.
- If NDVI values remain consistently high throughout the period (all above 0.4) AND no tillage pattern is detected, conclude that the crop was already established before the analysis period began.
- For rainfed fields where significant rainfall (>10mm) occurred but NDVI remained low (<0.2), this may indicate planting failure or poor germination. In these cases, mention this possibility as provided in the planting date statement.

🧠 Your Analysis Must Include:
1. NDVI Pattern Interpretation (flat, rising, declining, or mixed patterns including tillage events)
2. Temperature & GDD Implications (if data available)
3. {'Rainfall Response (rainfed crop triggers, dry periods)' if irrigated == 'No' else 'Irrigation & Crop Management Implications'}
4. Crop Status Summary (bare soil, emergence, stress, maturity, potential replanting)
5. Planting Date Statement - use EXACTLY the statement provided above
6. Confidence Rating (High, Medium, Low)

🧭 Examples of Language to Use:
- "NDVI remained flat at ~0.18, indicating bare soil or no active vegetation."
- "The NDVI drop from 0.6 to 0.2 followed by a rise suggests tillage and replanting occurred in mid-December."
- "Rising temperatures and accumulated GDD of 120 indicate favorable conditions for emergence."
- "NDVI shows a rapid increase rate of 0.05 per day after Jan 15, indicating vigorous early growth."
{'' if irrigated == 'Yes' else '- "Rainfall was insufficient to support rainfed planting."'}
- "NDVI decline suggests senescence or water stress."
- "Crop variety is unrecognized -- general vegetation analysis applied."
- "NDVI values remained consistently high throughout the period, suggesting the crop was already established before the analysis period began."
- "Despite significant rainfall on April 15, the lack of NDVI response suggests potential planting failure or poor germination conditions."

🧵 Output Format:
Respond in 2--3 sentences as a trained agronomist advising a field agent or insurer. Always include the exact planting date statement provided above. Avoid referencing GPT, AI, or farmer-declared dates."""

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
            
            # If we have planting date results from our algorithm, use that confidence
            if planting_date_results and planting_date_results.get("confidence"):
                confidence_level = planting_date_results["confidence"]
                print(f"Using algorithm-determined confidence level: {confidence_level}")
            
            # IMPROVED CONFIDENCE LOGIC: Boost confidence based on data quality
            if confidence_level != "high" and ndvi_data and len(ndvi_data) >= 10:
                # Check if NDVI pattern is consistent
                ndvi_values = [item["ndvi"] for item in ndvi_data]
                ndvi_std_dev = calculate_std_dev(ndvi_values)
                
                # If we have low cloud cover and consistent NDVI pattern, boost confidence
                if avg_cloud_cover is not None and avg_cloud_cover < 20 and ndvi_std_dev < 0.15:
                    confidence_level = "high"
                    print(f"Boosted confidence to high based on data quality: {len(ndvi_data)} observations, {avg_cloud_cover:.1f}% cloud cover")
            
            # If we detected consistently high NDVI (but no tillage), set confidence high
            if consistently_high_ndvi and not tillage_replanting_detected:
                confidence_level = "high"
                print("Set confidence to high for consistently high NDVI pattern (pre-established crop)")
            
            # If we detected rainfall without emergence, ensure confidence is low
            if planting_date_results and "rainfall_without_emergence" in planting_date_results and planting_date_results["rainfall_without_emergence"]:
                confidence_level = "low"
                print("Set confidence to low for rainfall without emergence pattern")
            
            # Add planting date estimation results to the response
            response_data = {
                "success": True,
                "insight": insight,
                "confidence_level": confidence_level,
                "tillage_detected": tillage_replanting_detected,
                "consistently_high_ndvi": consistently_high_ndvi and not tillage_replanting_detected
            }
            
            # Add planting date estimation if available
            if planting_date_results:
                response_data["planting_date_estimation"] = {
                    "emergence_date": planting_date_results.get("emergenceDate"),
                    "planting_window_start": planting_date_results.get("plantingWindowStart"),
                    "planting_window_end": planting_date_results.get("plantingWindowEnd"),
                    "rainfall_adjusted_planting": planting_date_results.get("rainfallAdjustedPlanting"),
                    "pre_established": planting_date_results.get("preEstablished", False) and not tillage_replanting_detected,
                    "confidence": planting_date_results.get("confidence"),
                    "message": planting_date_results.get("message"),
                    "formatted_planting_window": planting_window_text,
                    "rainfall_without_emergence": planting_date_results.get("rainfall_without_emergence", False),
                    "tillage_replanting_detected": tillage_replanting_detected,
                    "tillage_date": tillage_date if tillage_replanting_detected else None
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
