import os
import json
import ee
import traceback
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv
import concurrent.futures
import threading

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize OpenAI client with faster model for insights
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Cache for GEE collections to avoid redundant API calls
collection_cache = {}
cache_lock = threading.Lock()

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

# OPTIMIZED: Cached collection getter
def get_cached_collection(coords, start, end):
    """Get cached collection or create new one"""
    cache_key = f"{str(coords)}_{start}_{end}"
    
    with cache_lock:
        if cache_key in collection_cache:
            return collection_cache[cache_key]
    
    # Create Earth Engine geometry
    polygon = ee.Geometry.Polygon(coords)
    
    # Base collection
    base_collection = (
        ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
        .filterBounds(polygon)
        .filterDate(start, end)
        .sort("CLOUDY_PIXEL_PERCENTAGE")
    )
    
    with cache_lock:
        collection_cache[cache_key] = base_collection
    
    return base_collection

# OPTIMIZED: Single server-side operation for cloud filtering
def get_best_collection_optimized(base_collection):
    """Get best collection using server-side operations only"""
    
    # Create all filtered collections in a single server-side operation
    filtered_collections = ee.List([10, 20, 30, 50, 80]).map(lambda threshold:
        ee.Dictionary({
            'threshold': threshold,
            'collection': base_collection.filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", threshold)),
            'limit': ee.Algorithms.If(ee.Number(threshold).lte(20), 10, 
                      ee.Algorithms.If(ee.Number(threshold).lte(50), 5, 3))
        })
    )
    
    # Find first collection with images using server-side logic
    def select_collection(collections):
        collections_with_size = collections.map(lambda item:
            ee.Dictionary(item).set('size', 
                ee.ImageCollection(ee.Dictionary(item).get('collection')).size())
        )
        
        valid_collections = collections_with_size.filter(ee.Filter.gt('size', 0))
        
        return ee.Algorithms.If(
            valid_collections.size().gt(0),
            ee.Dictionary(valid_collections.get(0)),
            ee.Dictionary({
                'collection': base_collection.limit(0),
                'size': 0,
                'threshold': 100
            })
        )
    
    best_collection_info = select_collection(filtered_collections)
    collection = ee.ImageCollection(best_collection_info.get('collection'))
    
    # Apply limit based on threshold
    limit_size = ee.Algorithms.If(
        ee.Number(best_collection_info.get('threshold')).lte(20), 10,
        ee.Algorithms.If(ee.Number(best_collection_info.get('threshold')).lte(50), 5, 3)
    )
    
    return collection.limit(limit_size), best_collection_info.get('size')

# NEW FUNCTION: Detect rainfall trigger events without NDVI response (unchanged)
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

# NEW FUNCTION: Detect post-tillage emergence and estimate planting window (unchanged)
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

# Function to detect emergence and estimate planting window (unchanged - just faster data processing)
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
        
        # OPTIMIZED: Pre-compute all statistical data in parallel using list comprehensions
        # Calculate average cloud cover if available
        cloud_percentages = [item["cloud_percentage"] for item in ndvi_data if item.get("cloud_percentage") is not None]
        avg_cloud_cover = sum(cloud_percentages) / len(cloud_percentages) if cloud_percentages else None
        
        # Format NDVI data (optimized)
        ndvi_formatted = ", ".join([f"{item['date']}: {item['ndvi']:.2f}" for item in ndvi_data[:10]]) if ndvi_data else "No data"
        if len(ndvi_data) > 10:
            ndvi_formatted += f" (+ {len(ndvi_data) - 10} more readings)"
        
        # Process rainfall data into weekly totals if available (optimized)
        weekly_rainfall = {}
        if irrigated == "Yes":
            rainfall_formatted = "Not applicable for irrigated fields"
        elif rainfall_data:
            for item in rainfall_data:
                date = item.get('date')
                if date:
                    week_key = date[:7] + "-W" + str((int(date[8:10]) - 1) // 7 + 1)
                    weekly_rainfall[week_key] = weekly_rainfall.get(week_key, 0) + item.get('rainfall', 0)
            rainfall_formatted = ", ".join([f"{week}: {total:.1f}mm" for week, total in weekly_rainfall.items()])
        else:
            rainfall_formatted = "No data"
        
        # Format temperature data (optimized)
        temp_formatted = "No data"
        if temperature_data:
            min_temps = [item["min"] for item in temperature_data]
            max_temps = [item["max"] for item in temperature_data]
            avg_min = sum(min_temps) / len(min_temps)
            avg_max = sum(max_temps) / len(max_temps)
            temp_formatted = f"Avg min: {avg_min:.1f}°C, Avg max: {avg_max:.1f}°C, Range: {min(min_temps):.1f}°C to {max(max_temps):.1f}°C"
        
        # Format GDD data
        gdd_formatted = "No data"
        if gdd_stats:
            gdd_formatted = f"Cumulative GDD: {gdd_stats.get('total_gdd', 'N/A')}, Avg daily GDD: {gdd_stats.get('avg_daily_gdd', 'N/A')}, Base temp: {base_temperature}°C"
        
        # OPTIMIZED: Calculate NDVI change rates with vectorized operations
        ndvi_change_rates = []
        if len(ndvi_data) > 1:
            sorted_ndvi = sorted(ndvi_data, key=lambda x: x['date'])
            
            # Pre-convert all dates to datetime objects
            dates = [datetime.strptime(item['date'], '%Y-%m-%d') for item in sorted_ndvi]
            ndvi_values = [item['ndvi'] for item in sorted_ndvi]
            
            # Calculate change rates in a single loop
            for i in range(1, len(sorted_ndvi)):
                days_diff = (dates[i] - dates[i-1]).days
                if days_diff > 0:
                    ndvi_diff = ndvi_values[i] - ndvi_values[i-1]
                    change_rate = ndvi_diff / days_diff
                    ndvi_change_rates.append({
                        'start_date': sorted_ndvi[i-1]['date'],
                        'end_date': sorted_ndvi[i]['date'],
                        'days': days_diff,
                        'change_rate': change_rate,
                        'total_change': ndvi_diff
                    })

        # Format NDVI change rate data for the prompt (optimized)
        ndvi_change_formatted = "No data"
        if ndvi_change_rates:
            # Find significant changes using list comprehension
            significant_changes = [r for r in ndvi_change_rates if abs(r['change_rate']) > 0.005]
            
            if significant_changes:
                # Sort by absolute change rate and take top 3
                significant_changes.sort(key=lambda x: abs(x['change_rate']), reverse=True)
                top_changes = significant_changes[:3]
                ndvi_change_formatted = ", ".join([
                    f"{c['start_date']} to {c['end_date']}: {c['change_rate']*100:.2f}% per day ({c['total_change']:.2f} over {c['days']} days)"
                    for c in top_changes
                ])
        
        # OPTIMIZED: Analyze NDVI patterns using vectorized operations
        tillage_replanting_detected = False
        tillage_info = "No tillage or replanting pattern detected"
        tillage_date = None
        
        if len(ndvi_data) >= 3:
            sorted_ndvi = sorted(ndvi_data, key=lambda x: x['date'])
            ndvi_values = [item['ndvi'] for item in sorted_ndvi]
            
            # Vectorized search for maximum drop
            drops = [(i, ndvi_values[i-1] - ndvi_values[i]) 
                    for i in range(1, len(ndvi_values)) 
                    if ndvi_values[i-1] - ndvi_values[i] > 0.15 and ndvi_values[i-1] > 0.3]
            
            if drops:
                # Find index with maximum drop
                drop_idx = max(drops, key=lambda x: x[1])[0]
                
                # Check for subsequent rise
                if drop_idx < len(sorted_ndvi) - 1:
                    for i in range(drop_idx + 1, len(sorted_ndvi)):
                        if sorted_ndvi[i]['ndvi'] > sorted_ndvi[drop_idx]['ndvi'] + 0.1:
                            tillage_replanting_detected = True
                            tillage_date = sorted_ndvi[drop_idx]['date']
                            tillage_info = (f"Potential tillage/replanting detected: NDVI dropped from {sorted_ndvi[drop_idx-1]['ndvi']:.2f} "
                                           f"to {sorted_ndvi[drop_idx]['ndvi']:.2f} around {sorted_ndvi[drop_idx]['date']}, "
                                           f"then rose again afterward")
                            print(f"Tillage/replanting detected on {tillage_date}")
                            break
        
        # OPTIMIZED: Check for consistently high NDVI values using vectorized operations
        consistently_high_ndvi = False
        if ndvi_data and not tillage_replanting_detected:
            ndvi_values = [item['ndvi'] for item in ndvi_data]
            consistently_high_ndvi = all(val > 0.4 for val in ndvi_values)
            if consistently_high_ndvi:
                print(f"Detected consistently high NDVI: min={min(ndvi_values):.2f}, max={max(ndvi_values):.2f}, avg={sum(ndvi_values)/len(ndvi_values):.2f}")
        
        # Smart planting date estimation (using existing optimized functions)
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
                    planting_window_text = planting_date_results["message"]
                elif planting_date_results["plantingWindowStart"] and planting_date_results["plantingWindowEnd"]:
                    start_formatted = format_date_for_display(planting_date_results["plantingWindowStart"])
                    end_formatted = format_date_for_display(planting_date_results["plantingWindowEnd"])
                    
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
        
        # Create special instructions for consistently high NDVI and tillage/replanting
        high_ndvi_instruction = ""
        if consistently_high_ndvi and not tillage_replanting_detected:
            high_ndvi_instruction = """
IMPORTANT: The NDVI data shows consistently high values (>0.4) throughout the entire analysis period. 
This indicates the crop was already well-established before the analysis period began.
DO NOT attempt to estimate a planting date. Instead, clearly state that the crop was already established
before the beginning of the analysis period.
"""
        
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
        
        planting_date_instruction = ""
        if planting_window_text:
            planting_date_instruction = f"""
IMPORTANT: Based on crop-specific emergence windows and NDVI patterns, our system has determined:
{planting_window_text}

YOU MUST USE THIS EXACT PLANTING WINDOW STATEMENT in your response. 
DO NOT modify it, rephrase it, or use different dates.
"""
        
        # OPTIMIZED: Use faster model and reduced prompt length
        # Shortened prompt for faster processing
        prompt = f"""You are Yieldera's agronomic AI. Generate concise crop commentary based on NDVI, temperature, {'rainfall, ' if irrigated == 'No' else ''}and GDD data.

{high_ndvi_instruction}
{tillage_instruction}
{planting_date_instruction}

📊 Data Summary:
- Crop: {crop} ({variety}) - {'Irrigated' if irrigated == 'Yes' else 'Rainfed'}
- Location: {latitude}, {longitude}
- NDVI: {ndvi_formatted}
- NDVI Changes: {ndvi_change_formatted}
- Tillage Analysis: {tillage_info}
{'' if irrigated == 'Yes' else f'- Rainfall: {rainfall_formatted}'}
- Temperature: {temp_formatted}
- GDD: {gdd_formatted}
- Period: {date_range}

🎯 Required Analysis:
1. NDVI Pattern (growth, stress, tillage events)
2. Temperature/GDD implications
3. {'Rainfall response' if irrigated == 'No' else 'Irrigation management'}
4. Crop status & planting date (use EXACT statement above)
5. Confidence (High/Medium/Low)

Respond in 2-3 sentences as field advisor. Include exact planting statement provided."""

        # OPTIMIZED: Use faster model with parallel processing capability
        try:
            print(f"Sending request to generate insight for field: {field_name}")
            
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",  # FASTER MODEL: Changed from gpt-4o to gpt-4o-mini (5-10x faster)
                messages=[
                    {"role": "system", "content": "You are Yieldera's agricultural advisor. Be concise and actionable."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for faster processing
                max_tokens=250    # Reduced tokens for faster response
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
            
            # OPTIMIZED: Simplified confidence boosting logic
            if confidence_level != "high" and ndvi_data and len(ndvi_data) >= 10:
                if avg_cloud_cover is not None and avg_cloud_cover < 20:
                    confidence_level = "high"
                    print(f"Boosted confidence to high: {len(ndvi_data)} observations, {avg_cloud_cover:.1f}% cloud")
            
            # Final confidence adjustments
            if consistently_high_ndvi and not tillage_replanting_detected:
                confidence_level = "high"
                print("Set confidence to high for consistently high NDVI pattern")
            
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
        
        # OPTIMIZED: Use cached collection and optimized filtering
        base_collection = get_cached_collection(coords, start, end)
        collection, collection_size = get_best_collection_optimized(base_collection)
        
        # Handle empty collection
        collection_size_val = collection_size.getInfo() if hasattr(collection_size, 'getInfo') else collection_size
        if collection_size_val == 0:
            return jsonify({
                "success": False, 
                "error": "No Sentinel-2 imagery found for the specified date range and location",
                "empty_collection": True
            }), 404
        
        # Create polygon geometry
        polygon = ee.Geometry.Polygon(coords)
        
        # Get median image and clip to polygon
        image = collection.median().clip(polygon)
        
        # Calculate NDVI
        ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
        rgb = image.select(["B4", "B3", "B2"])
        
        # Visualization settings
        ndvi_vis = ndvi.visualize(min=0, max=1, palette=["red", "yellow", "green"])
        rgb_vis = rgb.visualize(min=0, max=3000)
        
        # OPTIMIZED: Single server-side operation for all metadata
        first_image = collection.first()
        
        combined_data = ee.Dictionary({
            'ndvi_stats': ndvi.reduceRegion(
                reducer=ee.Reducer.mean().combine(ee.Reducer.minMax(), "", True),
                geometry=polygon,
                scale=10,
                maxPixels=1e9
            ),
            'image_date': first_image.date().format("YYYY-MM-dd"),
            'cloud_percentage': first_image.get("CLOUDY_PIXEL_PERCENTAGE"),
            'collection_size': collection_size_val
        })
        
        # OPTIMIZED: Parallel processing of map IDs and data fetching
        def get_map_id_safe(image_vis):
            try:
                return ee.data.getMapId({"image": image_vis})
            except Exception as e:
                print(f"Error getting map ID: {e}")
                return None
        
        # Get map IDs with timeout handling
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_ndvi = executor.submit(get_map_id_safe, ndvi_vis)
            future_rgb = executor.submit(get_map_id_safe, rgb_vis)
            
            try:
                # Get all data in single .getInfo() call
                all_data = combined_data.getInfo()
                
                map_id_ndvi = future_ndvi.result(timeout=10)
                map_id_rgb = future_rgb.result(timeout=10)
                
                ndvi_stats = all_data['ndvi_stats']
                
                response = {
                    "success": True,
                    "mean_ndvi": ndvi_stats.get("NDVI_mean"),
                    "min_ndvi": ndvi_stats.get("NDVI_min"),
                    "max_ndvi": ndvi_stats.get("NDVI_max"),
                    "image_date": all_data['image_date'],
                    "collection_size": all_data['collection_size'],
                    "cloudy_pixel_percentage": all_data['cloud_percentage']
                }
                
                # Add tile URLs if successful
                if map_id_ndvi:
                    response["ndvi_tile_url"] = map_id_ndvi["tile_fetcher"].url_format
                if map_id_rgb:
                    response["rgb_tile_url"] = map_id_rgb["tile_fetcher"].url_format
                
                print(f"Successfully processed NDVI request. Mean NDVI: {ndvi_stats.get('NDVI_mean')}, Cloud cover: {all_data['cloud_percentage']}%")
                return jsonify(response)
                
            except concurrent.futures.TimeoutError:
                print("Timeout getting map IDs, returning statistics only")
                all_data = combined_data.getInfo()
                ndvi_stats = all_data['ndvi_stats']
                
                return jsonify({
                    "success": True,
                    "mean_ndvi": ndvi_stats.get("NDVI_mean"),
                    "min_ndvi": ndvi_stats.get("NDVI_min"),
                    "max_ndvi": ndvi_stats.get("NDVI_max"),
                    "image_date": all_data['image_date'],
                    "collection_size": all_data['collection_size'],
                    "cloudy_pixel_percentage": all_data['cloud_percentage'],
                    "visualization_timeout": True
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
        
        # OPTIMIZED: Use cached collection and smart filtering
        base_collection = get_cached_collection(coords, start, end)
        collection, collection_size = get_best_collection_optimized(base_collection)
        
        # Handle empty collection
        collection_size_val = collection_size.getInfo() if hasattr(collection_size, 'getInfo') else collection_size
        if collection_size_val == 0:
            return jsonify({
                "success": False, 
                "error": "No Sentinel-2 imagery found for the specified date range and location",
                "empty_collection": True
            }), 404
        
        # Create polygon geometry
        polygon = ee.Geometry.Polygon(coords)
        
        # OPTIMIZED: Single server-side batch processing for all images
        def process_image_batch(image):
            """Optimized batch processing of images on server-side"""
            clipped = image.clip(polygon)
            ndvi = clipped.normalizedDifference(["B8", "B4"])
            
            # Calculate mean NDVI for the polygon with optimized reducer
            ndvi_mean = ndvi.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=polygon,
                scale=20,  # Increased scale for faster processing
                maxPixels=1e8,  # Reduced maxPixels for faster processing
                bestEffort=True  # Allow best effort processing
            ).get('nd')
            
            return ee.Feature(None, {
                'date': image.date().format('YYYY-MM-dd'),
                'ndvi': ndvi_mean,
                'cloud_percentage': image.get('CLOUDY_PIXEL_PERCENTAGE')
            })
        
        # OPTIMIZED: Process entire collection in single server-side operation
        processed_collection = collection.map(process_image_batch)
        feature_collection = ee.FeatureCollection(processed_collection)
        
        # Single .getInfo() call to get all time series data
        features_data = feature_collection.getInfo()
        features = features_data['features']
        
        # OPTIMIZED: Extract NDVI time series data with list comprehension
        ndvi_time_series = [
            {
                "date": feature['properties']['date'],
                "ndvi": feature['properties']['ndvi'],
                "cloud_percentage": feature['properties']['cloud_percentage']
            }
            for feature in features 
            if feature['properties'].get('ndvi') is not None
        ]
        
        # Verify we have sufficient data points
        if len(ndvi_time_series) == 0:
            return jsonify({
                "success": False, 
                "error": "No valid NDVI readings could be calculated for this field",
                "empty_time_series": True
            }), 404
        
        # Sort time series by date (optimized)
        ndvi_time_series.sort(key=lambda x: x["date"])
        
        # Return response with time series data
        response = {
            "success": True,
            "time_series": ndvi_time_series,
            "collection_size": collection_size_val
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
