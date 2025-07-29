import os
import json
import ee
import traceback
import hashlib
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_compress import Compress
from openai import OpenAI
from dotenv import load_dotenv
from cachetools import TTLCache
import threading

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Enable GZIP compression
Compress(app)

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# In-memory caching with TTL (Time To Live)
cache = TTLCache(maxsize=1000, ttl=3600)  # Cache for 1 hour, max 1000 items
cache_lock = threading.Lock()

# Define crop-specific emergence windows (in days)
EMERGENCE_WINDOWS = {
    "Maize": (6, 10),
    "Soyabeans": (7, 11),
    "Sorghum": (6, 10),
    "Cotton": (5, 9),
    "Groundnuts": (6, 10),
    "Barley": (7, 11),
    "Wheat": (9, 13),
    "Millet": (4, 8),
    "Tobacco": (7, 11)  # For nursery emergence
}

# Constants for emergence detection
EMERGENCE_THRESHOLD = 0.2
DEFAULT_EMERGENCE_WINDOW = (5, 10)  # Default for unknown crops
SIGNIFICANT_RAINFALL = 10  # mm, threshold for significant rainfall

# Global variable to track GEE initialization
gee_initialization_time = None
gee_initialized = False

def initialize_gee_at_startup():
    """Initialize Google Earth Engine once at server startup"""
    global gee_initialization_time, gee_initialized
    
    if gee_initialized:
        return True, "GEE already initialized"
    
    try:
        print("Initializing Google Earth Engine at startup...")
        start_time = datetime.now()
        
        # Load service account info from environment variable
        service_account_info = json.loads(os.environ["GEE_CREDENTIALS"])
        
        # Initialize Earth Engine using service account credentials
        credentials = ee.ServiceAccountCredentials(
            email=service_account_info["client_email"],
            key_data=json.dumps(service_account_info)
        )
        ee.Initialize(credentials)
        
        # Test the connection - FIXED: Use ImageCollection, not Image
        test_collection = ee.ImageCollection("COPERNICUS/S2_HARMONIZED").first()
        test_info = test_collection.getInfo()
        
        gee_initialization_time = datetime.now()
        init_duration = (gee_initialization_time - start_time).total_seconds()
        gee_initialized = True
        print(f"GEE initialized successfully at startup in {init_duration:.2f} seconds")
        
        return True, f"GEE initialized in {init_duration:.2f}s"
        
    except Exception as e:
        print(f"GEE initialization error at startup: {str(e)}")
        gee_initialized = False
        return False, f"GEE initialization failed: {str(e)}"

def get_cache_key(coords, start_date, end_date, endpoint_type):
    """Generate a cache key for the given parameters"""
    coords_str = json.dumps(coords, sort_keys=True)
    key_string = f"{endpoint_type}_{coords_str}_{start_date}_{end_date}"
    return hashlib.md5(key_string.encode()).hexdigest()

def calculate_field_cloud_cover(image, polygon):
    """
    Calculate cloud cover percentage specifically for the user's field polygon
    Uses available Sentinel-2 cloud detection bands
    """
    try:
        # Clip image to the field boundary
        clipped = image.clip(polygon)
        
        # Get available band names to determine which method to use
        band_names = image.bandNames().getInfo()
        print(f"Available bands for cloud detection: {band_names}")
        
        # Method 1: Try MSK_CLASSI_OPAQUE and MSK_CLASSI_CIRRUS (most common in S2_HARMONIZED)
        if 'MSK_CLASSI_OPAQUE' in band_names and 'MSK_CLASSI_CIRRUS' in band_names:
            print("Using MSK_CLASSI cloud masks for field-specific calculation")
            
            # Get cloud masks
            opaque_clouds = clipped.select('MSK_CLASSI_OPAQUE')
            cirrus_clouds = clipped.select('MSK_CLASSI_CIRRUS')
            
            # Combine cloud masks (both are binary: 1 = cloud, 0 = clear)
            combined_cloud_mask = opaque_clouds.Or(cirrus_clouds)
            
            # Calculate total pixels
            total_pixels = combined_cloud_mask.reduceRegion(
                reducer=ee.Reducer.count(),
                geometry=polygon,
                scale=20,
                maxPixels=1e9
            ).get('MSK_CLASSI_OPAQUE')
            
            # Calculate cloudy pixels
            cloudy_pixels = combined_cloud_mask.reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=polygon,
                scale=20,
                maxPixels=1e9
            ).get('MSK_CLASSI_OPAQUE')
            
            # Calculate percentage
            field_cloud_percentage = ee.Algorithms.If(
                ee.Number(total_pixels).gt(0),
                ee.Number(cloudy_pixels).divide(ee.Number(total_pixels)).multiply(100),
                0
            )
            
            return field_cloud_percentage
            
        # Method 2: Try SCL band (if available)
        elif 'SCL' in band_names:
            print("Using SCL band for field-specific calculation")
            
            scl = clipped.select('SCL')
            
            # SCL classification values:
            # 8 = CLOUD_MEDIUM_PROBABILITY, 9 = CLOUD_HIGH_PROBABILITY, 3 = CLOUD_SHADOWS, 10 = THIN_CIRRUS
            cloud_mask = scl.eq(8).Or(scl.eq(9)).Or(scl.eq(3)).Or(scl.eq(10))
            
            # Calculate total valid pixels
            total_pixels = scl.gte(0).reduceRegion(
                reducer=ee.Reducer.count(),
                geometry=polygon,
                scale=20,
                maxPixels=1e9
            ).get('SCL')
            
            # Calculate cloudy pixels
            cloudy_pixels = cloud_mask.reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=polygon,
                scale=20,
                maxPixels=1e9
            ).get('SCL')
            
            # Calculate percentage
            field_cloud_percentage = ee.Algorithms.If(
                ee.Number(total_pixels).gt(0),
                ee.Number(cloudy_pixels).divide(ee.Number(total_pixels)).multiply(100),
                0
            )
            
            return field_cloud_percentage
            
        # Method 3: Fallback to QA60 band
        elif 'QA60' in band_names:
            print("Using QA60 band for field-specific calculation")
            
            qa60 = clipped.select('QA60')
            
            # QA60 bit flags: Bit 10 = Cirrus, Bit 11 = Clouds
            cloud_bit_10 = qa60.bitwiseAnd(1 << 10).gt(0)  # Cirrus
            cloud_bit_11 = qa60.bitwiseAnd(1 << 11).gt(0)  # Clouds
            
            # Combine cloud masks
            cloud_mask = cloud_bit_10.Or(cloud_bit_11)
            
            # Calculate total pixels
            total_pixels = qa60.gte(0).reduceRegion(
                reducer=ee.Reducer.count(),
                geometry=polygon,
                scale=20,
                maxPixels=1e9
            ).get('QA60')
            
            # Calculate cloudy pixels
            cloudy_pixels = cloud_mask.reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=polygon,
                scale=20,
                maxPixels=1e9
            ).get('QA60')
            
            # Calculate percentage
            field_cloud_percentage = ee.Algorithms.If(
                ee.Number(total_pixels).gt(0),
                ee.Number(cloudy_pixels).divide(ee.Number(total_pixels)).multiply(100),
                0
            )
            
            return field_cloud_percentage
            
        else:
            print(f"No suitable cloud detection bands found in: {band_names}")
            return None
            
    except Exception as e:
        print(f"Error calculating field cloud cover: {e}")
        return None

def get_optimized_collection(polygon, start_date, end_date, limit_images=True):
    """Get optimized Sentinel-2 collection with smart cloud filtering and pre-sorting"""
    
    # Start with base collection
    base_collection = (
        ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
        .filterBounds(polygon)
        .filterDate(start_date, end_date)
    )
    
    # Check collection size first
    total_size = base_collection.size().getInfo()
    print(f"Total available images: {total_size}")
    
    if total_size == 0:
        return None, 0
    
    # Smart cloud filtering with progressive thresholds
    if total_size > 50:
        cloud_threshold = 10
        max_images = 15
    elif total_size > 20:
        cloud_threshold = 20
        max_images = 20
    elif total_size > 10:
        cloud_threshold = 30
        max_images = 25
    else:
        cloud_threshold = 80
        max_images = total_size
    
    # Apply cloud filtering and pre-sort by cloud percentage
    collection = (
        base_collection
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_threshold))
        .sort("CLOUDY_PIXEL_PERCENTAGE")  # Best images first
    )
    
    # Apply smart limiting if requested
    if limit_images:
        collection = collection.limit(max_images)
    
    collection_size = collection.size().getInfo()
    print(f"Filtered collection size: {collection_size} (cloud < {cloud_threshold}%)")
    
    # Fallback if no images after filtering
    if collection_size == 0:
        print("No images found with initial cloud threshold, trying fallback...")
        for fallback_threshold in [50, 80]:
            collection = (
                base_collection
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", fallback_threshold))
                .sort("CLOUDY_PIXEL_PERCENTAGE")
                .limit(max_images)
            )
            collection_size = collection.size().getInfo()
            if collection_size > 0:
                print(f"Fallback successful: {collection_size} images with cloud < {fallback_threshold}%")
                break
    
    return collection, collection_size

@app.route("/")
def index():
    return "NDVI & RGB backend is live!"

@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    try:
        if not gee_initialized:
            return jsonify({
                "success": False, 
                "message": "GEE not initialized",
                "timestamp": datetime.now().isoformat(),
                "gee_initialized": False
            }), 500
        
        return jsonify({
            "success": True, 
            "message": f"Backend is healthy. GEE initialized at startup.",
            "timestamp": datetime.now().isoformat(),
            "gee_initialized": True,
            "gee_init_time": gee_initialization_time.isoformat() if gee_initialization_time else None,
            "cache_size": len(cache)
        })
        
    except Exception as e:
        return jsonify({
            "success": False, 
            "message": f"Health check failed: {str(e)}",
            "timestamp": datetime.now().isoformat(),
            "gee_initialized": gee_initialized
        }), 500

@app.route("/api/ping", methods=["GET"])
def ping():
    """Simple ping endpoint for basic connectivity"""
    return jsonify({
        "success": True,
        "message": "Pong",
        "timestamp": datetime.now().isoformat(),
        "cache_size": len(cache)
    })

@app.route("/api/warmup", methods=["POST"])
def warmup():
    """Dedicated warmup endpoint"""
    try:
        if not gee_initialized:
            return jsonify({
                "success": False,
                "message": "GEE not initialized at startup",
                "timestamp": datetime.now().isoformat()
            }), 500
        
        # Test a simple Sentinel-2 operation to warm up
        print("Warming up with test Sentinel-2 query...")
        start_time = datetime.now()
        
        # Simple test query
        test_collection = (
            ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
            .filterDate("2023-01-01", "2023-01-02")
            .first()
        )
        
        # Force execution to warm up the pipeline
        test_info = test_collection.getInfo()
        
        warmup_duration = (datetime.now() - start_time).total_seconds()
        print(f"Warmup completed in {warmup_duration:.2f} seconds")
        
        return jsonify({
            "success": True,
            "message": f"Backend warmed up successfully.",
            "warmup_duration_seconds": warmup_duration,
            "timestamp": datetime.now().isoformat(),
            "gee_initialized": True,
            "cache_size": len(cache)
        })
        
    except Exception as e:
        print(f"Warmup error: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Warmup failed: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500

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
    
    # FIXED: If still no emergence detected, this is clearly "no planting detected"
    if not emergence_date:
        if irrigated == "No" and rainfall_data:
            rainfall_failure = detect_rainfall_without_emergence(ndvi_data, rainfall_data)
            if rainfall_failure and rainfall_failure['detected']:
                return {
                    "emergenceDate": None,
                    "plantingWindowStart": None,
                    "plantingWindowEnd": None,
                    "preEstablished": False,
                    "confidence": "medium",
                    "message": rainfall_failure['message'],
                    "rainfall_without_emergence": True,
                    "rainfall_date": rainfall_failure['rainfall_date'],
                    "primary_emergence": False,
                    "no_planting_detected": True
                }
        
        # FIXED: Clear "no planting detected" case with high confidence
        if sorted_ndvi:
            start_date = format_date_for_display(sorted_ndvi[0]['date'])
            end_date = format_date_for_display(sorted_ndvi[-1]['date'])
            message = f"No planting activity detected from {start_date} to {end_date}."
        else:
            message = "No planting activity detected during the analysis period."
            
        return {
            "emergenceDate": None,
            "plantingWindowStart": None,
            "plantingWindowEnd": None,
            "preEstablished": False,
            "confidence": "high",  # High confidence that no planting occurred
            "message": message,
            "primary_emergence": False,
            "no_planting_detected": True
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
        if not gee_initialized:
            return jsonify({
                "success": False,
                "error": "GEE not initialized"
            }), 500
        
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
        
        # Add tillage information if detected (only for planted fields)
        if not primary_results.get("no_planting_detected") and tillage_results["tillage_detected"]:
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
        
        # FIXED: Create appropriate prompts based on planting detection
        if primary_results.get("no_planting_detected"):
            # For no planting detected - be very direct and clear
            prompt = f"""Field Analysis Summary:
Field: {field_name} - {crop} ({'Irrigated' if irrigated == 'Yes' else 'Rainfed'})
Analysis Period: {date_range}
Finding: {planting_window_text}

Instructions: Write a clear, professional response (2-3 sentences maximum) that:
1. States definitively that NO PLANTING was detected during this period
2. Provides ONE specific actionable recommendation for this unplanted field
3. Uses simple, direct language - no uncertainty or NDVI technical details

Example format: "No planting activity was detected during this period. [Specific recommendation for the unplanted field]."
"""
        else:
            # For normal planting detection
            prompt = f"""Field Analysis Summary:
Field: {field_name} - {crop} ({'Irrigated' if irrigated == 'Yes' else 'Rainfed'})
Analysis Period: {date_range}
Planting Status: {planting_window_text}
NDVI Pattern: {ndvi_formatted[:200]}...
{'' if irrigated == 'Yes' else f'Rainfall: {rainfall_formatted[:100]}...'}

Instructions: Write a clear, professional assessment (2-3 sentences maximum) that:
1. Confirms the planting timeline found
2. Adds ONE specific farming recommendation
3. Uses simple, professional language - no technical jargon
4. Be definitive, not uncertain

Keep it simple and actionable for farmers."""

        # Call OpenAI API
        try:
            print(f"Sending request to generate insight for field: {field_name}")
            
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a farm advisor. Give clear, simple advice in 2-3 sentences. No jargon, no formatting, just plain professional language."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Very low for consistent, simple responses
                max_tokens=150    # Short, focused responses
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
                    "primary_emergence": primary_results.get("primary_emergence", False),
                    "no_planting_detected": primary_results.get("no_planting_detected", False)
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
        if not gee_initialized:
            return jsonify({
                "success": False, 
                "error": "GEE not initialized",
                "details": "Please restart the server or contact support."
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
        
        # Check cache first
        cache_key = get_cache_key(coords, start, end, "ndvi_tiles")
        with cache_lock:
            if cache_key in cache:
                print(f"Cache hit for NDVI tiles request")
                return jsonify(cache[cache_key])
        
        # Log incoming request
        print(f"Processing NDVI tiles request: start={start}, end={end}, coords length={len(coords)}")
        
        # Create Earth Engine geometry
        polygon = ee.Geometry.Polygon(coords)

        # Get optimized collection
        collection, collection_size = get_optimized_collection(polygon, start, end, limit_images=True)
        
        if collection is None or collection_size == 0:
            return jsonify({
                "success": False, 
                "error": "No Sentinel-2 imagery found for the specified date range and location",
                "empty_collection": True
            }), 404
        
        # Use mosaic instead of median for better performance
        image = collection.mosaic().clip(polygon)
        first_image = collection.first()
        
        # Calculate NDVI
        ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
        rgb = image.select(["B4", "B3", "B2"])
        
        # Visualization settings
        ndvi_vis = ndvi.visualize(min=-0.5, max=1, palette=["blue", "red", "yellow", "green"])
        rgb_vis = rgb.visualize(min=0, max=3000)
        
        # Calculate scene-level cloud cover
        scene_cloud_percentage = first_image.get("CLOUDY_PIXEL_PERCENTAGE")
        
        # Optimize metadata extraction - single operation (without field cloud cover)
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
            'scene_cloud_percentage': scene_cloud_percentage
        })
        
        # Calculate field-specific cloud cover separately to handle errors gracefully
        field_cloud_percentage = None
        try:
            field_cloud_calculation = calculate_field_cloud_cover(first_image, polygon)
            if field_cloud_calculation is not None:
                field_cloud_percentage = field_cloud_calculation.getInfo()
                print(f"Field-specific cloud cover calculated: {field_cloud_percentage}%")
        except Exception as e:
            print(f"Field cloud cover calculation failed, using scene-level: {e}")
            field_cloud_percentage = None
        
        # Get map IDs for tile URLs with timeout handling
        try:
            map_id_ndvi = ee.data.getMapId({"image": ndvi_vis})
            map_id_rgb = ee.data.getMapId({"image": rgb_vis})
            
            # Single .getInfo() call to get all data
            all_data = combined_data.getInfo()
            ndvi_stats = all_data['ndvi_stats']
            image_date = all_data['image_date']
            scene_cloud_percentage = all_data['scene_cloud_percentage']
            
            # Use field-specific cloud cover if available, otherwise fall back to scene-level
            display_cloud_percentage = field_cloud_percentage if field_cloud_percentage is not None else scene_cloud_percentage
            
            # Prepare response
            response = {
                "success": True,
                "ndvi_tile_url": map_id_ndvi["tile_fetcher"].url_format,
                "rgb_tile_url": map_id_rgb["tile_fetcher"].url_format,
                "mean_ndvi": ndvi_stats.get("NDVI_mean"),
                "min_ndvi": ndvi_stats.get("NDVI_min"),
                "max_ndvi": ndvi_stats.get("NDVI_max"),
                "image_date": image_date,
                "collection_size": collection_size,
                "cloudy_pixel_percentage": display_cloud_percentage,  # Primary display value
                "scene_cloud_percentage": scene_cloud_percentage,    # Scene-level for reference
                "field_cloud_percentage": field_cloud_percentage,    # Field-specific calculation
                "cloud_calculation_method": "field_specific" if field_cloud_percentage is not None else "scene_level"
            }
            
            # Cache the response
            with cache_lock:
                cache[cache_key] = response
            
            print(f"Successfully processed NDVI tiles request. Mean NDVI: {ndvi_stats.get('NDVI_mean')}")
            print(f"Scene cloud cover: {scene_cloud_percentage}%, Field cloud cover: {field_cloud_percentage}%")
            return jsonify(response)
            
        except Exception as e:
            print(f"Error getting map IDs: {e}")
            # Still return statistics even if visualization fails
            all_data = combined_data.getInfo()
            ndvi_stats = all_data['ndvi_stats']
            image_date = all_data['image_date']
            scene_cloud_percentage = all_data['scene_cloud_percentage']
            
            display_cloud_percentage = field_cloud_percentage if field_cloud_percentage is not None else scene_cloud_percentage
            
            response = {
                "success": True,
                "mean_ndvi": ndvi_stats.get("NDVI_mean"),
                "min_ndvi": ndvi_stats.get("NDVI_min"),
                "max_ndvi": ndvi_stats.get("NDVI_max"),
                "image_date": image_date,
                "collection_size": collection_size,
                "cloudy_pixel_percentage": display_cloud_percentage,
                "scene_cloud_percentage": scene_cloud_percentage,
                "field_cloud_percentage": field_cloud_percentage,
                "cloud_calculation_method": "field_specific" if field_cloud_percentage is not None else "scene_level",
                "visualization_error": str(e)
            }
            
            # Cache the response
            with cache_lock:
                cache[cache_key] = response
                
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
        if not gee_initialized:
            return jsonify({
                "success": False, 
                "error": "GEE not initialized",
                "details": "Please restart the server or contact support."
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
        
        # Check cache first
        cache_key = get_cache_key(coords, start, end, "ndvi_timeseries")
        with cache_lock:
            if cache_key in cache:
                print(f"Cache hit for NDVI timeseries request")
                return jsonify(cache[cache_key])
        
        # Log incoming request
        print(f"Processing NDVI time series request: start={start}, end={end}, coords length={len(coords[0])}")
        
        # Create Earth Engine geometry
        polygon = ee.Geometry.Polygon(coords)

        # Get optimized collection (don't limit for time series)
        collection, collection_size = get_optimized_collection(polygon, start, end, limit_images=False)
        
        if collection is None or collection_size == 0:
            return jsonify({
                "success": False, 
                "error": "No Sentinel-2 imagery found for the specified date range and location",
                "empty_collection": True
            }), 404
        
        # OPTIMIZED: Use reduceRegions for batch processing instead of map
        def add_ndvi_and_cloud_stats(image):
            """Add NDVI statistics to image properties (field cloud cover calculated separately)"""
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
            
            # Get scene-level cloud cover
            scene_cloud_cover = image.get("CLOUDY_PIXEL_PERCENTAGE")
            
            # Return image with NDVI mean and scene cloud cover
            return image.set({
                'ndvi_mean': ndvi_mean,
                'date_formatted': image.date().format('YYYY-MM-dd'),
                'scene_cloud_percentage': scene_cloud_cover
            })
        
        # Map the function over the collection to add NDVI and cloud stats
        collection_with_stats = collection.map(add_ndvi_and_cloud_stats)
        
        # OPTIMIZED: Use aggregate_array to get all data in fewer server calls
        dates_array = collection_with_stats.aggregate_array('date_formatted')
        ndvi_array = collection_with_stats.aggregate_array('ndvi_mean')
        scene_cloud_array = collection_with_stats.aggregate_array('scene_cloud_percentage')
        
        # Single .getInfo() call to get all arrays
        print("Getting time series data in batch...")
        batch_data = ee.Dictionary({
            'dates': dates_array,
            'ndvi_values': ndvi_array,
            'scene_cloud_percentages': scene_cloud_array
        }).getInfo()
        
        dates = batch_data['dates']
        ndvi_values = batch_data['ndvi_values']
        scene_cloud_percentages = batch_data['scene_cloud_percentages']
        
        # Combine into time series data with simplified cloud cover
        ndvi_time_series = []
        
        for i in range(len(dates)):
            ndvi_value = ndvi_values[i]
            
            # Only add valid NDVI readings
            if ndvi_value is not None:
                # For time series, use scene-level cloud cover for performance
                # Field-specific calculation would be too slow for many images
                ndvi_time_series.append({
                    "date": dates[i],
                    "ndvi": ndvi_value,
                    "cloud_percentage": scene_cloud_percentages[i],  # Scene-level for performance
                    "scene_cloud_percentage": scene_cloud_percentages[i],
                    "field_cloud_percentage": None,  # Not calculated for time series (performance)
                    "cloud_calculation_method": "scene_level"  # Always scene-level for time series
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
        
        # Prepare response
        response = {
            "success": True,
            "time_series": ndvi_time_series,
            "collection_size": collection_size
        }
        
        # Cache the response
        with cache_lock:
            cache[cache_key] = response
        
        print(f"Successfully processed NDVI time series request. {len(ndvi_time_series)} data points returned.")
        print(f"Using scene-level cloud cover for all time series data points for performance.")
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

# Initialize GEE at startup
print("Starting backend initialization...")
success, init_message = initialize_gee_at_startup()
if success:
    print(f"✓ Backend ready: {init_message}")
else:
    print(f"✗ Backend startup failed: {init_message}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
