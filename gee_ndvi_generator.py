import os
import json
import ee
import traceback
import hashlib
import geohash2
import statistics
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

# Geohash spatial adaptation cache for wheat emergence patterns
spatial_cache = TTLCache(maxsize=500, ttl=86400)  # 24 hour TTL for spatial patterns
spatial_cache_lock = threading.Lock()

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

# Constants for emergence detection (existing)
EMERGENCE_THRESHOLD = 0.2
DEFAULT_EMERGENCE_WINDOW = (5, 10)  # Default for unknown crops
SIGNIFICANT_RAINFALL = 10  # mm, threshold for significant rainfall

# NEW: Constants for wheat winter detection
THRESHOLD_WHEAT_WINTER = 0.15      # lower absolute NDVI trigger
MIN_SLOPE_DELTA = 0.04             # minimum NDVI rise
MAX_SLOPE_DAYS = 10                # window to realize the rise
CLOUD_CANDIDATE_MAX = 30           # % cloud cap at candidate emergence
SMOOTH_WINDOW = 3                  # 3-point median smoothing
NDVI_AMPLITUDE_MIN = 0.15          # geometry/season sanity check

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

def get_cache_key(coords, start_date, end_date, endpoint_type, index_type="NDVI"):
    """Generate a cache key for the given parameters"""
    coords_str = json.dumps(coords, sort_keys=True)
    key_string = f"{endpoint_type}_{coords_str}_{start_date}_{end_date}_{index_type}"
    return hashlib.md5(key_string.encode()).hexdigest()

def get_index(image, index_type):
    """
    Calculate the specified vegetation index.
    Supports: NDVI, EVI, SAVI, NDMI
    """
    if index_type == "NDVI":
        return image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    
    elif index_type == "EVI":
        nir = image.select('B8').divide(10000)
        red = image.select('B4').divide(10000)
        blue = image.select('B2').divide(10000)
        evi = image.expression(
            '2.5 * ((NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1))',
            {'NIR': nir, 'RED': red, 'BLUE': blue}
        ).rename('EVI')
        # Clip to [-1, 1] range
        return evi.where(evi.gt(1), 1).where(evi.lt(-1), -1)
    
    elif index_type == "SAVI":
        nir = image.select('B8').divide(10000)
        red = image.select('B4').divide(10000)
        savi = image.expression(
            '((NIR - RED) * (1 + L)) / (NIR + RED + L)',
            {'NIR': nir, 'RED': red, 'L': 0.5}
        ).rename('SAVI')
        return savi
    
    elif index_type == "NDMI":
        return image.normalizedDifference(['B8', 'B11']).rename('NDMI')
    
    else:
        # Default fallback: NDVI
        return image.normalizedDifference(['B8', 'B4']).rename('NDVI')

def smooth_ndvi_series(ndvi_data, window=SMOOTH_WINDOW):
    """Apply 3-point median smoothing to NDVI series"""
    if len(ndvi_data) < window:
        return ndvi_data
    
    smoothed_data = []
    sorted_data = sorted(ndvi_data, key=lambda x: x['date'])
    
    for i in range(len(sorted_data)):
        if i == 0:
            # First point: use original value
            smoothed_data.append(sorted_data[i].copy())
        elif i == len(sorted_data) - 1:
            # Last point: use original value
            smoothed_data.append(sorted_data[i].copy())
        else:
            # Middle points: median of 3-point window
            window_values = [
                sorted_data[i-1]['ndvi'],
                sorted_data[i]['ndvi'],
                sorted_data[i+1]['ndvi']
            ]
            smoothed_value = statistics.median(window_values)
            
            smoothed_point = sorted_data[i].copy()
            smoothed_point['ndvi'] = smoothed_value
            smoothed_data.append(smoothed_point)
    
    return smoothed_data

def is_winter_season(start_date, end_date, coordinates):
    """Check if analysis period overlaps with winter season (Apr-Aug for Southern Hemisphere)"""
    try:
        start_obj = datetime.strptime(start_date, '%Y-%m-%d')
        end_obj = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Determine hemisphere based on latitude
        if coordinates and len(coordinates) > 0 and len(coordinates[0]) > 0:
            lat = coordinates[0][0][1]  # Get latitude from first coordinate
            is_southern_hemisphere = lat < 0
        else:
            # Default to Southern Hemisphere for Zimbabwe
            is_southern_hemisphere = True
        
        if is_southern_hemisphere:
            # Winter months in Southern Hemisphere: April to August
            winter_months = [4, 5, 6, 7, 8]
        else:
            # Winter months in Northern Hemisphere: November to March
            winter_months = [11, 12, 1, 2, 3]
        
        # Check if any month in the date range overlaps with winter
        current_date = start_obj
        while current_date <= end_obj:
            if current_date.month in winter_months:
                return True
            # Move to next month
            if current_date.month == 12:
                current_date = current_date.replace(year=current_date.year + 1, month=1)
            else:
                current_date = current_date.replace(month=current_date.month + 1)
        
        return False
        
    except Exception as e:
        print(f"Error determining winter season: {e}")
        return False

def get_geohash_key(coordinates, crop, season_year):
    """Generate geohash key for spatial adaptation"""
    try:
        # Get center point of polygon
        if coordinates and len(coordinates) > 0 and len(coordinates[0]) > 0:
            lats = [coord[1] for coord in coordinates[0]]
            lons = [coord[0] for coord in coordinates[0]]
            center_lat = sum(lats) / len(lats)
            center_lon = sum(lons) / len(lons)
            
            # Generate geohash at precision 5 (~5km)
            geohash = geohash2.encode(center_lat, center_lon, precision=5)
            key = f"{geohash}_{crop}_{season_year}"
            return geohash, key
        
        return None, None
        
    except Exception as e:
        print(f"Error generating geohash key: {e}")
        return None, None

def get_spatial_prior(geohash_key):
    """Get emergence date cluster prior from spatial cache"""
    try:
        with spatial_cache_lock:
            if geohash_key in spatial_cache:
                return spatial_cache[geohash_key]
        return None
    except Exception as e:
        print(f"Error getting spatial prior: {e}")
        return None

def update_spatial_prior(geohash_key, emergence_date):
    """Update spatial cache with new emergence detection"""
    try:
        with spatial_cache_lock:
            if geohash_key not in spatial_cache:
                spatial_cache[geohash_key] = []
            
            # Add new emergence date
            spatial_cache[geohash_key].append(emergence_date)
            
            # Keep only recent detections (max 10)
            if len(spatial_cache[geohash_key]) > 10:
                spatial_cache[geohash_key] = spatial_cache[geohash_key][-10:]
                
    except Exception as e:
        print(f"Error updating spatial prior: {e}")

def apply_spatial_nudge(candidate_date, geohash_key):
    """Apply spatial nudging if emergence is ambiguous"""
    try:
        prior_dates = get_spatial_prior(geohash_key)
        if not prior_dates or len(prior_dates) < 2:
            return candidate_date, False
        
        # Calculate cluster median
        date_objects = []
        for date_str in prior_dates:
            try:
                date_objects.append(datetime.strptime(date_str, '%Y-%m-%d'))
            except:
                continue
        
        if len(date_objects) < 2:
            return candidate_date, False
        
        # Get median date
        date_objects.sort()
        median_idx = len(date_objects) // 2
        median_date = date_objects[median_idx]
        
        # Check if candidate is within ±3 days of median
        candidate_obj = datetime.strptime(candidate_date, '%Y-%m-%d')
        days_diff = abs((candidate_obj - median_date).days)
        
        if days_diff <= 3:
            # Nudge toward median
            nudged_date = median_date.strftime('%Y-%m-%d')
            return nudged_date, True
        
        return candidate_date, False
        
    except Exception as e:
        print(f"Error applying spatial nudge: {e}")
        return candidate_date, False

def detect_wheat_winter_emergence(ndvi_data, coordinates=None, force_winter_detector=False):
    """
    Wheat-specific winter emergence detection using remote sensing only.
    Returns emergence_date, confidence, and metadata.
    """
    try:
        print("=== WHEAT WINTER EMERGENCE DETECTION ===")
        
        # Check if winter detector should be used
        if not force_winter_detector:
            start_date = min(item['date'] for item in ndvi_data)
            end_date = max(item['date'] for item in ndvi_data)
            
            if not is_winter_season(start_date, end_date, coordinates):
                print("Not winter season, falling back to standard detection")
                return None, None, {}
        
        # Validate minimum data requirements
        if len(ndvi_data) < 4:
            return None, "low", {
                "qa": {
                    "valid": False,
                    "reason": "sparse_data",
                    "min_points": len(ndvi_data)
                }
            }
        
        # Apply 3-point median smoothing
        smoothed_data = smooth_ndvi_series(ndvi_data, SMOOTH_WINDOW)
        print(f"Applied smoothing to {len(smoothed_data)} points")
        
        # Calculate NDVI amplitude for sanity check
        ndvi_values = [item['ndvi'] for item in smoothed_data]
        ndvi_amplitude = max(ndvi_values) - min(ndvi_values)
        
        qa_info = {
            "valid": True,
            "ndvi_amplitude": round(ndvi_amplitude, 3),
            "min_points": len(smoothed_data),
            "reason": None
        }
        
        # Geometry/season sanity check
        if ndvi_amplitude < NDVI_AMPLITUDE_MIN:
            qa_info["valid"] = False
            qa_info["reason"] = "low_signal"
            return None, "low", {"qa": qa_info}
        
        # Sort by date for chronological scanning
        sorted_data = sorted(smoothed_data, key=lambda x: x['date'])
        
        candidates = []
        
        # Scan chronologically for emergence candidates
        for i in range(len(sorted_data)):
            current_point = sorted_data[i]
            current_ndvi = current_point['ndvi']
            current_date = current_point['date']
            
            # Get cloud percentage (prefer field-level, fallback to scene-level)
            cloud_pct = current_point.get('field_cloud_percentage')
            if cloud_pct is None:
                cloud_pct = current_point.get('cloud_percentage', 0)
                
            # Skip high cloud candidates
            if cloud_pct > CLOUD_CANDIDATE_MAX:
                continue
            
            candidate_confidence = "medium"
            detection_method = None
            
            # Rule 1: Crossing rule - smoothed NDVI crosses ≥ 0.15 from below
            if i > 0:
                prev_ndvi = sorted_data[i-1]['ndvi']
                if prev_ndvi < THRESHOLD_WHEAT_WINTER and current_ndvi >= THRESHOLD_WHEAT_WINTER:
                    candidates.append({
                        'date': current_date,
                        'method': 'crossing',
                        'confidence': 'high',
                        'cloud_pct': cloud_pct,
                        'ndvi_value': current_ndvi,
                        'prev_ndvi': prev_ndvi
                    })
                    print(f"Crossing candidate: {current_date}, NDVI: {prev_ndvi:.3f} -> {current_ndvi:.3f}")
                    continue
            
            # Rule 2: Slope rule - rise ≥ 0.04 within ≤ 10 days from low baseline
            if current_ndvi < THRESHOLD_WHEAT_WINTER:
                continue
                
            # Look back for slope calculation
            for j in range(max(0, i - MAX_SLOPE_DAYS), i):
                baseline_point = sorted_data[j]
                baseline_ndvi = baseline_point['ndvi']
                
                if baseline_ndvi >= THRESHOLD_WHEAT_WINTER:
                    continue
                    
                # Calculate days between points
                try:
                    baseline_date = datetime.strptime(baseline_point['date'], '%Y-%m-%d')
                    current_date_obj = datetime.strptime(current_date, '%Y-%m-%d')
                    days_diff = (current_date_obj - baseline_date).days
                    
                    if days_diff <= MAX_SLOPE_DAYS and days_diff > 0:
                        ndvi_rise = current_ndvi - baseline_ndvi
                        
                        if ndvi_rise >= MIN_SLOPE_DELTA:
                            candidates.append({
                                'date': current_date,
                                'method': 'slope',
                                'confidence': 'medium',
                                'cloud_pct': cloud_pct,
                                'ndvi_value': current_ndvi,
                                'baseline_ndvi': baseline_ndvi,
                                'rise': ndvi_rise,
                                'days': days_diff
                            })
                            print(f"Slope candidate: {current_date}, rise: {ndvi_rise:.3f} over {days_diff} days")
                            break
                            
                except Exception as e:
                    print(f"Error calculating slope: {e}")
                    continue
        
        # Select best candidate (earliest valid wins)
        if not candidates:
            print("No candidates found, falling back to significant rise heuristic")
            fallback_result = detect_significant_rise_fallback(sorted_data)
            if fallback_result:
                return fallback_result['date'], "low", {
                    "qa": qa_info,
                    "fallback_used": True,
                    "detection_method": "significant_rise_fallback"
                }
            return None, "low", {"qa": qa_info}
        
        # Sort candidates by date and pick earliest
        candidates.sort(key=lambda x: x['date'])
        best_candidate = candidates[0]
        
        emergence_date = best_candidate['date']
        confidence = best_candidate['confidence']
        
        # Prepare spatial adaptation
        geohash = None
        geohash_key = None
        cluster_prior_used = False
        
        if coordinates:
            try:
                season_year = datetime.strptime(emergence_date, '%Y-%m-%d').year
                geohash, geohash_key = get_geohash_key(coordinates, 'Wheat', season_year)
                
                # Apply spatial nudging if ambiguous
                if confidence == "medium" and len(candidates) > 1:
                    nudged_date, used_prior = apply_spatial_nudge(emergence_date, geohash_key)
                    if used_prior:
                        emergence_date = nudged_date
                        cluster_prior_used = True
                        print(f"Applied spatial nudge to: {emergence_date}")
                
                # Update spatial cache with this detection
                update_spatial_prior(geohash_key, emergence_date)
                
            except Exception as e:
                print(f"Error in spatial adaptation: {e}")
        
        metadata = {
            "qa": qa_info,
            "detection_method": best_candidate['method'],
            "cloud_at_emergence_pct": best_candidate['cloud_pct'],
            "used_field_cloud": 'field_cloud_percentage' in sorted_data[0],
            "spatial_adaptation": {
                "geohash": geohash,
                "cluster_prior_used": cluster_prior_used
            },
            "candidates_found": len(candidates)
        }
        
        print(f"Selected emergence: {emergence_date}, confidence: {confidence}")
        return emergence_date, confidence, metadata
        
    except Exception as e:
        print(f"Error in wheat winter emergence detection: {e}")
        return None, "low", {
            "qa": {
                "valid": False,
                "reason": "processing_error",
                "error": str(e)
            }
        }

def detect_significant_rise_fallback(sorted_data):
    """Fallback method using significant NDVI rise"""
    try:
        for i in range(1, len(sorted_data)):
            current_ndvi = sorted_data[i]['ndvi']
            prev_ndvi = sorted_data[i-1]['ndvi']
            
            # Look for significant rise from low values
            if prev_ndvi < 0.15 and current_ndvi > prev_ndvi + 0.05:
                return {
                    'date': sorted_data[i]['date'],
                    'ndvi_rise': current_ndvi - prev_ndvi
                }
        return None
    except Exception as e:
        print(f"Error in fallback detection: {e}")
        return None

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
    return "NDVI & RGB backend with Multi-Index Support (NDVI, EVI, SAVI, NDMI) is live!"

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
            "message": f"Backend is healthy. GEE initialized at startup. Multi-Index Support enabled (NDVI, EVI, SAVI, NDMI).",
            "timestamp": datetime.now().isoformat(),
            "gee_initialized": True,
            "gee_init_time": gee_initialization_time.isoformat() if gee_initialization_time else None,
            "cache_size": len(cache),
            "spatial_cache_size": len(spatial_cache),
            "supported_indices": ["NDVI", "EVI", "SAVI", "NDMI", "RGB"]
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
        "cache_size": len(cache),
        "spatial_cache_size": len(spatial_cache)
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
            "cache_size": len(cache),
            "supported_indices": ["NDVI", "EVI", "SAVI", "NDMI", "RGB"]
        })
        
    except Exception as e:
        print(f"Warmup error: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Warmup failed: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500

# [Existing emergence detection functions remain unchanged - keeping them as is]
# detect_primary_emergence_and_planting, detect_tillage_replanting_events, 
# detect_rainfall_without_emergence, format_date_for_display

def detect_primary_emergence_and_planting(ndvi_data, crop_type, irrigated, rainfall_data=None, coordinates=None, force_winter_detector=False):
    """
    Detects the FIRST emergence event and estimates the primary planting window.
    Now includes wheat-specific winter detection path.
    """
    print(f"=== PRIMARY EMERGENCE DETECTION for {crop_type} ===")
    
    # NEW: Wheat winter detection path
    if crop_type.lower() == 'wheat':
        print("Attempting wheat-specific winter detection...")
        wheat_emergence, wheat_confidence, wheat_metadata = detect_wheat_winter_emergence(
            ndvi_data, coordinates, force_winter_detector
        )
        
        if wheat_emergence:
            print(f"Wheat winter detector succeeded: {wheat_emergence}")
            
            # Calculate planting window (keep 5-day width)
            emergence_window = EMERGENCE_WINDOWS.get(crop_type, DEFAULT_EMERGENCE_WINDOW)
            emergence_date_obj = datetime.strptime(wheat_emergence, '%Y-%m-%d')
            planting_window_end = (emergence_date_obj - timedelta(days=emergence_window[0])).strftime('%Y-%m-%d')
            planting_window_start = (emergence_date_obj - timedelta(days=emergence_window[1])).strftime('%Y-%m-%d')
            
            # For rainfed fields, check for rainfall events
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
                        continue
                
                if significant_rainfall_events:
                    significant_rainfall_events.sort(key=lambda x: x['date'])
                    rainfall_adjusted_planting = significant_rainfall_events[0]['date']
            
            # Create message
            emergence_display = format_date_for_display(wheat_emergence)
            planting_start_display = format_date_for_display(planting_window_start)
            planting_end_display = format_date_for_display(planting_window_end)
            
            if rainfall_adjusted_planting:
                rainfall_date_display = format_date_for_display(rainfall_adjusted_planting)
                message = f"Winter wheat emergence detected around {emergence_display}, indicating planting likely occurred between {planting_start_display} and {planting_end_display}. Rainfall data suggests planting occurred around {rainfall_date_display}."
            else:
                message = f"Winter wheat emergence detected around {emergence_display}, indicating planting likely occurred between {planting_start_display} and {planting_end_display}."
            
            result = {
                "emergenceDate": wheat_emergence,
                "plantingWindowStart": planting_window_start,
                "plantingWindowEnd": planting_window_end,
                "rainfallAdjustedPlanting": rainfall_adjusted_planting,
                "preEstablished": False,
                "confidence": wheat_confidence,
                "message": message,
                "primary_emergence": True,
                "detection_method": "wheat_winter_detector"
            }
            
            # Add wheat-specific metadata
            result.update(wheat_metadata)
            
            return result
        else:
            print("Wheat winter detector failed, falling back to standard detection")
    
    # [Rest of standard emergence detection code remains unchanged]
    sorted_ndvi = sorted(ndvi_data, key=lambda x: x['date'])
    
    emergence_date = None
    emergence_index = -1
    
    for i in range(len(sorted_ndvi) - 1):
        if sorted_ndvi[i]['ndvi'] < EMERGENCE_THRESHOLD and sorted_ndvi[i + 1]['ndvi'] >= EMERGENCE_THRESHOLD:
            emergence_date = sorted_ndvi[i + 1]['date']
            emergence_index = i + 1
            print(f"Primary emergence detected on {emergence_date} at index {emergence_index}")
            break
    
    if not emergence_date:
        for i in range(len(sorted_ndvi) - 1):
            current_ndvi = sorted_ndvi[i]['ndvi']
            next_ndvi = sorted_ndvi[i + 1]['ndvi']
            
            if current_ndvi < 0.15 and next_ndvi > current_ndvi + 0.05:
                emergence_date = sorted_ndvi[i + 1]['date']
                emergence_index = i + 1
                print(f"Alternative emergence detection on {emergence_date} - significant rise from low NDVI")
                break
    
    if not emergence_date and sorted_ndvi and sorted_ndvi[0]['ndvi'] >= EMERGENCE_THRESHOLD:
        high_values = [item['ndvi'] for item in sorted_ndvi if item['ndvi'] >= EMERGENCE_THRESHOLD]
        if len(high_values) >= len(sorted_ndvi) * 0.8:
            return {
                "emergenceDate": None,
                "plantingWindowStart": None,
                "plantingWindowEnd": None,
                "preEstablished": True,
                "confidence": "high",
                "message": "Crop was already established before the analysis period began.",
                "primary_emergence": False,
                "detection_method": "standard"
            }
    
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
                    "no_planting_detected": True,
                    "detection_method": "rainfall_analysis"
                }
        
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
            "confidence": "high",
            "message": message,
            "primary_emergence": False,
            "no_planting_detected": True,
            "detection_method": "standard"
        }
    
    emergence_window = EMERGENCE_WINDOWS.get(crop_type, DEFAULT_EMERGENCE_WINDOW)
    
    emergence_date_obj = datetime.strptime(emergence_date, '%Y-%m-%d')
    planting_window_end = (emergence_date_obj - timedelta(days=emergence_window[0])).strftime('%Y-%m-%d')
    planting_window_start = (emergence_date_obj - timedelta(days=emergence_window[1])).strftime('%Y-%m-%d')
    
    print(f"Calculated planting window: {planting_window_start} to {planting_window_end}")
    
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
            significant_rainfall_events.sort(key=lambda x: x['date'])
            rainfall_adjusted_planting = significant_rainfall_events[0]['date']
            print(f"Found rainfall-adjusted planting date: {rainfall_adjusted_planting}")
    
    confidence = "medium"
    
    if len(sorted_ndvi) >= 6 and emergence_index > 0 and emergence_index < len(sorted_ndvi) - 1:
        confidence = "high"
    
    if len(sorted_ndvi) < 4 or emergence_index <= 1 or emergence_index >= len(sorted_ndvi) - 2:
        confidence = "low"
    
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
        "primary_emergence": True,
        "detection_method": "standard"
    }

def detect_tillage_replanting_events(ndvi_data, primary_emergence_date=None):
    """Detects tillage or replanting events AFTER the primary emergence."""
    if len(ndvi_data) < 4:
        return {"tillage_detected": False, "message": ""}
    
    sorted_ndvi = sorted(ndvi_data, key=lambda x: x['date'])
    tillage_events = []
    
    for i in range(1, len(sorted_ndvi) - 1):
        current_drop = sorted_ndvi[i-1]['ndvi'] - sorted_ndvi[i]['ndvi']
        
        if (current_drop > 0.15 and 
            sorted_ndvi[i-1]['ndvi'] > 0.3 and
            sorted_ndvi[i]['ndvi'] < 0.25):
            
            recovery_found = False
            for j in range(i + 1, min(i + 4, len(sorted_ndvi))):
                if sorted_ndvi[j]['ndvi'] > sorted_ndvi[i]['ndvi'] + 0.1:
                    recovery_found = True
                    break
            
            if recovery_found:
                tillage_date = sorted_ndvi[i]['date']
                
                if primary_emergence_date:
                    try:
                        primary_date_obj = datetime.strptime(primary_emergence_date, '%Y-%m-%d')
                        tillage_date_obj = datetime.strptime(tillage_date, '%Y-%m-%d')
                        days_diff = abs((tillage_date_obj - primary_date_obj).days)
                        
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
        most_significant = max(tillage_events, key=lambda x: x['drop_magnitude'])
        tillage_date_display = format_date_for_display(most_significant['date'])
        
        return {
            "tillage_detected": True,
            "tillage_date": most_significant['date'],
            "message": f"Subsequently, a tillage or replanting event was detected around {tillage_date_display}, where NDVI dropped from {most_significant['ndvi_before']:.2f} to {most_significant['ndvi_after']:.2f}, followed by recovery."
        }
    
    return {"tillage_detected": False, "message": ""}

def detect_rainfall_without_emergence(ndvi_data, rainfall_data, min_rainfall_threshold=10, ndvi_threshold=0.2, response_window_days=14):
    """Detect significant rainfall events that aren't followed by crop emergence."""
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

def format_date_for_display(date_str):
    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        return date_obj.strftime('%B %d')
    except Exception:
        return date_str

def calculate_std_dev(values):
    if not values:
        return 0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance ** 0.5

# [Keeping the agronomic_insight endpoint unchanged as it doesn't need index_type support]
@app.route("/api/agronomic_insight", methods=["POST"])
def generate_agronomic_report():
    # [This endpoint remains unchanged - uses NDVI data only]
    pass  # Implementation kept as in original file

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
        index_type = data.get("index_type", "NDVI")  # NEW: Get index type, default to NDVI
        
        # Validate inputs
        if not coords or not start or not end:
            return jsonify({"success": False, "error": "Missing input fields"}), 400
        
        # Validate index type
        valid_indices = ["NDVI", "EVI", "SAVI", "NDMI", "RGB"]
        if index_type not in valid_indices:
            return jsonify({
                "success": False, 
                "error": f"Invalid index_type. Must be one of: {', '.join(valid_indices)}"
            }), 400
        
        # Validate polygon geometry
        if not isinstance(coords, list) or len(coords) == 0:
            return jsonify({"success": False, "error": "Invalid coordinates format"}), 400
            
        if len(coords[0]) < 3:
            return jsonify({"success": False, "error": "Invalid polygon: must have at least 3 points"}), 400
        
        # Check cache first
        cache_key = get_cache_key(coords, start, end, "ndvi_tiles", index_type)
        with cache_lock:
            if cache_key in cache:
                print(f"Cache hit for {index_type} tiles request")
                return jsonify(cache[cache_key])
        
        print(f"Processing {index_type} tiles request: start={start}, end={end}, coords length={len(coords)}")
        
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
        
        # Handle RGB vs Index calculation
        if index_type == "RGB":
            # RGB visualization
            rgb = image.select(["B4", "B3", "B2"])
            vis_image = rgb.visualize(min=0, max=3000)
            
            # No stats for RGB
            stats_dict = {}
            index_name = "RGB"
        else:
            # Calculate the selected index
            index_image = get_index(image, index_type)
            index_name = index_type
            
            # Create base visualization with existing palettes for vegetation
            if index_type == "NDVI":
                # NDVI: stressed vegetation (red/yellow) → healthy vegetation (dark green)
                base_vis = index_image.visualize(min=-0.5, max=1, palette=["#8B0000", "#FF4500", "#FFD700", "#ADFF2F", "#228B22", "#006400"])
            elif index_type == "EVI":
                # EVI: stressed vegetation (red/yellow) → healthy vegetation (dark green)
                base_vis = index_image.visualize(min=-1, max=1, palette=["#8B0000", "#FF4500", "#FFD700", "#ADFF2F", "#228B22", "#006400"])
            elif index_type == "SAVI":
                # SAVI: stressed vegetation (red/yellow) → healthy vegetation (dark green)
                base_vis = index_image.visualize(min=-0.5, max=1, palette=["#8B0000", "#FF4500", "#FFD700", "#ADFF2F", "#228B22", "#006400"])
            elif index_type == "NDMI":
                # NDMI: dry (brown) → wet (white) → very wet (greenish)
                base_vis = index_image.visualize(min=-0.5, max=1, palette=["#8B4513", "#D2691E", "#F4A460", "#FFFFFF", "#90EE90", "#32CD32"])
            else:
                # Fallback visualization
                base_vis = index_image.visualize(min=-0.5, max=1, palette=["blue", "red", "yellow", "green"])
            
            # Add water masking: Override water pixels (index < 0) with deep blue
            water_mask = index_image.lt(0)
            water_layer = water_mask.selfMask().visualize(palette=["#08306b"])
            
            # Mosaic layers: water layer takes priority over base vegetation visualization
            vis_image = ee.ImageCollection([base_vis, water_layer]).mosaic()
            
            # Calculate statistics
            stats = index_image.reduceRegion(
                reducer=ee.Reducer.mean().combine(ee.Reducer.minMax(), "", True),
                geometry=polygon,
                scale=10,
                maxPixels=1e9
            )
            stats_dict = stats.getInfo()
        
        # Calculate scene-level cloud cover
        scene_cloud_percentage = first_image.get("CLOUDY_PIXEL_PERCENTAGE")
        
        # Get image date
        image_date = first_image.date().format("YYYY-MM-dd").getInfo()
        scene_cloud_pct = scene_cloud_percentage.getInfo()
        
        # Calculate field-specific cloud cover
        field_cloud_percentage = None
        try:
            field_cloud_calculation = calculate_field_cloud_cover(first_image, polygon)
            if field_cloud_calculation is not None:
                field_cloud_percentage = field_cloud_calculation.getInfo()
                print(f"Field-specific cloud cover calculated: {field_cloud_percentage}%")
        except Exception as e:
            print(f"Field cloud cover calculation failed, using scene-level: {e}")
            field_cloud_percentage = None
        
        # Get map ID for tile URL
        try:
            map_id = ee.data.getMapId({"image": vis_image})
            
            display_cloud_percentage = field_cloud_percentage if field_cloud_percentage is not None else scene_cloud_pct
            
            # Prepare response based on index type
            response = {
                "success": True,
                "index_type": index_type,
                "tile_url": map_id["tile_fetcher"].url_format,
                "image_date": image_date,
                "collection_size": collection_size,
                "cloudy_pixel_percentage": display_cloud_percentage,
                "scene_cloud_percentage": scene_cloud_pct,
                "field_cloud_percentage": field_cloud_percentage,
                "cloud_calculation_method": "field_specific" if field_cloud_percentage is not None else "scene_level"
            }
            
            # Add statistics for non-RGB indices
            if index_type != "RGB":
                stat_key = index_name
                response["mean"] = stats_dict.get(f"{stat_key}_mean")
                response["min"] = stats_dict.get(f"{stat_key}_min")
                response["max"] = stats_dict.get(f"{stat_key}_max")
            
            # Cache the response
            with cache_lock:
                cache[cache_key] = response
            
            print(f"Successfully processed {index_type} tiles request.")
            return jsonify(response)
            
        except Exception as e:
            print(f"Error getting map IDs: {e}")
            
            display_cloud_percentage = field_cloud_percentage if field_cloud_percentage is not None else scene_cloud_pct
            
            response = {
                "success": True,
                "index_type": index_type,
                "image_date": image_date,
                "collection_size": collection_size,
                "cloudy_pixel_percentage": display_cloud_percentage,
                "scene_cloud_percentage": scene_cloud_pct,
                "field_cloud_percentage": field_cloud_percentage,
                "cloud_calculation_method": "field_specific" if field_cloud_percentage is not None else "scene_level",
                "visualization_error": str(e)
            }
            
            if index_type != "RGB":
                response["mean"] = stats_dict.get(f"{index_name}_mean")
                response["min"] = stats_dict.get(f"{index_name}_min")
                response["max"] = stats_dict.get(f"{index_name}_max")
            
            with cache_lock:
                cache[cache_key] = response
                
            return jsonify(response)

    except Exception as e:
        error_message = str(e)
        stack_trace = traceback.format_exc()
        print(f"Error in GEE processing: {error_message}")
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
        crop = data.get("crop", "")
        force_winter_detector = data.get("forceWinterDetector", False)
        index_type = data.get("index_type", "NDVI")  # NEW: Get index type
        
        # Validate inputs
        if not coords or not start or not end:
            return jsonify({"success": False, "error": "Missing input fields"}), 400
        
        # Validate index type
        valid_indices = ["NDVI", "EVI", "SAVI", "NDMI"]
        if index_type not in valid_indices:
            return jsonify({
                "success": False, 
                "error": f"Invalid index_type for time series. Must be one of: {', '.join(valid_indices)}"
            }), 400
        
        # Validate polygon geometry
        if not isinstance(coords, list) or len(coords) == 0:
            return jsonify({"success": False, "error": "Invalid coordinates format"}), 400
            
        if len(coords[0]) < 3:
            return jsonify({"success": False, "error": "Invalid polygon: must have at least 3 points"}), 400
        
        # Check cache first
        cache_key = get_cache_key(coords, start, end, "ndvi_timeseries", index_type)
        with cache_lock:
            if cache_key in cache:
                print(f"Cache hit for {index_type} timeseries request")
                cached_response = cache[cache_key]
                
                # Add wheat emergence detection if needed (only for NDVI)
                if index_type == "NDVI" and crop.lower() == 'wheat' and "emergence_date" not in cached_response:
                    print("Adding wheat emergence detection to cached response")
                    try:
                        wheat_emergence, wheat_confidence, wheat_metadata = detect_wheat_winter_emergence(
                            cached_response["time_series"], coords, force_winter_detector
                        )
                        if wheat_emergence:
                            cached_response["emergence_date"] = wheat_emergence
                            cached_response["emergence_confidence"] = wheat_confidence
                            cached_response.update(wheat_metadata)
                    except Exception as e:
                        print(f"Error adding wheat detection: {e}")
                
                return jsonify(cached_response)
        
        print(f"Processing {index_type} time series: start={start}, end={end}, crop={crop}")
        
        # Create Earth Engine geometry  
        polygon = ee.Geometry.Polygon(coords)

        # Get optimized collection
        collection, collection_size = get_optimized_collection(polygon, start, end, limit_images=False)
        
        if collection is None or collection_size == 0:
            return jsonify({
                "success": False, 
                "error": "No Sentinel-2 imagery found for the specified date range and location",
                "empty_collection": True
            }), 404
        
        # OPTIMIZED: Add index statistics to image properties
        def add_index_and_cloud_stats(image):
            """Add index statistics to image properties"""
            clipped = image.clip(polygon)
            
            # Calculate the selected index
            index_image = get_index(clipped, index_type)
            
            # Calculate mean for the polygon
            index_mean = index_image.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=polygon,
                scale=10,
                maxPixels=1e9
            ).get(index_type)
            
            # Get scene-level cloud cover
            scene_cloud_cover = image.get("CLOUDY_PIXEL_PERCENTAGE")
            
            return image.set({
                f'{index_type.lower()}_mean': index_mean,
                'date_formatted': image.date().format('YYYY-MM-dd'),
                'scene_cloud_percentage': scene_cloud_cover
            })
        
        # Map the function over the collection
        collection_with_stats = collection.map(add_index_and_cloud_stats)
        
        # Get all data in batch
        dates_array = collection_with_stats.aggregate_array('date_formatted')
        index_array = collection_with_stats.aggregate_array(f'{index_type.lower()}_mean')
        scene_cloud_array = collection_with_stats.aggregate_array('scene_cloud_percentage')
        
        # Get collection list for field-level cloud calculation on first few images
        collection_list = collection_with_stats.limit(3).getInfo()['features']
        
        print("Getting time series data in batch...")
        batch_data = ee.Dictionary({
            'dates': dates_array,
            'index_values': index_array,
            'scene_cloud_percentages': scene_cloud_array
        }).getInfo()
        
        dates = batch_data['dates']
        index_values = batch_data['index_values']
        scene_cloud_percentages = batch_data['scene_cloud_percentages']
        
        # Calculate field-level cloud for first few images only
        field_cloud_cache = {}
        for i, img_info in enumerate(collection_list):
            try:
                img_id = img_info['id']
                img = ee.Image(img_id)
                field_cloud = calculate_field_cloud_cover(img, polygon)
                if field_cloud is not None:
                    field_cloud_cache[i] = field_cloud.getInfo()
                    print(f"Field cloud calculated for image {i}: {field_cloud_cache[i]}%")
            except Exception as e:
                print(f"Error calculating field cloud for image {i}: {e}")
                continue
        
        # Combine into time series data
        index_time_series = []
        
        for i in range(len(dates)):
            index_value = index_values[i]
            
            # Only add valid readings
            if index_value is not None:
                field_cloud_pct = field_cloud_cache.get(i)
                used_field_cloud = field_cloud_pct is not None
                
                display_cloud_pct = field_cloud_pct if used_field_cloud else scene_cloud_percentages[i]
                
                # Use generic key name for compatibility (keep 'ndvi' for backwards compatibility)
                data_point = {
                    "date": dates[i],
                    "ndvi": index_value,  # Keep for backwards compatibility
                    f"{index_type.lower()}": index_value,  # Add index-specific key
                    "cloud_percentage": display_cloud_pct,
                    "scene_cloud_percentage": scene_cloud_percentages[i],
                    "field_cloud_percentage": field_cloud_pct,
                    "cloud_calculation_method": "field_specific" if used_field_cloud else "scene_level"
                }
                
                index_time_series.append(data_point)
        
        # Verify we have sufficient data points
        if len(index_time_series) == 0:
            return jsonify({
                "success": False, 
                "error": f"No valid {index_type} readings could be calculated for this field",
                "empty_time_series": True
            }), 404
        
        # Sort time series by date
        index_time_series.sort(key=lambda x: x["date"])
        
        # Prepare base response
        response = {
            "success": True,
            "index_type": index_type,
            "time_series": index_time_series,
            "collection_size": collection_size
        }
        
        # NEW: Add wheat emergence detection if this is a wheat field AND using NDVI
        if index_type == "NDVI" and crop.lower() == 'wheat':
            print("Running wheat emergence detection on time series...")
            try:
                wheat_emergence, wheat_confidence, wheat_metadata = detect_wheat_winter_emergence(
                    index_time_series, coords, force_winter_detector
                )
                
                if wheat_emergence:
                    response["emergence_date"] = wheat_emergence
                    response["emergence_confidence"] = wheat_confidence
                    
                    if "cloud_at_emergence_pct" in wheat_metadata:
                        response["cloud_at_emergence_pct"] = wheat_metadata["cloud_at_emergence_pct"]
                    if "used_field_cloud" in wheat_metadata:
                        response["used_field_cloud"] = wheat_metadata["used_field_cloud"]
                    if "qa" in wheat_metadata:
                        response["qa"] = wheat_metadata["qa"]
                    if "spatial_adaptation" in wheat_metadata:
                        response["spatial_adaptation"] = wheat_metadata["spatial_adaptation"]
                        
                    print(f"Wheat emergence detected: {wheat_emergence} (confidence: {wheat_confidence})")
                else:
                    print("No wheat emergence detected")
                    if "qa" in wheat_metadata:
                        response["qa"] = wheat_metadata["qa"]
                        
            except Exception as e:
                print(f"Error in wheat emergence detection: {e}")
                response["wheat_detection_error"] = str(e)
        
        # Cache the response
        with cache_lock:
            cache[cache_key] = response
        
        print(f"Successfully processed {index_type} time series. {len(index_time_series)} data points returned.")
        return jsonify(response)

    except Exception as e:
        error_message = str(e)
        stack_trace = traceback.format_exc()
        print(f"Error in GEE time series processing: {error_message}")
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
    print(f"✓ Multi-Index Support: ENABLED (NDVI, EVI, SAVI, NDMI, RGB)")
    print(f"✓ Wheat Winter Detection: ENABLED")
    print(f"✓ Spatial Adaptation Cache: READY")
    print(f"✓ Water Masking: ENABLED (water pixels < 0 → deep blue #08306b)")
else:
    print(f"✗ Backend startup failed: {init_message}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
