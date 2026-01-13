"""
Updated with real-time logging configuration for Gunicorn multi-worker deployment.
All print() statements replaced with logger.info() for immediate DigitalOcean console visibility.
Cloud cover calculation updated to use Google Earth Engine's standard S2_CLOUD_PROBABILITY method.
Authentication middleware integrated for API security.
Performance timing logs added for deployment speed measurement.
"""

import os
from dotenv import load_dotenv

# Load environment variables immediately
load_dotenv()

import json
import ee
import traceback
import hashlib
import geohash2
import statistics
import logging
import sys
import time
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_compress import Compress
from openai import OpenAI
# from dotenv import load_dotenv # Removed redundant import
from cachetools import TTLCache
import threading
from middleware.auth import require_auth, log_authentication_status
from services.report_service import analyze_growth_stage, build_report_structure, validate_indices
from services.openai_client import generate_ai_analysis
from services.pdf_service import generate_pdf_report
import base64


# Configure real-time logging for Gunicorn multi-worker setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Force stdout line buffering for immediate log output
sys.stdout.reconfigure(line_buffering=True)

# Load environment variables - Moved to top
# load_dotenv() 

app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": [
            "http://localhost:3000",
            "https://yieldera.co.zw",
            "https://www.yieldera.co.zw",
            "https://yieldera.net",
            "https://www.yieldera.net",
            "https://dashboard.yieldera.co.zw",
            "https://api.yieldera.co.zw",
            "https://staging.yieldera.co.zw",
            "https://ndvi.staging.yieldera.co.zw"
        ],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True,
        "expose_headers": ["Content-Type", "Authorization"],
        "max_age": 3600
    }
})

# Enable GZIP compression
Compress(app)

# Log authentication status
log_authentication_status()

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# In-memory caching with TTL (Time To Live)
cache = TTLCache(maxsize=1000, ttl=3600)  # Cache for 1 hour, max 1000 items
cache_lock = threading.Lock()

# Geohash spatial adaptation cache for wheat emergence patterns
spatial_cache = TTLCache(maxsize=500, ttl=86400)  # 24 hour TTL for spatial patterns
spatial_cache_lock = threading.Lock()

# FINAL SCIENTIFIC COLOR AND RANGE CONFIGURATION FOR AFRICAN CROPLANDS
INDEX_CONFIGS = {
    # Vegetation health indices: Red (stressed) → Yellow (moderate) → Green (healthy dense canopy)
    "NDVI": {
        "range": [0, 1],  # Clamped to 0-1 to avoid blue tints from negative values
        "palette": ["#a50026", "#f46d43", "#fee08b", "#a6d96a", "#1a9850"],
        "explanation": "Vegetation health scale: red = sparse/stressed vegetation, yellow = moderate growth, green = healthy dense canopy."
    },
    "EVI": {
        "range": [0, 1],  # Clamped to 0-1 to avoid blue tints from negative values
        "palette": ["#a50026", "#f46d43", "#fee08b", "#a6d96a", "#1a9850"],
        "explanation": "Enhanced vegetation health scale: red = sparse/stressed vegetation, yellow = moderate growth, green = healthy dense canopy."
    },
    "SAVI": {
        "range": [0, 1],  # Clamped to 0-1 to avoid blue tints from negative values
        "palette": ["#a50026", "#f46d43", "#fee08b", "#a6d96a", "#1a9850"],
        "explanation": "Soil-adjusted vegetation scale: red = sparse/stressed vegetation, yellow = moderate growth, green = healthy dense canopy."
    },
    # Canopy moisture index: Yellow (dry) → Light Green (moderate) → Dark Green (moist/saturated)
    "NDMI": {
        "range": [-0.2, 0.6],  # Updated range for optimal moisture detection in African croplands
        "palette": ["#fdae61", "#ffffbf", "#a6d96a", "#1a9850"],
        "explanation": "Canopy moisture index: yellow = dry canopy, light green = moderate moisture, dark green = moist/saturated canopy."
    },
    # Water detection index: Yellow/Green (dry/vegetated) → Blue (open water)
    "NDWI": {
        "range": [0.05, 0.4],  # Optimized range for surface and sub-canopy water detection
        "palette": ["#fff7bc", "#c7e9b4", "#7fcdbb", "#41b6c4", "#1d91c0", "#0c2c84"],
        "explanation": "Water detection index: yellow/green = dry or vegetated areas, blue = water bodies or very high moisture."
    },
    # True-color composite (unchanged)
    "RGB": {
        "range": [0, 255],
        "palette": [],
        "explanation": "True-color imagery, showing the field as it would appear to the human eye."
    }
}

# Define crop-specific emergence windows (in days)
EMERGENCE_WINDOWS = {
    "Maize": (6, 10),
    "Soyabeans": (7, 11),
    "Sorghum": (6, 10),
    "Cotton": (5, 9),
    "Groundnuts": (6, 10),
    "Barley": (7, 11),
    "Wheat": (3,6),
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
gee_initialization_error = None
gee_initializing = False

def initialize_gee_at_startup():
    """Initialize Google Earth Engine once at server startup"""
    global gee_initialization_time, gee_initialized, gee_initialization_error, gee_initializing
    
    if gee_initialized:
        return True, "GEE already initialized"
    
    if gee_initialization_error:
        return False, f"GEE initialization previously failed: {gee_initialization_error}"
    
    if gee_initializing:
        return False, "GEE initialization in progress"
    
    try:
        gee_initializing = True
        logger.info("Initializing Google Earth Engine at startup...")
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
        gee_initializing = False
        logger.info(f"GEE initialized successfully at startup in {init_duration:.2f} seconds")
        
        return True, f"GEE initialized in {init_duration:.2f}s"
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"GEE initialization error at startup: {error_msg}")
        gee_initialization_error = error_msg
        gee_initialized = False
        gee_initializing = False
        return False, f"GEE initialization failed: {error_msg}"

def get_cache_key(coords, start_date, end_date, endpoint_type, index_type="NDVI"):
    """Generate a cache key for the given parameters"""
    coords_str = json.dumps(coords, sort_keys=True)
    key_string = f"{endpoint_type}_{coords_str}_{start_date}_{end_date}_{index_type}"
    return hashlib.md5(key_string.encode()).hexdigest()

def get_index(image, index_type):
    """
    Calculate the specified vegetation index with proper value clamping.
    Supports: NDVI, EVI, SAVI, NDMI, NDWI
    """
    if index_type == "NDVI":
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        # Clamp negative values to 0 to avoid blue tints in visualization
        return ndvi.where(ndvi.lt(0), 0)

    elif index_type == "EVI":
        nir = image.select('B8').divide(10000)
        red = image.select('B4').divide(10000)
        blue = image.select('B2').divide(10000)
        evi = image.expression(
            '2.5 * ((NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1))',
            {'NIR': nir, 'RED': red, 'BLUE': blue}
        ).rename('EVI')
        # Clamp to [0, 1] range to avoid blue tints
        return evi.where(evi.gt(1), 1).where(evi.lt(0), 0)

    elif index_type == "SAVI":
        nir = image.select('B8').divide(10000)
        red = image.select('B4').divide(10000)
        savi = image.expression(
            '((NIR - RED) * (1 + L)) / (NIR + RED + L)',
            {'NIR': nir, 'RED': red, 'L': 0.5}
        ).rename('SAVI')
        # Clamp negative values to 0 to avoid blue tints
        return savi.where(savi.lt(0), 0)

    elif index_type == "NDMI":
        return image.normalizedDifference(['B8', 'B11']).rename('NDMI')

    elif index_type == "NDWI":
        ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
        # Apply optimized range clipping for surface water detection
        config = INDEX_CONFIGS["NDWI"]
        min_val, max_val = config["range"]
        return ndwi.where(ndwi.lt(min_val), min_val).where(ndwi.gt(max_val), max_val)

    else:
        # Default fallback: NDVI with clamping
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        return ndvi.where(ndvi.lt(0), 0)

# NEW: Calculate dynamic visualization range
def calculate_dynamic_range(index_image, polygon, index_type):
    """
    Calculate dynamic min/max values for visualization using reduceRegion
    """
    try:
        # Get actual min/max from the image
        stats = index_image.reduceRegion(
            reducer=ee.Reducer.minMax(),
            geometry=polygon,
            scale=20,
            maxPixels=1e9
        )
        
        stats_dict = stats.getInfo()
        stat_key = index_type
        
        actual_min = stats_dict.get(f"{stat_key}_min")
        actual_max = stats_dict.get(f"{stat_key}_max")
        
        # Get default range from config
        config = INDEX_CONFIGS.get(index_type, {"range": [-1, 1]})
        default_min, default_max = config["range"]
        
        # Use actual values if available, otherwise use defaults
        vis_min = actual_min if actual_min is not None else default_min
        vis_max = actual_max if actual_max is not None else default_max
        
        # Ensure reasonable range (avoid extremely narrow ranges)
        if abs(vis_max - vis_min) < 0.01:
            vis_min = default_min
            vis_max = default_max
        
        logger.info(f"Dynamic range for {index_type}: actual({actual_min:.3f}, {actual_max:.3f}) -> vis({vis_min:.3f}, {vis_max:.3f})")
        
        return vis_min, vis_max, actual_min, actual_max
        
    except Exception as e:
        logger.error(f"Error calculating dynamic range for {index_type}: {e}")
        # Fall back to default range
        config = INDEX_CONFIGS.get(index_type, {"range": [-1, 1]})
        default_min, default_max = config["range"]
        return default_min, default_max, None, None

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
        logger.error(f"Error determining winter season: {e}")
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
        logger.error(f"Error generating geohash key: {e}")
        return None, None

def get_spatial_prior(geohash_key):
    """Get emergence date cluster prior from spatial cache"""
    try:
        with spatial_cache_lock:
            if geohash_key in spatial_cache:
                return spatial_cache[geohash_key]
        return None
    except Exception as e:
        logger.error(f"Error getting spatial prior: {e}")
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
        logger.error(f"Error updating spatial prior: {e}")

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
        logger.error(f"Error applying spatial nudge: {e}")
        return candidate_date, False

def detect_wheat_winter_emergence(ndvi_data, coordinates=None, force_winter_detector=False):
    """
    Wheat-specific winter emergence detection using remote sensing only.
    Returns emergence_date, confidence, and metadata.
    """
    try:
        logger.info("=== WHEAT WINTER EMERGENCE DETECTION ===")
        
        # Check if winter detector should be used
        if not force_winter_detector:
            start_date = min(item['date'] for item in ndvi_data)
            end_date = max(item['date'] for item in ndvi_data)
            
            if not is_winter_season(start_date, end_date, coordinates):
                logger.info("Not winter season, falling back to standard detection")
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
        logger.info(f"Applied smoothing to {len(smoothed_data)} points")
        
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
                    logger.info(f"Crossing candidate: {current_date}, NDVI: {prev_ndvi:.3f} -> {current_ndvi:.3f}")
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
                            logger.info(f"Slope candidate: {current_date}, rise: {ndvi_rise:.3f} over {days_diff} days")
                            break
                            
                except Exception as e:
                    logger.error(f"Error calculating slope: {e}")
                    continue
        
        # Select best candidate (earliest valid wins)
        if not candidates:
            logger.info("No candidates found, falling back to significant rise heuristic")
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
                        logger.info(f"Applied spatial nudge to: {emergence_date}")
                
                # Update spatial cache with this detection
                update_spatial_prior(geohash_key, emergence_date)
                
            except Exception as e:
                logger.error(f"Error in spatial adaptation: {e}")
        
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
        
        logger.info(f"Selected emergence: {emergence_date}, confidence: {confidence}")
        return emergence_date, confidence, metadata
        
    except Exception as e:
        logger.error(f"Error in wheat winter emergence detection: {e}")
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
        logger.error(f"Error in fallback detection: {e}")
        return None

def calculate_collection_cloud_cover(collection, polygon, start_date, end_date):
    """
    Calculate average cloud cover across all images in collection using S2_CLOUD_PROBABILITY
    """
    try:
        logger.info("Calculating collection-wide cloud cover using S2_CLOUD_PROBABILITY")
        
        # Get cloud probability collection for the same time period
        cloud_prob_collection = (
            ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
            .filterBounds(polygon)
            .filterDate(start_date, end_date)
        )
        
        # Function to calculate cloud cover for each image
        def add_cloud_cover(image):
            system_index = image.get('system:index')
            
            # Find matching cloud probability image
            cloud_prob_filtered = cloud_prob_collection.filter(
                ee.Filter.eq('system:index', system_index)
            )
            
            # Check if any cloud probability images exist for this index
            cloud_prob_size = cloud_prob_filtered.size()
            cloud_prob_exists = cloud_prob_size.gt(0)
            
            def calculate_with_probability():
                # Get the first (and should be only) matching cloud probability image
                cloud_prob_image = cloud_prob_filtered.first()
                # Get probability band and calculate mean for the field
                probability = cloud_prob_image.select('probability').clip(polygon)
                mean_cloud_prob = probability.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=polygon,
                    scale=20,
                    maxPixels=1e9
                ).get('probability')
                return mean_cloud_prob
            
            def calculate_with_qa60():
                # Fallback to QA60 method
                band_names = image.bandNames()
                qa60_exists = band_names.contains('QA60')
                
                def qa60_calculation():
                    qa60 = image.select('QA60').clip(polygon)
                    cloud_bit_10 = qa60.bitwiseAnd(1 << 10).gt(0)
                    cloud_bit_11 = qa60.bitwiseAnd(1 << 11).gt(0)
                    cloud_mask = cloud_bit_10.Or(cloud_bit_11)
                    
                    total_pixels = qa60.gte(0).reduceRegion(
                        reducer=ee.Reducer.count(),
                        geometry=polygon,
                        scale=20,
                        maxPixels=1e9
                    ).get('QA60')
                    
                    cloudy_pixels = cloud_mask.reduceRegion(
                        reducer=ee.Reducer.sum(),
                        geometry=polygon,
                        scale=20,
                        maxPixels=1e9
                    ).get('QA60')
                    
                    cloud_percentage = ee.Algorithms.If(
                        ee.Number(total_pixels).gt(0),
                        ee.Number(cloudy_pixels).divide(ee.Number(total_pixels)).multiply(100),
                        0
                    )
                    return cloud_percentage
                
                # Return QA60 calculation if QA60 exists, otherwise return 0
                return ee.Algorithms.If(qa60_exists, qa60_calculation(), 0)
            
            # Use cloud probability if available, otherwise fallback to QA60
            field_cloud = ee.Algorithms.If(
                cloud_prob_exists,
                calculate_with_probability(),
                calculate_with_qa60()
            )
            
            return image.set('field_cloud', field_cloud)
        
        # Map function over collection and get mean cloud cover
        collection_with_cloud = collection.map(add_cloud_cover)
        
        # Calculate average cloud cover across all images
        avg_cloud_cover = collection_with_cloud.aggregate_mean('field_cloud')
        
        return avg_cloud_cover
        
    except Exception as e:
        logger.error(f"Error calculating collection cloud cover: {e}")
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
    logger.info(f"Total available images: {total_size}")
    
    if total_size == 0:
        return None, 0, None
    
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
    logger.info(f"Filtered collection size: {collection_size} (cloud < {cloud_threshold}%)")
    
    # Fallback if no images after filtering
    if collection_size == 0:
        logger.info("No images found with initial cloud threshold, trying fallback...")
        for fallback_threshold in [50, 80]:
            collection = (
                base_collection
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", fallback_threshold))
                .sort("CLOUDY_PIXEL_PERCENTAGE")
                .limit(max_images)
            )
            collection_size = collection.size().getInfo()
            if collection_size > 0:
                logger.info(f"Fallback successful: {collection_size} images with cloud < {fallback_threshold}%")
                break
    
    # Calculate collection-wide cloud cover using new method
    avg_cloud_cover = None
    if collection_size > 0:
        try:
            avg_cloud_cover_calc = calculate_collection_cloud_cover(collection, polygon, start_date, end_date)
            if avg_cloud_cover_calc is not None:
                avg_cloud_cover = avg_cloud_cover_calc.getInfo()
                logger.info(f"Collection average cloud cover: {avg_cloud_cover:.1f}%")
        except Exception as e:
            logger.error(f"Error calculating collection cloud cover: {e}")
    
    return collection, collection_size, avg_cloud_cover

@app.route("/")
def index():
    return "NDVI & RGB backend with Multi-Index Support (NDVI, EVI, SAVI, NDMI, NDWI, RGB) is live!"

@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint that responds quickly during initialization"""
    try:
        # Return fast response even during initialization
        if gee_initializing:
            return jsonify({
                "success": True, 
                "message": "Backend is starting up, GEE initialization in progress",
                "timestamp": datetime.now().isoformat(),
                "gee_initialized": False,
                "gee_initializing": True,
                "cache_size": len(cache),
                "spatial_cache_size": len(spatial_cache),
                "supported_indices": ["NDVI", "EVI", "SAVI", "NDMI", "NDWI", "RGB"]
            }), 200
        
        if not gee_initialized:
            return jsonify({
                "success": False, 
                "message": f"GEE not initialized: {gee_initialization_error}",
                "timestamp": datetime.now().isoformat(),
                "gee_initialized": False,
                "gee_initializing": False
            }), 500
        
        return jsonify({
            "success": True, 
            "message": f"Backend is healthy. GEE initialized at startup. Multi-Index Support enabled (NDVI, EVI, SAVI, NDMI, NDWI, RGB).",
            "timestamp": datetime.now().isoformat(),
            "gee_initialized": True,
            "gee_initializing": False,
            "gee_init_time": gee_initialization_time.isoformat() if gee_initialization_time else None,
            "cache_size": len(cache),
            "spatial_cache_size": len(spatial_cache),
            "supported_indices": ["NDVI", "EVI", "SAVI", "NDMI", "NDWI", "RGB"]
        })
        
    except Exception as e:
        return jsonify({
            "success": False, 
            "message": f"Health check failed: {str(e)}",
            "timestamp": datetime.now().isoformat(),
            "gee_initialized": gee_initialized,
            "gee_initializing": gee_initializing
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
@require_auth
def warmup():
    """Dedicated warmup endpoint"""
    try:
        # Quick response during initialization
        if gee_initializing:
            return jsonify({
                "success": True,
                "message": "GEE initialization in progress",
                "timestamp": datetime.now().isoformat(),
                "gee_initializing": True
            }), 200
        
        if not gee_initialized:
            # Try to initialize if not already done
            success, message = initialize_gee_at_startup()
            if not success:
                return jsonify({
                    "success": False,
                    "message": f"GEE initialization failed: {message}",
                    "timestamp": datetime.now().isoformat()
                }), 500
        
        # Test a simple Sentinel-2 operation to warm up
        logger.info("Warming up with test Sentinel-2 query...")
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
        logger.info(f"Warmup completed in {warmup_duration:.2f} seconds")
        
        return jsonify({
            "success": True,
            "message": f"Backend warmed up successfully.",
            "warmup_duration_seconds": warmup_duration,
            "timestamp": datetime.now().isoformat(),
            "gee_initialized": True,
            "cache_size": len(cache),
            "supported_indices": ["NDVI", "EVI", "SAVI", "NDMI", "NDWI", "RGB"]
        })
        
    except Exception as e:
        logger.error(f"Warmup error: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Warmup failed: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500

# MODIFIED: Enhanced primary emergence detection with wheat winter path
def detect_primary_emergence_and_planting(ndvi_data, crop_type, irrigated, rainfall_data=None, coordinates=None, force_winter_detector=False):
    """
    Detects the FIRST emergence event and estimates the primary planting window.
    Now includes wheat-specific winter detection path.
    """
    logger.info(f"=== PRIMARY EMERGENCE DETECTION for {crop_type} ===")
    
    # NEW: Wheat winter detection path
    if crop_type.lower() == 'wheat':
        logger.info("Attempting wheat-specific winter detection...")
        wheat_emergence, wheat_confidence, wheat_metadata = detect_wheat_winter_emergence(
            ndvi_data, coordinates, force_winter_detector
        )
        
        if wheat_emergence:
            logger.info(f"Wheat winter detector succeeded: {wheat_emergence}")
            
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
            logger.info("Wheat winter detector failed, falling back to standard detection")
    
    # EXISTING: Standard emergence detection for non-wheat crops or wheat fallback
    sorted_ndvi = sorted(ndvi_data, key=lambda x: x['date'])
    
    emergence_date = None
    emergence_index = -1
    
    # Look for the first time NDVI crosses the emergence threshold
    for i in range(len(sorted_ndvi) - 1):
        if sorted_ndvi[i]['ndvi'] < EMERGENCE_THRESHOLD and sorted_ndvi[i + 1]['ndvi'] >= EMERGENCE_THRESHOLD:
            emergence_date = sorted_ndvi[i + 1]['date']
            emergence_index = i + 1
            logger.info(f"Primary emergence detected on {emergence_date} at index {emergence_index}")
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
                logger.info(f"Alternative emergence detection on {emergence_date} - significant rise from low NDVI")
                break
    
    # Check if crop was pre-established (all values already high)
    if not emergence_date and sorted_ndvi and sorted_ndvi[0]['ndvi'] >= EMERGENCE_THRESHOLD:
        high_values = [item['ndvi'] for item in sorted_ndvi if item['ndvi'] >= EMERGENCE_THRESHOLD]
        if len(high_values) >= len(sorted_ndvi) * 0.8:  # 80% of values are high
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
    
    # If still no emergence detected
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
    
    # Calculate planting window based on crop-specific emergence timing
    emergence_window = EMERGENCE_WINDOWS.get(crop_type, DEFAULT_EMERGENCE_WINDOW)
    
    # Calculate planting window by rolling back from emergence date
    emergence_date_obj = datetime.strptime(emergence_date, '%Y-%m-%d')
    planting_window_end = (emergence_date_obj - timedelta(days=emergence_window[0])).strftime('%Y-%m-%d')
    planting_window_start = (emergence_date_obj - timedelta(days=emergence_window[1])).strftime('%Y-%m-%d')
    
    logger.info(f"Calculated planting window: {planting_window_start} to {planting_window_end}")
    
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
                logger.error(f"Error processing rainfall event: {e}")
                continue
        
        if significant_rainfall_events:
            significant_rainfall_events.sort(key=lambda x: x['date'])
            rainfall_adjusted_planting = significant_rainfall_events[0]['date']
            logger.info(f"Found rainfall-adjusted planting date: {rainfall_adjusted_planting}")
    
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
        "primary_emergence": True,
        "detection_method": "standard"
    }

# SECONDARY FUNCTION: Detect tillage/replanting events (unchanged)
def detect_tillage_replanting_events(ndvi_data, primary_emergence_date=None):
    """
    Detects tillage or replanting events AFTER the primary emergence.
    This is secondary analysis to complement the primary planting date.
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
            logger.error(f"Error processing rainfall event: {e}")
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
                logger.error(f"Error processing NDVI date: {e}")
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

def calculate_std_dev(values):
    if not values:
        return 0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance ** 0.5

@app.route("/api/agronomic_insight", methods=["POST"])
@require_auth
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
        coordinates = data.get("coordinates")  # NEW: for wheat winter detection
        force_winter_detector = data.get("forceWinterDetector", False)  # NEW: override flag
        
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
                    logger.error(f"Error calculating NDVI change rate: {e}")

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
        
        # MODIFIED: Primary emergence analysis with wheat winter support
        logger.info("=== STARTING PRIMARY EMERGENCE ANALYSIS ===")
        
        # Step 1: Detect PRIMARY emergence and calculate planting window (now with wheat support)
        primary_results = detect_primary_emergence_and_planting(
            ndvi_data=ndvi_data,
            crop_type=crop,
            irrigated=irrigated,
            rainfall_data=rainfall_data if irrigated == "No" else None,
            coordinates=coordinates,  # NEW: for wheat spatial adaptation
            force_winter_detector=force_winter_detector  # NEW: override flag
        )
        
        logger.info(f"Primary emergence results: {primary_results}")
        
        # Step 2: Detect SECONDARY tillage/replanting events
        tillage_results = detect_tillage_replanting_events(
            ndvi_data=ndvi_data,
            primary_emergence_date=primary_results.get("emergenceDate")
        )
        
        logger.info(f"Tillage detection results: {tillage_results}")
        
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
                logger.info(f"Boosted confidence to high based on data quality")
        
        # Create appropriate prompts based on planting detection
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
            detection_method = primary_results.get("detection_method", "standard")
            method_text = " (using enhanced winter wheat detection)" if detection_method == "wheat_winter_detector" else ""
            
            prompt = f"""Field Analysis Summary:
Field: {field_name} - {crop} ({'Irrigated' if irrigated == 'Yes' else 'Rainfed'})
Analysis Period: {date_range}
Planting Status: {planting_window_text}{method_text}
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
            logger.info(f"Sending request to generate insight for field: {field_name}")
            
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
            
            # Add detailed planting date estimation with wheat metadata
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
                    "no_planting_detected": primary_results.get("no_planting_detected", False),
                    "detection_method": primary_results.get("detection_method", "standard")
                }
                
                # Add wheat-specific metadata if available
                if "qa" in primary_results:
                    response_data["planting_date_estimation"]["qa"] = primary_results["qa"]
                if "spatial_adaptation" in primary_results:
                    response_data["planting_date_estimation"]["spatial_adaptation"] = primary_results["spatial_adaptation"]
                if "cloud_at_emergence_pct" in primary_results:
                    response_data["planting_date_estimation"]["cloud_at_emergence_pct"] = primary_results["cloud_at_emergence_pct"]
                if "used_field_cloud" in primary_results:
                    response_data["planting_date_estimation"]["used_field_cloud"] = primary_results["used_field_cloud"]
                
                # Add tillage information if detected
                if tillage_results["tillage_detected"]:
                    response_data["planting_date_estimation"]["tillage_event"] = {
                        "detected": True,
                        "date": tillage_results["tillage_date"],
                        "message": tillage_results["message"]
                    }
            
            return jsonify(response_data)
            
        except Exception as e:
            logger.error(f"Insight generation error: {str(e)}")
            return jsonify({
                "success": False,
                "error": f"Insight generation error: {str(e)}",
                "fallback_insight": "Unable to generate agronomic insight due to service error. Please try again later."
            }), 500
            
    except Exception as e:
        error_message = str(e)
        stack_trace = traceback.format_exc()
        logger.error(f"Error generating agronomic report: {error_message}")
        logger.error(f"Stack trace: {stack_trace}")
        
        return jsonify({
            "success": False,
            "error": error_message,
            "stack_trace": stack_trace
        }), 500

@app.route('/api/gee_ndvi', methods=['OPTIONS'])
def gee_ndvi_options():
    response = jsonify({'status': 'ok'})
    origin = request.headers.get('Origin')
    allowed_origins = [
        'http://localhost:3000',
        'https://yieldera.co.zw',
        'https://www.yieldera.co.zw',
        'https://yieldera.net',
        'https://www.yieldera.net',
        'https://dashboard.yieldera.co.zw',
        'https://api.yieldera.co.zw',
        'https://staging.yieldera.co.zw',
        'https://ndvi.staging.yieldera.co.zw'
    ]
    if origin in allowed_origins:
        response.headers.add('Access-Control-Allow-Origin', origin)
        response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response, 200

@app.route("/api/gee_ndvi", methods=["POST"])
@require_auth
def generate_ndvi():
    # [TIMING] Start total request timer
    request_start_time = time.perf_counter()
    
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
        index_type = data.get("index_type", "NDVI") # NEW: Get index type, default to NDVI
        
        # [TIMING] Log request parameters
        coords_summary = f"[{len(coords[0]) if coords and len(coords) > 0 else 0} points]"
        logger.info(f"[TIMING] Request started: /api/gee_ndvi | params: {{index_type: {index_type}, coords: {coords_summary}, dates: {start} to {end}}}")
        
        # Validate inputs
        if not coords or not start or not end:
            return jsonify({"success": False, "error": "Missing input fields"}), 400
        
        # Validate index type
        valid_indices = ["NDVI", "EVI", "SAVI", "NDMI", "NDWI", "RGB"]
        if index_type not in valid_indices:
            return jsonify({
                "success": False,
                "error": f"Invalid index_type. Must be one of: {', '.join(valid_indices)}"
            }), 400
        
        # Validate polygon geometry
        if not isinstance(coords, list) or len(coords) == 0:
            return jsonify({"success": False, "error": "Invalid coordinates format"}), 400
            
        # Ensure we have a valid polygon (at least 3 points)
        if len(coords[0]) < 3:
            return jsonify({"success": False, "error": "Invalid polygon: must have at least 3 points"}), 400
        
        # Check cache first
        cache_start_time = time.perf_counter()
        cache_key = get_cache_key(coords, start, end, "ndvi_tiles", index_type)
        with cache_lock:
            if cache_key in cache:
                cache_elapsed = time.perf_counter() - cache_start_time
                total_elapsed = time.perf_counter() - request_start_time
                logger.info(f"[TIMING] Cache hit for {index_type} tiles request")
                logger.info(f"[TIMING] Cache lookup: {cache_elapsed:.3f}s")
                logger.info(f"[TIMING] Total request time (cache hit): {total_elapsed:.3f}s")
                return jsonify(cache[cache_key])
        cache_elapsed = time.perf_counter() - cache_start_time
        logger.info(f"[TIMING] Cache lookup (miss): {cache_elapsed:.3f}s")
        
        # Log incoming request
        logger.info(f"Processing {index_type} tiles request: start={start}, end={end}, coords length={len(coords)}")
        
        # [TIMING] Create Earth Engine geometry
        geometry_start_time = time.perf_counter()
        polygon = ee.Geometry.Polygon(coords)
        geometry_elapsed = time.perf_counter() - geometry_start_time
        
        # [TIMING] Geometry creation: {geometry_elapsed:.3f}s
        logger.info(f"[TIMING] Geometry creation: {geometry_elapsed:.3f}s")


        # [TIMING] Get optimized collection with new cloud cover calculation
        collection_start_time = time.perf_counter()
        collection, collection_size, avg_cloud_cover = get_optimized_collection(polygon, start, end, limit_images=True)
        collection_elapsed = time.perf_counter() - collection_start_time
        logger.info(f"[TIMING] GEE collection filtered: {collection_elapsed:.3f}s")
        
        if collection is None or collection_size == 0:
            total_elapsed = time.perf_counter() - request_start_time
            logger.info(f"[TIMING] Total request time (no data): {total_elapsed:.3f}s")
            return jsonify({
                "success": False, 
                "error": "No Sentinel-2 imagery found for the specified date range and location",
                "empty_collection": True
            }), 404
        
        # [TIMING] Use mosaic instead of median for better performance
        mosaic_start_time = time.perf_counter()
        image = collection.median().clip(polygon)
        first_image = collection.first()
        mosaic_elapsed = time.perf_counter() - mosaic_start_time
        logger.info(f"[TIMING] Image mosaic created: {mosaic_elapsed:.3f}s")
        
        # [TIMING] Handle RGB vs Index calculation
        index_calc_start_time = time.perf_counter()
        if index_type == "RGB":
            # RGB visualization
            rgb = image.select(["B4", "B3", "B2"])
            # Apply performance optimization with reproject
            vis_image = rgb.visualize(min=0, max=3000).reproject(crs='EPSG:4326', scale=30)
            
            # No stats for RGB
            stats_dict = {}
            index_name = "RGB"
        else:
            # Calculate the selected index
            index_image = get_index(image, index_type)
            index_name = index_type
            
            # Get visualization parameters from updated INDEX_CONFIGS
            config = INDEX_CONFIGS[index_type]
            vis_params = {
                "min": config["range"][0],
                "max": config["range"][1],
                "palette": config["palette"]
            }
            
            # Apply performance optimization with reproject
            vis_image = index_image.visualize(**vis_params).reproject(crs='EPSG:4326', scale=10)
            
            # [TIMING] Calculate statistics
            stats_start_time = time.perf_counter()
            stats = index_image.reduceRegion(
                reducer=ee.Reducer.mean().combine(ee.Reducer.minMax(), "", True),
                geometry=polygon,
                scale=10,
                maxPixels=1e9
            )
            stats_dict = stats.getInfo()
            stats_elapsed = time.perf_counter() - stats_start_time
            logger.info(f"[TIMING] Statistics calculation: {stats_elapsed:.3f}s")
        
        index_calc_elapsed = time.perf_counter() - index_calc_start_time
        logger.info(f"[TIMING] Index calculation completed: {index_calc_elapsed:.3f}s")
        
        # [TIMING] Calculate scene-level cloud cover
        cloud_calc_start_time = time.perf_counter()
        scene_cloud_percentage = first_image.get("CLOUDY_PIXEL_PERCENTAGE")
        
        # Get image date
        image_date = first_image.date().format("YYYY-MM-dd").getInfo()
        scene_cloud_pct = scene_cloud_percentage.getInfo()
        cloud_calc_elapsed = time.perf_counter() - cloud_calc_start_time
        logger.info(f"[TIMING] Cloud cover calculated: {cloud_calc_elapsed:.3f}s")
        
        # [TIMING] Get map ID for tile URL
        map_id_start_time = time.perf_counter()
        try:
            map_id = ee.data.getMapId({"image": vis_image})
            map_id_elapsed = time.perf_counter() - map_id_start_time
            logger.info(f"[TIMING] Map ID generation: {map_id_elapsed:.3f}s")
            
            # Use the new collection-wide cloud cover if available
            display_cloud_percentage = avg_cloud_cover if avg_cloud_cover is not None else scene_cloud_pct
            
            # Prepare response with updated configuration
            config = INDEX_CONFIGS[index_type]
            
            # Construct the actual tile URL (GEE API changed format)
            # The mapid now contains the full path: projects/earthengine-legacy/maps/{id}-{token}
            base_url = "https://earthengine.googleapis.com/v1alpha"
            mapid = map_id.get('mapid')
            tile_url = f"{base_url}/{mapid}/tiles/{{z}}/{{x}}/{{y}}"
            
            logger.info(f"[DEBUG] Constructed tile_url: {tile_url}")
            
            response = {
                "success": True,
                "index": index_type,
                "tile_url": tile_url,
                "palette": config["palette"],
                "range": config["range"],
                "explanation": config["explanation"],
                "image_date": image_date,
                "collection_size": collection_size,
                "cloud_cover": display_cloud_percentage,  # NEW: Standardized cloud cover using S2_CLOUD_PROBABILITY
                "scene_cloud_percentage": scene_cloud_pct,
                "cloud_calculation_method": "s2_cloud_probability" if avg_cloud_cover is not None else "scene_level"
            }
            
            # Add statistics for non-RGB indices
            if index_type != "RGB":
                stat_key = index_name
                response["mean"] = stats_dict.get(f"{stat_key}_mean")
                response["min"] = stats_dict.get(f"{stat_key}_min")
                response["max"] = stats_dict.get(f"{stat_key}_max")
            
            # [TIMING] Cache the response
            cache_store_start_time = time.perf_counter()
            with cache_lock:
                cache[cache_key] = response
            cache_store_elapsed = time.perf_counter() - cache_store_start_time
            logger.info(f"[TIMING] Response cached: {cache_store_elapsed:.3f}s")
            
            # [TIMING] Log total request time
            total_elapsed = time.perf_counter() - request_start_time
            logger.info(f"[TIMING] Successfully processed {index_type} tiles request with S2_CLOUD_PROBABILITY.")
            logger.info(f"[TIMING] Total request time: {total_elapsed:.3f}s")
            return jsonify(response)
            
        except Exception as e:
            map_id_elapsed = time.perf_counter() - map_id_start_time
            logger.error(f"[TIMING] Error getting map IDs: {map_id_elapsed:.3f}s - {e}")
            
            display_cloud_percentage = avg_cloud_cover if avg_cloud_cover is not None else scene_cloud_pct
            
            config = INDEX_CONFIGS[index_type]
            response = {
                "success": True,
                "index": index_type,
                "palette": config["palette"],
                "range": config["range"],
                "explanation": config["explanation"],
                "image_date": image_date,
                "collection_size": collection_size,
                "cloud_cover": display_cloud_percentage,
                "scene_cloud_percentage": scene_cloud_pct,
                "cloud_calculation_method": "s2_cloud_probability" if avg_cloud_cover is not None else "scene_level",
                "visualization_error": str(e)
            }
            
            if index_type != "RGB":
                response["mean"] = stats_dict.get(f"{index_name}_mean")
                response["min"] = stats_dict.get(f"{index_name}_min")
                response["max"] = stats_dict.get(f"{index_name}_max")
            
            # [TIMING] Cache the response
            cache_store_start_time = time.perf_counter()
            with cache_lock:
                cache[cache_key] = response
            cache_store_elapsed = time.perf_counter() - cache_store_start_time
            logger.info(f"[TIMING] Response cached (with error): {cache_store_elapsed:.3f}s")
            
            # [TIMING] Log total request time
            total_elapsed = time.perf_counter() - request_start_time
            logger.info(f"[TIMING] Total request time (with visualization error): {total_elapsed:.3f}s")
            return jsonify(response)

    except Exception as e:
        total_elapsed = time.perf_counter() - request_start_time
        error_message = str(e)
        stack_trace = traceback.format_exc()
        logger.error(f"[TIMING] Error in GEE processing: {error_message}")
        logger.error(f"[TIMING] Total request time (error): {total_elapsed:.3f}s")
        logger.error(f"Stack trace: {stack_trace}")
        
        return jsonify({
            "success": False, 
            "error": error_message,
            "stack_trace": stack_trace
        }), 500

@app.route("/api/gee_ndvi_timeseries", methods=["POST"])
@require_auth
def generate_ndvi_timeseries():
    # [TIMING] Start total request timer
    request_start_time = time.perf_counter()
    
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
        crop = data.get("crop", "")  # NEW: for wheat detection
        force_winter_detector = data.get("forceWinterDetector", False)  # NEW: override flag
        index_type = data.get("index_type", "NDVI") # NEW: Get index type
        
        # [TIMING] Log request parameters
        coords_summary = f"[{len(coords[0]) if coords and len(coords) > 0 else 0} points]"
        logger.info(f"[TIMING] Request started: /api/gee_ndvi_timeseries | params: {{index_type: {index_type}, coords: {coords_summary}, dates: {start} to {end}, crop: {crop}}}")
        
        # Validate inputs
        if not coords or not start or not end:
            return jsonify({"success": False, "error": "Missing input fields"}), 400
        
        # Validate index type
        valid_indices = ["NDVI", "EVI", "SAVI", "NDMI", "NDWI"]
        if index_type not in valid_indices:
            return jsonify({
                "success": False,
                "error": f"Invalid index_type for time series. Must be one of: {', '.join(valid_indices)}"
            }), 400
        
        # Validate polygon geometry
        if not isinstance(coords, list) or len(coords) == 0:
            return jsonify({"success": False, "error": "Invalid coordinates format"}), 400
            
        # Ensure we have a valid polygon (at least 3 points)
        if len(coords[0]) < 3:
            return jsonify({"success": False, "error": "Invalid polygon: must have at least 3 points"}), 400
        
        # [TIMING] Check cache first
        cache_start_time = time.perf_counter()
        cache_key = get_cache_key(coords, start, end, "ndvi_timeseries", index_type)
        with cache_lock:
            if cache_key in cache:
                cache_elapsed = time.perf_counter() - cache_start_time
                logger.info(f"[TIMING] Cache hit for {index_type} timeseries request")
                logger.info(f"[TIMING] Cache lookup: {cache_elapsed:.3f}s")
                cached_response = cache[cache_key]
                
                # Add wheat emergence detection if needed (only for NDVI)
                if index_type == "NDVI" and crop.lower() == 'wheat' and "emergence_date" not in cached_response:
                    logger.info("Adding wheat emergence detection to cached response")
                    try:
                        wheat_emergence, wheat_confidence, wheat_metadata = detect_wheat_winter_emergence(
                            cached_response["time_series"], coords, force_winter_detector
                        )
                        if wheat_emergence:
                            cached_response["emergence_date"] = wheat_emergence
                            cached_response["emergence_confidence"] = wheat_confidence
                            cached_response.update(wheat_metadata)
                    except Exception as e:
                        logger.error(f"Error adding wheat detection: {e}")
                
                total_elapsed = time.perf_counter() - request_start_time
                logger.info(f"[TIMING] Total request time (cache hit): {total_elapsed:.3f}s")
                return jsonify(cached_response)
        
        cache_elapsed = time.perf_counter() - cache_start_time
        logger.info(f"[TIMING] Cache lookup (miss): {cache_elapsed:.3f}s")
        
        # Log incoming request
        logger.info(f"Processing {index_type} time series: start={start}, end={end}, crop={crop}")
        
        # [TIMING] Create Earth Engine geometry  
        geometry_start_time = time.perf_counter()
        polygon = ee.Geometry.Polygon(coords)
        geometry_elapsed = time.perf_counter() - geometry_start_time
        logger.info(f"[TIMING] Geometry creation: {geometry_elapsed:.3f}s")

        # [TIMING] Get optimized collection (don't limit for time series)
        collection_start_time = time.perf_counter()
        collection, collection_size, avg_collection_cloud_cover = get_optimized_collection(polygon, start, end, limit_images=False)
        collection_elapsed = time.perf_counter() - collection_start_time
        logger.info(f"[TIMING] GEE collection filtered: {collection_elapsed:.3f}s")
        
        if collection is None or collection_size == 0:
            total_elapsed = time.perf_counter() - request_start_time
            logger.info(f"[TIMING] Total request time (no data): {total_elapsed:.3f}s")
            return jsonify({
                "success": False, 
                "error": "No Sentinel-2 imagery found for the specified date range and location",
                "empty_collection": True
            }), 404
        
        # [TIMING] Get cloud probability collection for joining
        cloud_prob_start_time = time.perf_counter()
        cloud_prob_collection = (
            ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
            .filterBounds(polygon)
            .filterDate(start, end)
        )
        cloud_prob_elapsed = time.perf_counter() - cloud_prob_start_time
        logger.info(f"[TIMING] Cloud probability collection prepared: {cloud_prob_elapsed:.3f}s")
        
        # [TIMING] OPTIMIZED: Add index and cloud statistics to image properties
        def add_index_and_cloud_stats(image):
            """Add index statistics and cloud cover to image properties using S2_CLOUD_PROBABILITY"""
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
            
            # Calculate field-specific cloud cover using S2_CLOUD_PROBABILITY
            system_index = image.get('system:index')
            cloud_prob_filtered = cloud_prob_collection.filter(
                ee.Filter.eq('system:index', system_index)
            )
            
            cloud_prob_size = cloud_prob_filtered.size()
            cloud_prob_exists = cloud_prob_size.gt(0)
            
            def calculate_with_probability():
                cloud_prob_image = cloud_prob_filtered.first()
                probability = cloud_prob_image.select('probability').clip(polygon)
                mean_cloud_prob = probability.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=polygon,
                    scale=20,
                    maxPixels=1e9
                ).get('probability')
                return mean_cloud_prob
            
            def calculate_with_qa60():
                band_names = image.bandNames()
                qa60_exists = band_names.contains('QA60')
                
                def qa60_calculation():
                    qa60 = image.select('QA60').clip(polygon)
                    cloud_bit_10 = qa60.bitwiseAnd(1 << 10).gt(0)
                    cloud_bit_11 = qa60.bitwiseAnd(1 << 11).gt(0)
                    cloud_mask = cloud_bit_10.Or(cloud_bit_11)
                    
                    total_pixels = qa60.gte(0).reduceRegion(
                        reducer=ee.Reducer.count(),
                        geometry=polygon,
                        scale=20,
                        maxPixels=1e9
                    ).get('QA60')
                    
                    cloudy_pixels = cloud_mask.reduceRegion(
                        reducer=ee.Reducer.sum(),
                        geometry=polygon,
                        scale=20,
                        maxPixels=1e9
                    ).get('QA60')
                    
                    cloud_percentage = ee.Algorithms.If(
                        ee.Number(total_pixels).gt(0),
                        ee.Number(cloudy_pixels).divide(ee.Number(total_pixels)).multiply(100),
                        0
                    )
                    return cloud_percentage
                
                return ee.Algorithms.If(qa60_exists, qa60_calculation(), 0)
            
            field_cloud = ee.Algorithms.If(
                cloud_prob_exists,
                calculate_with_probability(),
                calculate_with_qa60()
            )
            
            return image.set({
                f'{index_type.lower()}_mean': index_mean,
                'date_formatted': image.date().format('YYYY-MM-dd'),
                'scene_cloud_percentage': scene_cloud_cover,
                'field_cloud_percentage': field_cloud,
                'cloud_method': ee.Algorithms.If(cloud_prob_exists, 's2_cloud_probability', 'qa60_fallback')
            })
        
        # [TIMING] Map the function over the collection
        stats_mapping_start_time = time.perf_counter()
        collection_with_stats = collection.map(add_index_and_cloud_stats)
        stats_mapping_elapsed = time.perf_counter() - stats_mapping_start_time
        logger.info(f"[TIMING] Statistics mapping prepared: {stats_mapping_elapsed:.3f}s")
        
        # [TIMING] Get all data in batch
        batch_start_time = time.perf_counter()
        dates_array = collection_with_stats.aggregate_array('date_formatted')
        index_array = collection_with_stats.aggregate_array(f'{index_type.lower()}_mean')
        scene_cloud_array = collection_with_stats.aggregate_array('scene_cloud_percentage')
        field_cloud_array = collection_with_stats.aggregate_array('field_cloud_percentage')
        cloud_method_array = collection_with_stats.aggregate_array('cloud_method')
        
        logger.info("Getting time series data in batch with S2_CLOUD_PROBABILITY...")
        batch_data = ee.Dictionary({
            'dates': dates_array,
            'index_values': index_array,
            'scene_cloud_percentages': scene_cloud_array,
            'field_cloud_percentages': field_cloud_array,
            'cloud_methods': cloud_method_array
        }).getInfo()
        batch_elapsed = time.perf_counter() - batch_start_time
        logger.info(f"[TIMING] Time series data extracted: {batch_elapsed:.3f}s")
        
        # [TIMING] Process batch data
        processing_start_time = time.perf_counter()
        dates = batch_data['dates']
        index_values = batch_data['index_values']
        scene_cloud_percentages = batch_data['scene_cloud_percentages']
        field_cloud_percentages = batch_data['field_cloud_percentages']
        cloud_methods = batch_data['cloud_methods']
        
        # Combine into time series data
        index_time_series = []
        
        for i in range(len(dates)):
            index_value = index_values[i]
            
            # Only add valid readings
            if index_value is not None:
                field_cloud_pct = field_cloud_percentages[i] if i < len(field_cloud_percentages) else None
                cloud_method = cloud_methods[i] if i < len(cloud_methods) else 'unknown'
                
                # Use field-specific cloud cover as primary, fallback to scene
                display_cloud_pct = field_cloud_pct if field_cloud_pct is not None else scene_cloud_percentages[i]
                
                # Use generic key name for compatibility (keep 'ndvi' for backwards compatibility)
                data_point = {
                    "date": dates[i],
                    "ndvi": index_value, # Keep for backwards compatibility
                    f"{index_type.lower()}": index_value, # Add index-specific key
                    "cloud_percentage": display_cloud_pct,
                    "scene_cloud_percentage": scene_cloud_percentages[i],
                    "field_cloud_percentage": field_cloud_pct,
                    "cloud_calculation_method": cloud_method
                }
                
                index_time_series.append(data_point)
        
        processing_elapsed = time.perf_counter() - processing_start_time
        logger.info(f"[TIMING] Data processing completed: {processing_elapsed:.3f}s")
        
        # Verify we have sufficient data points
        if len(index_time_series) == 0:
            total_elapsed = time.perf_counter() - request_start_time
            logger.info(f"[TIMING] Total request time (no valid readings): {total_elapsed:.3f}s")
            return jsonify({
                "success": False, 
                "error": f"No valid {index_type} readings could be calculated for this field",
                "empty_time_series": True
            }), 404
        
        # Sort time series by date
        index_time_series.sort(key=lambda x: x["date"])
        
        # Calculate average cloud cover across time series
        valid_cloud_values = [item["cloud_percentage"] for item in index_time_series if item["cloud_percentage"] is not None]
        avg_cloud_cover_ts = sum(valid_cloud_values) / len(valid_cloud_values) if valid_cloud_values else None
        
        # Prepare enriched response with updated index config
        config = INDEX_CONFIGS[index_type]
        response = {
            "success": True,
            "index": index_type,
            "palette": config["palette"],
            "range": config["range"],
            "explanation": config["explanation"],
            "time_series": index_time_series,
            "collection_size": collection_size,
            "cloud_cover": avg_cloud_cover_ts,  # NEW: Average cloud cover using S2_CLOUD_PROBABILITY
            "cloud_calculation_method": "s2_cloud_probability_timeseries"
        }
        
        # [TIMING] NEW: Add wheat emergence detection if this is a wheat field AND using NDVI
        if index_type == "NDVI" and crop.lower() == 'wheat':
            wheat_detection_start_time = time.perf_counter()
            logger.info("Running wheat emergence detection on time series...")
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
                        
                    logger.info(f"Wheat emergence detected: {wheat_emergence} (confidence: {wheat_confidence})")
                else:
                    logger.info("No wheat emergence detected")
                    if "qa" in wheat_metadata:
                        response["qa"] = wheat_metadata["qa"]
                        
            except Exception as e:
                logger.error(f"Error in wheat emergence detection: {e}")
                response["wheat_detection_error"] = str(e)
            
            wheat_detection_elapsed = time.perf_counter() - wheat_detection_start_time
            logger.info(f"[TIMING] Wheat emergence detection: {wheat_detection_elapsed:.3f}s")
        
        # [TIMING] Cache the response
        cache_store_start_time = time.perf_counter()
        with cache_lock:
            cache[cache_key] = response
        cache_store_elapsed = time.perf_counter() - cache_store_start_time
        logger.info(f"[TIMING] Response cached: {cache_store_elapsed:.3f}s")
        
        # [TIMING] Log total time and return
        total_elapsed = time.perf_counter() - request_start_time
        logger.info(f"Successfully processed {index_type} time series with S2_CLOUD_PROBABILITY. {len(index_time_series)} data points returned.")
        logger.info(f"[TIMING] Total request time: {total_elapsed:.3f}s")
        return jsonify(response)

    except Exception as e:
        total_elapsed = time.perf_counter() - request_start_time
        error_message = str(e)
        stack_trace = traceback.format_exc()
        logger.error(f"[TIMING] Error in GEE time series processing: {error_message}")
        logger.error(f"[TIMING] Total request time (error): {total_elapsed:.3f}s")
        logger.error(f"Stack trace: {stack_trace}")
        
        return jsonify({
            "success": False, 
            "error": error_message,
            "stack_trace": stack_trace
        }), 500

# NEW: Advanced Intelligence Report Endpoint
@app.route("/api/advanced-report", methods=["POST"])
@require_auth
def advanced_report():
    try:
        # 1. Parse Input
        data = request.get_json()
        field_name = data.get("field_name", "Unknown Field")
        crop = data.get("crop", "Unknown Crop")
        area = data.get("area", 0.0)
        irrigation = data.get("irrigation", "rainfed")
        planting_date = data.get("planting_date")
        coordinates = data.get("coordinates")
        start_date = data.get("start_date")
        end_date = data.get("end_date")

        if not coordinates or not start_date or not end_date:
             return jsonify({"success": False, "error": "Missing required fields: coordinates, start_date, end_date"}), 400

        # 2. Fetch Data from GEE
        # Create geometry
        # Handle GeoJSON format correctly
        if isinstance(coordinates, dict) and coordinates.get("type") == "Polygon":
            polygon = ee.Geometry.Polygon(coordinates["coordinates"])
        else:
            # Assume it's the raw list of lists format used elsewhere
            polygon = ee.Geometry.Polygon(coordinates)
        
        # Get collection
        collection, size, avg_cloud = get_optimized_collection(polygon, start_date, end_date, limit_images=False)
        
        if not collection or size == 0:
             return jsonify({"success": False, "error": "No imagery found"}), 404

        # Get latest image for snapshot indices
        # Sort by date descending
        latest_image = collection.sort("system:time_start", False).first()
        
        # CRITICAL FIX: Extract actual satellite image observation date
        image_date_ms = latest_image.get("system:time_start").getInfo()
        image_date_obj = datetime.fromtimestamp(image_date_ms / 1000)
        image_date = image_date_obj.strftime("%Y-%m-%d")
        
        logger.info(f"Using satellite image observation date: {image_date}")
        
        # Calculate all indices for latest image
        indices = {}
        for idx in ["NDVI", "EVI", "SAVI", "NDMI", "NDWI"]:
             img = get_index(latest_image, idx)
             val = img.reduceRegion(
                 reducer=ee.Reducer.mean(),
                 geometry=polygon,
                 scale=20,
                 maxPixels=1e9
             ).get(idx).getInfo()
             indices[idx] = val

        # Calculate NDVI time series for growth analysis
        # Map over collection to get NDVI and Date
        def get_ndvi_date(img):
             ndvi = get_index(img, "NDVI")
             val = ndvi.reduceRegion(
                 reducer=ee.Reducer.mean(),
                 geometry=polygon,
                 scale=20,
                 maxPixels=1e9
             ).get("NDVI")
             return ee.Feature(None, {"ndvi": val, "date": img.date().format("YYYY-MM-dd")})
        
        ts_features = collection.map(get_ndvi_date).getInfo()
        ndvi_series = [{"date": f["properties"]["date"], "ndvi": f["properties"]["ndvi"]} for f in ts_features["features"] if f["properties"]["ndvi"] is not None]
        
        # 3. Analyze Growth Stage using ACTUAL IMAGE DATE (not end_date)
        growth_data = analyze_growth_stage(crop, planting_date, image_date)
        
        # 4. Build Report Data
        field_info = {
            "field_name": field_name,
            "crop": crop,
            "area": area,
            "irrigation": irrigation,
            "planting_date": planting_date,
            "coordinates": coordinates
        }
        
        validated_indices = validate_indices(indices)
        
        # Add observation metadata
        observation_metadata = {
            "satellite_observation_date": image_date,
            "date_range_start": start_date,
            "date_range_end": end_date,
            "data_source": "Sentinel-2 L2A",
            "number_of_observations": len(ndvi_series)
        }
        
        # Structure for AI
        report_context = build_report_structure(field_info, growth_data, validated_indices)
        # Add time series summary to context for AI
        report_context["time_series_summary"] = {
            "start_date": start_date,
            "end_date": end_date,
            "data_points": len(ndvi_series),
            "trend": "increasing" if len(ndvi_series) > 1 and ndvi_series[-1]["ndvi"] > ndvi_series[0]["ndvi"] else "decreasing"
        }
        # Add observation metadata
        report_context["observation_metadata"] = observation_metadata

        # 5. Generate AI Analysis
        ai_analysis = generate_ai_analysis(report_context)
        
        # Merge AI analysis into report
        final_report = build_report_structure(field_info, growth_data, validated_indices, ai_analysis)
        # Add observation metadata to final report
        final_report["observation_metadata"] = observation_metadata
        
        # 6. Generate PDF
        pdf_buffer = generate_pdf_report(final_report)
        pdf_base64 = base64.b64encode(pdf_buffer.getvalue()).decode('utf-8') if pdf_buffer else None

        return jsonify({
            "success": True,
            "message": "Intelligence report generated successfully",
            "report": final_report,
            "pdf": {
                "filename": f"{field_name.replace(' ', '_')}_{image_date}_report.pdf",
                "content_type": "application/pdf",
                "base64": pdf_base64
            }
        })

    except Exception as e:
        logger.error(f"Error generating advanced report: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

# NEW: Pre-initialization at startup for preload mode
def startup_initialization():
    """Called during app startup when using --preload"""
    logger.info("=== STARTUP INITIALIZATION (PRELOAD MODE) ===")
    success, message = initialize_gee_at_startup()
    if success:
        logger.info(f"✓ GEE Preload Success: {message}")
        logger.info(f"✓ Multi-Index Support: ENABLED (NDVI, EVI, SAVI, NDMI, NDWI, RGB)")
        logger.info(f"✓ Wheat Winter Detection: ENABLED")
        logger.info(f"✓ Spatial Adaptation Cache: READY")
        logger.info(f"✓ Updated Visualization Ranges: NDMI [-0.2, 0.6], NDWI [0.05, 0.4]")
        logger.info(f"✓ S2_CLOUD_PROBABILITY Method: ENABLED")
        logger.info(f"✓ Authentication Middleware: LOADED")
        logger.info(f"✓ Performance Timing Logs: ENABLED")
    else:
        logger.error(f"✗ GEE Preload Failed: {message}")
        logger.error("✗ Application may not function properly")

# Initialize GEE immediately when module loads (for preload mode)
startup_initialization()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
