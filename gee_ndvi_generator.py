"""
FastAPI version with real-time logging configuration for Gunicorn multi-worker deployment.
All print() statements replaced with logger.info() for immediate console visibility.
Cloud cover calculation updated to use Google Earth Engine's standard S2_CLOUD_PROBABILITY method.
Authentication middleware integrated for API security.
Performance timing logs added for deployment speed measurement.
"""

import os
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
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZIPMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv
from cachetools import TTLCache
import threading
from middleware.auth import require_auth, log_authentication_status

# Configure real-time logging for Gunicorn multi-worker setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Force stdout line buffering for immediate log output
sys.stdout.reconfigure(line_buffering=True)

# Load environment variables
load_dotenv()

app = FastAPI(title="GEE NDVI Generator API", version="2.0")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://yieldera.co.zw",
        "https://www.yieldera.co.zw",
        "https://dashboard.yieldera.co.zw",
        "https://api.yieldera.co.zw",
        "https://staging.yieldera.co.zw",
        "https://ndvi.staging.yieldera.co.zw"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    expose_headers=["Content-Type", "Authorization"],
    max_age=3600
)

# Enable GZIP compression
app.add_middleware(GZIPMiddleware, minimum_size=1000)

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
        
        # Test the connection
        test = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").limit(1).first()
        _ = test.getInfo()
        
        gee_initialization_time = (datetime.now() - start_time).total_seconds()
        gee_initialized = True
        gee_initializing = False
        
        logger.info(f"GEE initialization successful in {gee_initialization_time:.2f} seconds")
        return True, f"Initialized in {gee_initialization_time:.2f}s"
        
    except Exception as e:
        gee_initialization_error = str(e)
        gee_initializing = False
        logger.error(f"GEE initialization failed: {e}")
        logger.error(traceback.format_exc())
        return False, str(e)

def check_gee_initialized():
    """Dependency to check if GEE is initialized"""
    if not gee_initialized:
        raise HTTPException(
            status_code=503,
            detail={
                "success": False,
                "error": "Google Earth Engine not initialized",
                "initialization_error": gee_initialization_error
            }
        )
    return True

# Pydantic models for request/response
class MapGenerationRequest(BaseModel):
    coords: List[List[float]]
    start_date: str
    end_date: str
    index_type: str = "NDVI"

class TimeSeriesRequest(BaseModel):
    coords: List[List[float]]
    start_date: str
    end_date: str
    index_type: str = "NDVI"
    crop: Optional[str] = None
    planting_date: Optional[str] = None
    force_winter_detector: Optional[bool] = False

class EmergenceDetectionRequest(BaseModel):
    coords: List[List[float]]
    planting_date: str
    crop: str = "Maize"
    emergence_window: Optional[tuple] = None

class WheatEmergenceRequest(BaseModel):
    coords: List[List[float]]
    start_date: str
    end_date: str
    force_winter_detector: bool = False

class AIFieldAnalysisRequest(BaseModel):
    polygon_coords: List[List[float]]
    crop: str
    planting_date: str
    end_date: str

# ============================================================================
# HELPER FUNCTIONS (Copied from original Flask app)
# ============================================================================

def median_smooth(values, window=SMOOTH_WINDOW):
    """Applies a moving median filter to smooth NDVI values."""
    if len(values) < window:
        return values
    
    smoothed = []
    half_window = window // 2
    
    for i in range(len(values)):
        start_idx = max(0, i - half_window)
        end_idx = min(len(values), i + half_window + 1)
        window_values = values[start_idx:end_idx]
        smoothed.append(statistics.median(window_values))
    
    return smoothed

def detect_wheat_winter_emergence(time_series, coords, force_winter_detector=False):
    """
    Detects wheat emergence in winter conditions using slope-based analysis
    with spatial adaptation and median smoothing.
    """
    if len(time_series) < 3:
        logger.info("Not enough data points for wheat winter emergence detection")
        return None, 0, {"qa": "insufficient_data", "reason": "Less than 3 data points"}
    
    # Generate geohash for spatial adaptation
    avg_lat = sum(coord[1] for coord in coords) / len(coords)
    avg_lon = sum(coord[0] for coord in coords) / len(coords)
    geohash = geohash2.encode(avg_lat, avg_lon, precision=5)
    
    # Check spatial cache for adaptive thresholds
    spatial_key = f"wheat_params_{geohash}"
    adaptive_params = None
    
    with spatial_cache_lock:
        if spatial_key in spatial_cache:
            adaptive_params = spatial_cache[spatial_key]
            logger.info(f"Using cached adaptive parameters for geohash {geohash}")
    
    # Use adaptive or default thresholds
    if adaptive_params:
        threshold = adaptive_params.get("threshold", THRESHOLD_WHEAT_WINTER)
        min_slope = adaptive_params.get("min_slope", MIN_SLOPE_DELTA)
        logger.info(f"Adaptive thresholds - threshold: {threshold:.3f}, min_slope: {min_slope:.3f}")
    else:
        threshold = THRESHOLD_WHEAT_WINTER
        min_slope = MIN_SLOPE_DELTA
        logger.info(f"Using default thresholds - threshold: {threshold:.3f}, min_slope: {min_slope:.3f}")
    
    # Extract NDVI values and dates
    dates = [item["date"] for item in time_series]
    ndvi_values = [item["ndvi"] for item in time_series]
    cloud_percentages = [item.get("cloud_percentage", 100) for item in time_series]
    field_cloud_percentages = [item.get("field_cloud_percentage") for item in time_series]
    
    # Apply median smoothing
    smoothed_ndvi = median_smooth(ndvi_values)
    
    # Calculate NDVI amplitude for sanity check
    ndvi_min = min(smoothed_ndvi)
    ndvi_max = max(smoothed_ndvi)
    ndvi_amplitude = ndvi_max - ndvi_min
    
    if ndvi_amplitude < NDVI_AMPLITUDE_MIN:
        logger.info(f"NDVI amplitude too low ({ndvi_amplitude:.3f}) for wheat emergence detection")
        return None, 0, {
            "qa": "low_amplitude",
            "ndvi_amplitude": ndvi_amplitude,
            "reason": f"NDVI amplitude {ndvi_amplitude:.3f} below threshold {NDVI_AMPLITUDE_MIN}"
        }
    
    # Find emergence point
    emergence_date = None
    emergence_confidence = 0
    cloud_at_emergence = None
    used_field_cloud = False
    
    for i in range(1, len(smoothed_ndvi)):
        if smoothed_ndvi[i] >= threshold and cloud_percentages[i] <= CLOUD_CANDIDATE_MAX:
            # Check for slope increase
            slope_window = min(MAX_SLOPE_DAYS, i)
            if slope_window > 0:
                ndvi_increase = smoothed_ndvi[i] - smoothed_ndvi[i - slope_window]
                
                if ndvi_increase >= min_slope:
                    emergence_date = dates[i]
                    emergence_confidence = min(95, int((ndvi_increase / min_slope) * 70) + 25)
                    cloud_at_emergence = cloud_percentages[i]
                    
                    # Check if field-specific cloud cover was used
                    if field_cloud_percentages[i] is not None:
                        used_field_cloud = True
                    
                    logger.info(f"Wheat emergence detected on {emergence_date} with confidence {emergence_confidence}%")
                    logger.info(f"NDVI increase: {ndvi_increase:.3f}, Cloud cover: {cloud_at_emergence:.1f}%")
                    break
    
    # Update spatial cache if emergence was detected
    if emergence_date and adaptive_params is None:
        # Store successful detection parameters for this region
        new_params = {
            "threshold": threshold,
            "min_slope": min_slope,
            "last_updated": datetime.now().isoformat()
        }
        with spatial_cache_lock:
            spatial_cache[spatial_key] = new_params
        logger.info(f"Cached successful parameters for geohash {geohash}")
    
    metadata = {
        "qa": "success" if emergence_date else "no_emergence",
        "ndvi_amplitude": ndvi_amplitude,
        "smoothing_applied": True,
        "smoothing_window": SMOOTH_WINDOW
    }
    
    if emergence_date:
        metadata["cloud_at_emergence_pct"] = cloud_at_emergence
        metadata["used_field_cloud"] = used_field_cloud
        if adaptive_params:
            metadata["spatial_adaptation"] = {
                "geohash": geohash,
                "threshold": threshold,
                "min_slope": min_slope
            }
    
    return emergence_date, emergence_confidence, metadata

def calculate_field_cloud_cover(image, region):
    """
    Calculate field-specific cloud cover using S2_CLOUD_PROBABILITY.
    Returns cloud percentage and method used.
    """
    try:
        # Get the image date for matching cloud probability
        image_date = ee.Date(image.get('system:time_start'))
        
        # Search for S2_CLOUD_PROBABILITY within 1 hour of image capture
        cloud_collection = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY') \
            .filterDate(image_date.advance(-30, 'minute'), image_date.advance(30, 'minute')) \
            .filterBounds(region)
        
        cloud_image = cloud_collection.first()
        
        # If cloud probability data exists, use it
        if cloud_image:
            # Calculate mean cloud probability over the field
            cloud_stats = cloud_image.select('probability').reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region,
                scale=20,
                maxPixels=1e9
            )
            
            field_cloud_pct = ee.Number(cloud_stats.get('probability')).getInfo()
            return field_cloud_pct, 's2_cloud_probability'
        else:
            # Fallback: use scene-level cloud cover
            scene_cloud = image.get('CLOUDY_PIXEL_PERCENTAGE')
            return ee.Number(scene_cloud).getInfo(), 'scene_metadata'
            
    except Exception as e:
        logger.warning(f"Error calculating field cloud cover: {e}")
        # Final fallback
        try:
            scene_cloud = image.get('CLOUDY_PIXEL_PERCENTAGE')
            return ee.Number(scene_cloud).getInfo(), 'scene_metadata_fallback'
        except:
            return None, 'unavailable'

def add_cloud_cover_and_stats(image, region, index_type):
    """
    Enhanced function that adds both cloud cover and index statistics to each image.
    Uses S2_CLOUD_PROBABILITY for accurate field-level cloud assessment.
    """
    # Calculate field-specific cloud cover
    field_cloud_pct, cloud_method = calculate_field_cloud_cover(image, region)
    
    # Get scene-level cloud cover as backup
    scene_cloud_pct = image.get('CLOUDY_PIXEL_PERCENTAGE')
    
    # Calculate index statistics for the field
    index_band = index_type.lower()
    stats = image.select(index_band).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region,
        scale=10,
        maxPixels=1e9
    )
    
    index_mean = stats.get(index_band)
    
    # Format date as YYYY-MM-DD
    date = ee.Date(image.get('system:time_start'))
    date_formatted = date.format('YYYY-MM-dd')
    
    return image.set({
        'field_cloud_percentage': field_cloud_pct,
        'scene_cloud_percentage': scene_cloud_pct,
        'cloud_method': cloud_method,
        f'{index_band}_mean': index_mean,
        'date_formatted': date_formatted
    })

def calculate_vegetation_index(image, index_type):
    """Calculate various vegetation indices"""
    if index_type == "NDVI":
        return image.normalizedDifference(['B8', 'B4']).rename('ndvi')
    elif index_type == "EVI":
        evi = image.expression(
            '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
            {
                'NIR': image.select('B8'),
                'RED': image.select('B4'),
                'BLUE': image.select('B2')
            }
        ).rename('evi')
        return evi
    elif index_type == "SAVI":
        L = 0.5
        savi = image.expression(
            '((NIR - RED) / (NIR + RED + L)) * (1 + L)',
            {
                'NIR': image.select('B8'),
                'RED': image.select('B4'),
                'L': L
            }
        ).rename('savi')
        return savi
    elif index_type == "NDMI":
        return image.normalizedDifference(['B8', 'B11']).rename('ndmi')
    elif index_type == "NDWI":
        return image.normalizedDifference(['B3', 'B8']).rename('ndwi')
    elif index_type == "RGB":
        return image.select(['B4', 'B3', 'B2']).rename(['red', 'green', 'blue'])
    else:
        raise ValueError(f"Unknown index type: {index_type}")

# ============================================================================
# FASTAPI ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "GEE NDVI Generator API - FastAPI Version",
        "version": "2.0",
        "status": "operational" if gee_initialized else "initializing"
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "gee_initialized": gee_initialized,
        "gee_initialization_time": gee_initialization_time,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/gee-status")
async def gee_status():
    """Check GEE initialization status"""
    return {
        "initialized": gee_initialized,
        "initialization_time": gee_initialization_time,
        "error": gee_initialization_error,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/map")
@require_auth
async def generate_map(
    request: MapGenerationRequest,
    gee_check: bool = Depends(check_gee_initialized)
):
    """Generate map tiles for visualization"""
    request_start_time = time.perf_counter()
    logger.info(f"[TIMING] Map generation request started for {request.index_type}")
    
    try:
        # Parse inputs
        coords = request.coords
        start_date = request.start_date
        end_date = request.end_date
        index_type = request.index_type
        
        # Validate index type
        if index_type not in INDEX_CONFIGS:
            raise HTTPException(status_code=400, detail=f"Invalid index_type: {index_type}")
        
        # Generate cache key
        coords_str = json.dumps(coords, sort_keys=True)
        cache_key = hashlib.md5(f"{coords_str}_{start_date}_{end_date}_{index_type}_map".encode()).hexdigest()
        
        # Check cache
        with cache_lock:
            if cache_key in cache:
                logger.info(f"[CACHE HIT] Returning cached map for {index_type}")
                return cache[cache_key]
        
        logger.info(f"[CACHE MISS] Generating new map for {index_type}")
        
        # Create geometry
        polygon = ee.Geometry.Polygon(coords)
        region = polygon.bounds()
        
        # Get Sentinel-2 imagery
        collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
            .filterDate(start_date, end_date) \
            .filterBounds(region) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
        
        collection_size = collection.size().getInfo()
        logger.info(f"Found {collection_size} Sentinel-2 images")
        
        if collection_size == 0:
            raise HTTPException(
                status_code=404,
                detail="No cloud-free Sentinel-2 imagery available for the specified date range"
            )
        
        # Calculate index
        if index_type == "RGB":
            # For RGB, use median composite of RGB bands
            composite = collection.median().select(['B4', 'B3', 'B2'])
            vis_params = {
                'bands': ['B4', 'B3', 'B2'],
                'min': 0,
                'max': 3000,
                'gamma': 1.4
            }
        else:
            # Calculate vegetation index
            def add_index(image):
                return image.addBands(calculate_vegetation_index(image, index_type))
            
            collection_with_index = collection.map(add_index)
            composite = collection_with_index.median()
            
            config = INDEX_CONFIGS[index_type]
            vis_params = {
                'bands': [index_type.lower()],
                'min': config["range"][0],
                'max': config["range"][1],
                'palette': config["palette"]
            }
        
        # Clip to field boundary
        clipped = composite.clip(polygon)
        
        # Generate map tile URL
        map_id = clipped.getMapId(vis_params)
        tile_url = map_id['tile_fetcher'].url_format
        
        # Prepare response
        config = INDEX_CONFIGS[index_type]
        response = {
            "success": True,
            "tile_url": tile_url,
            "index": index_type,
            "palette": config["palette"],
            "range": config["range"],
            "explanation": config["explanation"],
            "collection_size": collection_size
        }
        
        # Cache the response
        with cache_lock:
            cache[cache_key] = response
        
        total_elapsed = time.perf_counter() - request_start_time
        logger.info(f"[TIMING] Map generation completed: {total_elapsed:.3f}s")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        total_elapsed = time.perf_counter() - request_start_time
        logger.error(f"[TIMING] Error in map generation: {str(e)}")
        logger.error(f"[TIMING] Total request time (error): {total_elapsed:.3f}s")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/time-series")
@require_auth
async def get_time_series(
    request: TimeSeriesRequest,
    gee_check: bool = Depends(check_gee_initialized)
):
    """Get time series data for a field"""
    request_start_time = time.perf_counter()
    logger.info(f"[TIMING] Time series request started for {request.index_type}")
    
    try:
        # Parse inputs
        coords = request.coords
        start_date = request.start_date
        end_date = request.end_date
        index_type = request.index_type
        crop = request.crop
        force_winter_detector = request.force_winter_detector
        
        # Validate index type
        if index_type not in INDEX_CONFIGS:
            raise HTTPException(status_code=400, detail=f"Invalid index_type: {index_type}")
        
        # RGB doesn't support time series
        if index_type == "RGB":
            raise HTTPException(
                status_code=400,
                detail="RGB composite does not support time series analysis"
            )
        
        # Generate cache key
        coords_str = json.dumps(coords, sort_keys=True)
        cache_key = hashlib.md5(
            f"{coords_str}_{start_date}_{end_date}_{index_type}_{crop}_{force_winter_detector}_timeseries".encode()
        ).hexdigest()
        
        # Check cache
        with cache_lock:
            if cache_key in cache:
                logger.info(f"[CACHE HIT] Returning cached time series for {index_type}")
                return cache[cache_key]
        
        logger.info(f"[CACHE MISS] Generating new time series for {index_type}")
        
        # Create geometry
        polygon = ee.Geometry.Polygon(coords)
        region = polygon.bounds()
        
        # Get Sentinel-2 imagery
        collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
            .filterDate(start_date, end_date) \
            .filterBounds(region) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 80))
        
        collection_size = collection.size().getInfo()
        logger.info(f"Found {collection_size} Sentinel-2 images")
        
        if collection_size == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No Sentinel-2 imagery available for the specified date range"
            )
        
        # Calculate index for each image
        def add_index(image):
            return image.addBands(calculate_vegetation_index(image, index_type))
        
        collection_with_index = collection.map(add_index)
        
        # Add statistics and cloud cover to each image
        collection_with_stats = collection_with_index.map(
            lambda img: add_cloud_cover_and_stats(img, polygon, index_type)
        )
        
        # Get all data in batch
        batch_data = ee.Dictionary({
            'dates': collection_with_stats.aggregate_array('date_formatted'),
            'index_values': collection_with_stats.aggregate_array(f'{index_type.lower()}_mean'),
            'scene_cloud_percentages': collection_with_stats.aggregate_array('scene_cloud_percentage'),
            'field_cloud_percentages': collection_with_stats.aggregate_array('field_cloud_percentage'),
            'cloud_methods': collection_with_stats.aggregate_array('cloud_method')
        }).getInfo()
        
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
                
                data_point = {
                    "date": dates[i],
                    "ndvi": index_value,  # Keep for backwards compatibility
                    f"{index_type.lower()}": index_value,
                    "cloud_percentage": display_cloud_pct,
                    "scene_cloud_percentage": scene_cloud_percentages[i],
                    "field_cloud_percentage": field_cloud_pct,
                    "cloud_calculation_method": cloud_method
                }
                
                index_time_series.append(data_point)
        
        # Verify we have sufficient data points
        if len(index_time_series) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No valid {index_type} readings could be calculated for this field"
            )
        
        # Sort time series by date
        index_time_series.sort(key=lambda x: x["date"])
        
        # Calculate average cloud cover across time series
        valid_cloud_values = [item["cloud_percentage"] for item in index_time_series if item["cloud_percentage"] is not None]
        avg_cloud_cover_ts = sum(valid_cloud_values) / len(valid_cloud_values) if valid_cloud_values else None
        
        # Prepare response
        config = INDEX_CONFIGS[index_type]
        response = {
            "success": True,
            "index": index_type,
            "palette": config["palette"],
            "range": config["range"],
            "explanation": config["explanation"],
            "time_series": index_time_series,
            "collection_size": collection_size,
            "cloud_cover": avg_cloud_cover_ts,
            "cloud_calculation_method": "s2_cloud_probability_timeseries"
        }
        
        # Add wheat emergence detection if this is a wheat field AND using NDVI
        if index_type == "NDVI" and crop and crop.lower() == 'wheat':
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
        
        # Cache the response
        with cache_lock:
            cache[cache_key] = response
        
        total_elapsed = time.perf_counter() - request_start_time
        logger.info(f"[TIMING] Time series completed: {total_elapsed:.3f}s")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        total_elapsed = time.perf_counter() - request_start_time
        logger.error(f"[TIMING] Error in time series: {str(e)}")
        logger.error(f"[TIMING] Total request time (error): {total_elapsed:.3f}s")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    """Initialize GEE on startup"""
    logger.info("=== FASTAPI STARTUP INITIALIZATION ===")
    success, message = initialize_gee_at_startup()
    if success:
        logger.info(f"✓ GEE Initialization Success: {message}")
        logger.info(f"✓ Multi-Index Support: ENABLED (NDVI, EVI, SAVI, NDMI, NDWI, RGB)")
        logger.info(f"✓ Wheat Winter Detection: ENABLED")
        logger.info(f"✓ Spatial Adaptation Cache: READY")
        logger.info(f"✓ Updated Visualization Ranges: NDMI [-0.2, 0.6], NDWI [0.05, 0.4]")
        logger.info(f"✓ S2_CLOUD_PROBABILITY Method: ENABLED")
        logger.info(f"✓ Authentication Middleware: LOADED")
        logger.info(f"✓ Performance Timing Logs: ENABLED")
    else:
        logger.error(f"✗ GEE Initialization Failed: {message}")
        logger.error("✗ Application may not function properly")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)
