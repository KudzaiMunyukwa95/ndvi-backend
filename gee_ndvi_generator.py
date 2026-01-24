"""
FastAPI-based NDVI & RGB generator with Multi-Index Support.
Institutional-grade performance with async/await and background thread offloading.
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
import asyncio
import base64
from datetime import datetime, timedelta
from typing import List, Optional, Union, Dict, Any
from fastapi import FastAPI, Request, HTTPException, Depends, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from cachetools import TTLCache
import threading
from openai import OpenAI

# Import middleware and services
from middleware.auth import verify_auth, log_authentication_status
from services.report_service import analyze_growth_stage, build_report_structure, validate_indices
from services.openai_client import generate_ai_analysis
from services.pdf_service import generate_pdf_report
from core.sentinel1_core import get_radar_visualization_url

# Configure real-time logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Force stdout line buffering
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

app = FastAPI(title="Yieldera NDVI API", version="2.0.0")

# CORS configuration
origins = [
    "http://localhost:3000",
    "https://yieldera.co.zw",
    "https://www.yieldera.co.zw",
    "https://yieldera.net",
    "https://www.yieldera.net",
    "https://dashboard.yieldera.co.zw",
    "https://api.yieldera.co.zw",
    "https://staging.yieldera.co.zw",
    "https://ndvi.staging.yieldera.co.zw"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# In-memory caching
cache = TTLCache(maxsize=1000, ttl=3600)
cache_lock = threading.Lock()
spatial_cache = TTLCache(maxsize=500, ttl=86400)
spatial_cache_lock = threading.Lock()

# Scientific color and range configuration
INDEX_CONFIGS = {
    "NDVI": {
        "range": [0, 1],
        "palette": ["#a50026", "#f46d43", "#fee08b", "#a6d96a", "#1a9850"],
        "explanation": "Vegetation health scale: red = sparse/stressed vegetation, yellow = moderate growth, green = healthy dense canopy."
    },
    "EVI": {
        "range": [0, 1],
        "palette": ["#a50026", "#f46d43", "#fee08b", "#a6d96a", "#1a9850"],
        "explanation": "Enhanced vegetation health scale: red = sparse/stressed vegetation, yellow = moderate growth, green = healthy dense canopy."
    },
    "SAVI": {
        "range": [0, 1],
        "palette": ["#a50026", "#f46d43", "#fee08b", "#a6d96a", "#1a9850"],
        "explanation": "Soil-adjusted vegetation scale: red = sparse/stressed vegetation, yellow = moderate growth, green = healthy dense canopy."
    },
    "NDMI": {
        "range": [0, 1],
        "palette": ["#fdae61", "#ffffbf", "#a6d96a", "#1a9850"],
        "explanation": "Canopy moisture index: yellow = dry canopy, light green = moderate moisture, dark green = moist/saturated canopy."
    },
    "NDWI": {
        "range": [0, 1],
        "palette": ["#fff7bc", "#c7e9b4", "#7fcdbb", "#41b6c4", "#1d91c0", "#0c2c84"],
        "explanation": "Water detection index: yellow/green = dry or vegetated areas, blue = water bodies or very high moisture."
    },
    "RADAR": {
        "range": [0, 1],
        "palette": ["#a50026", "#f46d43", "#fee08b", "#a6d96a", "#1a9850"],
        "explanation": "Radar Vegetation Index (RVI): measures crop structure and biomass independently of cloud cover."
    },
    "RGB": {
        "range": [0, 255],
        "palette": [],
        "explanation": "True-color imagery, showing the field as it would appear to the human eye."
    }
}

EMERGENCE_WINDOWS = {
    "Maize": (6, 10), "Soyabeans": (7, 11), "Sorghum": (6, 10),
    "Cotton": (5, 9), "Groundnuts": (6, 10), "Barley": (7, 11),
    "Wheat": (3, 6), "Millet": (4, 8), "Tobacco": (7, 11)
}

EMERGENCE_THRESHOLD = 0.2
DEFAULT_EMERGENCE_WINDOW = (5, 10)
SIGNIFICANT_RAINFALL = 10
THRESHOLD_WHEAT_WINTER = 0.15
MIN_SLOPE_DELTA = 0.04
MAX_SLOPE_DAYS = 10
CLOUD_CANDIDATE_MAX = 30
SMOOTH_WINDOW = 3
NDVI_AMPLITUDE_MIN = 0.15

# State tracking
gee_state = {
    "initialized": False,
    "error": None,
    "initializing": False,
    "init_time": None
}

# --- Pydantic Models for Input Validation ---

class NdviRequest(BaseModel):
    coordinates: List[List[List[float]]]
    startDate: str
    endDate: str
    index_type: str = "NDVI"

class TimeSeriesRequest(BaseModel):
    coordinates: List[List[List[float]]]
    startDate: str
    endDate: str
    crop: str = ""
    forceWinterDetector: bool = False
    index_type: str = "NDVI"

class AgronomicRequest(BaseModel):
    field_name: str = "Unknown"
    crop: str = "Unknown"
    variety: str = "Unknown"
    irrigated: bool = False
    latitude: Union[float, str] = "Unknown"
    longitude: Union[float, str] = "Unknown"
    date_range: str = "Unknown"
    ndvi_data: List[Dict[str, Any]] = []
    rainfall_data: List[Dict[str, Any]] = []
    temperature_data: List[Dict[str, Any]] = []
    gdd_data: List[Dict[str, Any]] = []
    gdd_stats: Dict[str, Any] = {}
    temperature_summary: Dict[str, Any] = {}
    base_temperature: float = 10.0
    coordinates: Optional[List[List[List[float]]]] = None
    forceWinterDetector: bool = False

class AdvancedReportRequest(BaseModel):
    field_name: str
    crop: str
    area: float
    irrigation: str = "rainfed"
    planting_date: Optional[str] = None
    coordinates: Union[Dict, List]
    start_date: str
    end_date: str

# --- Core Earth Engine Logic (Ported from Backup) ---

def initialize_gee_at_startup():
    if gee_state["initialized"]: return True, "Already initialized"
    try:
        gee_state["initializing"] = True
        logger.info("Initializing Google Earth Engine...")
        service_account_info = json.loads(os.environ["GEE_CREDENTIALS"])
        credentials = ee.ServiceAccountCredentials(
            email=service_account_info["client_email"],
            key_data=json.dumps(service_account_info)
        )
        ee.Initialize(credentials)
        ee.ImageCollection("COPERNICUS/S2_HARMONIZED").first().getInfo()
        gee_state["initialized"] = True
        gee_state["init_time"] = datetime.now()
        logger.info("GEE initialized successfully.")
        return True, "Success"
    except Exception as e:
        gee_state["error"] = str(e)
        logger.error(f"GEE initialization failed: {e}")
        return False, str(e)
    finally:
        gee_state["initializing"] = False

def get_cache_key(coords, start_date, end_date, endpoint_type, index_type="NDVI"):
    coords_str = json.dumps(coords, sort_keys=True)
    key_string = f"{endpoint_type}_{coords_str}_{start_date}_{end_date}_{index_type}"
    return hashlib.md5(key_string.encode()).hexdigest()

def get_index(image, index_type):
    if index_type == "NDVI":
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        return ndvi.where(ndvi.lt(0), 0).where(ndvi.gt(1), 1)
    elif index_type == "EVI":
        nir = image.select('B8').divide(10000)
        red = image.select('B4').divide(10000)
        blue = image.select('B2').divide(10000)
        evi = image.expression('2.5 * ((NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1))', {'NIR': nir, 'RED': red, 'BLUE': blue}).rename('EVI')
        return evi.where(evi.lt(0), 0).where(evi.gt(1), 1)
    elif index_type == "SAVI":
        nir = image.select('B8').divide(10000)
        red = image.select('B4').divide(10000)
        savi = image.expression('((NIR - RED) * (1 + L)) / (NIR + RED + L)', {'NIR': nir, 'RED': red, 'L': 0.5}).rename('SAVI')
        return savi.where(savi.lt(0), 0).where(savi.gt(1), 1)
    elif index_type == "NDMI":
        ndmi = image.normalizedDifference(['B8', 'B11']).rename('NDMI')
        return ndmi.where(ndmi.lt(0), 0).where(ndmi.gt(1), 1)
    elif index_type == "NDWI":
        ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
        return ndwi.where(ndwi.lt(0), 0).where(ndwi.gt(1), 1)
    elif index_type == "RADAR":
        # Sentinel-1 VV and VH are in dB, convert to power for RVI calculation
        # power = 10^(dB/10)
        vv_pwr = ee.Image(10).pow(image.select('VV').divide(10))
        vh_pwr = ee.Image(10).pow(image.select('VH').divide(10))
        # RVI = 4 * VH / (VV + VH)
        rvi = vh_pwr.multiply(4).divide(vv_pwr.add(vh_pwr)).rename('RADAR')
        return rvi.where(rvi.lt(0), 0).where(rvi.gt(1), 1)
    return image.normalizedDifference(['B8', 'B4']).rename('NDVI')

def calculate_std_dev(values):
    if not values: return 0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance ** 0.5

def format_date_for_display(date_str):
    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        return date_obj.strftime('%B %d')
    except Exception:
        return date_str

def smooth_ndvi_series(ndvi_data, window=SMOOTH_WINDOW):
    """Apply 3-point median smoothing to NDVI series"""
    if len(ndvi_data) < window: return ndvi_data
    smoothed_data = []
    sorted_data = sorted(ndvi_data, key=lambda x: x['date'])
    for i in range(len(sorted_data)):
        if i == 0 or i == len(sorted_data) - 1:
            smoothed_data.append(sorted_data[i].copy())
        else:
            window_values = [sorted_data[i-1]['ndvi'], sorted_data[i]['ndvi'], sorted_data[i+1]['ndvi']]
            smoothed_value = statistics.median(window_values)
            smoothed_point = sorted_data[i].copy()
            smoothed_point['ndvi'] = smoothed_value
            smoothed_data.append(smoothed_point)
    return smoothed_data

def is_winter_season(start_date, end_date, coordinates):
    try:
        start_obj = datetime.strptime(start_date, '%Y-%m-%d')
        end_obj = datetime.strptime(end_date, '%Y-%m-%d')
        if coordinates and len(coordinates) > 0 and len(coordinates[0]) > 0:
            lat = coordinates[0][0][1]
        else: lat = -19.0
        winter_months = [4, 5, 6, 7, 8] if lat < 0 else [11, 12, 1, 2, 3]
        curr = start_obj
        while curr <= end_obj:
            if curr.month in winter_months: return True
            if curr.month == 12: curr = curr.replace(year=curr.year + 1, month=1)
            else: curr = curr.replace(month=curr.month + 1)
        return False
    except: return False

def get_geohash_key(coordinates, crop, season_year):
    try:
        if coordinates and len(coordinates) > 0 and len(coordinates[0]) > 0:
            lats = [coord[1] for coord in coordinates[0]]
            lons = [coord[0] for coord in coordinates[0]]
            center_lat, center_lon = sum(lats) / len(lats), sum(lons) / len(lons)
            geohash = geohash2.encode(center_lat, center_lon, precision=5)
            return geohash, f"{geohash}_{crop}_{season_year}"
        return None, None
    except: return None, None

def detect_significant_rise_fallback(sorted_data):
    try:
        for i in range(1, len(sorted_data)):
            if sorted_data[i-1]['ndvi'] < 0.15 and sorted_data[i]['ndvi'] > sorted_data[i-1]['ndvi'] + 0.05:
                return {'date': sorted_data[i]['date'], 'ndvi_rise': sorted_data[i]['ndvi'] - sorted_data[i-1]['ndvi']}
        return None
    except: return None

def detect_wheat_winter_emergence(ndvi_data, coordinates=None, force_winter_detector=False):
    try:
        if not force_winter_detector:
            start_date, end_date = min(item['date'] for item in ndvi_data), max(item['date'] for item in ndvi_data)
            if not is_winter_season(start_date, end_date, coordinates): return None, None, {}
        
        if len(ndvi_data) < 4: return None, "low", {"qa": {"valid": False, "reason": "sparse_data"}}
        smoothed_data = smooth_ndvi_series(ndvi_data)
        ndvi_values = [item['ndvi'] for item in smoothed_data]
        ndvi_amplitude = max(ndvi_values) - min(ndvi_values)
        if ndvi_amplitude < NDVI_AMPLITUDE_MIN: return None, "low", {"qa": {"valid": False, "reason": "low_signal", "ndvi_amplitude": ndvi_amplitude}}
        
        sorted_data = sorted(smoothed_data, key=lambda x: x['date'])
        candidates = []
        for i in range(len(sorted_data)):
            curr = sorted_data[i]
            cloud = curr.get('field_cloud_percentage') or curr.get('cloud_percentage', 0)
            if cloud > CLOUD_CANDIDATE_MAX: continue
            
            if i > 0:
                prev_ndvi = sorted_data[i-1]['ndvi']
                if prev_ndvi < THRESHOLD_WHEAT_WINTER and curr['ndvi'] >= THRESHOLD_WHEAT_WINTER:
                    candidates.append({'date': curr['date'], 'method': 'crossing', 'confidence': 'high', 'cloud_pct': cloud})
                    continue
            
            if curr['ndvi'] < THRESHOLD_WHEAT_WINTER: continue
            for j in range(max(0, i - MAX_SLOPE_DAYS), i):
                base = sorted_data[j]
                if base['ndvi'] >= THRESHOLD_WHEAT_WINTER: continue
                days = (datetime.strptime(curr['date'], '%Y-%m-%d') - datetime.strptime(base['date'], '%Y-%m-%d')).days
                if 0 < days <= MAX_SLOPE_DAYS and (curr['ndvi'] - base['ndvi']) >= MIN_SLOPE_DELTA:
                    candidates.append({'date': curr['date'], 'method': 'slope', 'confidence': 'medium', 'cloud_pct': cloud})
                    break
        
        if not candidates:
            fb = detect_significant_rise_fallback(sorted_data)
            if fb: return fb['date'], "low", {"qa": {"valid": True}, "detection_method": "fallback"}
            return None, "low", {"qa": {"valid": True}}
            
        candidates.sort(key=lambda x: x['date'])
        best = candidates[0]
        return best['date'], best['confidence'], {"detection_method": best['method'], "cloud_at_emergence_pct": best['cloud_pct'], "candidates_found": len(candidates)}
    except Exception as e:
        logger.error(f"Wheat detection error: {e}")
        return None, "low", {"qa": {"valid": False, "error": str(e)}}

def calculate_collection_cloud_cover(collection, polygon, start, end):
    try:
        cloud_prob_col = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY').filterBounds(polygon).filterDate(start, end)
        def add_cloud(img):
            si = img.get('system:index')
            cp_img = cloud_prob_col.filter(ee.Filter.eq('system:index', si)).first()
            field_cloud = ee.Algorithms.If(cp_img, cp_img.select('probability').clip(polygon).reduceRegion(ee.Reducer.mean(), polygon, 20).get('probability'), 0)
            return img.set('field_cloud', field_cloud)
        return collection.map(add_cloud).aggregate_mean('field_cloud')
    except: return None

def get_optimized_collection(polygon, start, end, limit_images=True, index_type="NDVI"):
    if index_type == "RADAR":
        base = ee.ImageCollection("COPERNICUS/S1_GRD")\
                 .filterBounds(polygon)\
                 .filterDate(start, end)\
                 .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))\
                 .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))\
                 .filter(ee.Filter.eq('instrumentMode', 'IW'))
        size = base.size().getInfo()
        if size == 0: return None, 0, None
        col = base.limit(10) if limit_images else base
        final_size = col.size().getInfo()
        return col, final_size, 0 # Cloud cover not applicable for RADAR
        
    base = ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filterBounds(polygon).filterDate(start, end)
    size = base.size().getInfo()
    if size == 0: return None, 0, None
    threshold = 10 if size > 50 else (20 if size > 20 else 80)
    max_img = 15 if size > 50 else (20 if size > 20 else size)
    col = base.filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", threshold)).sort("CLOUDY_PIXEL_PERCENTAGE")
    if limit_images: col = col.limit(max_img)
    
    # Removed collection-level resampling to ensure tile engine stability
    
    final_size = col.size().getInfo()
    avg_cloud = calculate_collection_cloud_cover(col, polygon, start, end)
    return col, final_size, avg_cloud.getInfo() if avg_cloud else None

def detect_rainfall_without_emergence(ndvi_data, rainfall_data, min_rainfall_threshold=10, ndvi_threshold=0.2, response_window_days=14):
    if not rainfall_data or not ndvi_data: return None
    sorted_ndvi = sorted(ndvi_data, key=lambda x: x['date'])
    sorted_rainfall = sorted(rainfall_data, key=lambda x: x['date'])
    failures = []
    for rain in sorted_rainfall:
        if rain.get('rainfall', 0) >= min_rainfall_threshold:
            r_date = datetime.strptime(rain['date'], '%Y-%m-%d')
            end = r_date + timedelta(days=response_window_days)
            window = [n for n in sorted_ndvi if r_date <= datetime.strptime(n['date'], '%Y-%m-%d') <= end]
            if window and len(window) >= 2:
                if all(n['ndvi'] < ndvi_threshold for n in window):
                    days = (datetime.strptime(window[-1]['date'], '%Y-%m-%d') - datetime.strptime(window[0]['date'], '%Y-%m-%d')).days
                    if days >= 7: failures.append({'rainfall_date': rain['date'], 'rainfall_amount': rain['rainfall'], 'days': days})
    if failures:
        failures.sort(key=lambda x: x['rainfall_date'], reverse=True)
        f = failures[0]
        return {'detected': True, 'message': f"Significant rainfall around {format_date_for_display(f['rainfall_date'])} ({f['rainfall_amount']:.1f}mm) without NDVI response in {f['days']} days.", 'rainfall_date': f['rainfall_date']}
    return None

def detect_tillage_replanting_events(ndvi_data, primary_emergence_date=None):
    if len(ndvi_data) < 4: return {"tillage_detected": False, "message": ""}
    sorted_ndvi = sorted(ndvi_data, key=lambda x: x['date'])
    events = []
    for i in range(1, len(sorted_ndvi) - 1):
        drop = sorted_ndvi[i-1]['ndvi'] - sorted_ndvi[i]['ndvi']
        if drop > 0.15 and sorted_ndvi[i-1]['ndvi'] > 0.3 and sorted_ndvi[i]['ndvi'] < 0.25:
            if any(sorted_ndvi[j]['ndvi'] > sorted_ndvi[i]['ndvi'] + 0.1 for j in range(i+1, min(i+4, len(sorted_ndvi)))):
                t_date = sorted_ndvi[i]['date']
                if primary_emergence_date:
                    if abs((datetime.strptime(t_date, '%Y-%m-%d') - datetime.strptime(primary_emergence_date, '%Y-%m-%d')).days) < 14: continue
                events.append({'date': t_date, 'drop': drop, 'before': sorted_ndvi[i-1]['ndvi'], 'after': sorted_ndvi[i]['ndvi']})
    if events:
        e = max(events, key=lambda x: x['drop'])
        return {"tillage_detected": True, "tillage_date": e['date'], "message": f"Tillage event detected around {format_date_for_display(e['date'])} (NDVI drop {e['before']:.2f} to {e['after']:.2f})."}
    return {"tillage_detected": False, "message": ""}

def detect_primary_emergence_and_planting(ndvi_data, crop_type, irrigated, rainfall_data=None, coordinates=None, force_winter_detector=False):
    logger.info(f"=== DETECTING EMERGENCE: {crop_type} ===")
    if crop_type.lower() == 'wheat':
        date, conf, meta = detect_wheat_winter_emergence(ndvi_data, coordinates, force_winter_detector)
        if date:
            win = EMERGENCE_WINDOWS.get('Wheat', DEFAULT_EMERGENCE_WINDOW)
            p_end = (datetime.strptime(date, '%Y-%m-%d') - timedelta(days=win[0])).strftime('%Y-%m-%d')
            p_start = (datetime.strptime(date, '%Y-%m-%d') - timedelta(days=win[1])).strftime('%Y-%m-%d')
            msg = f"Winter wheat emergence detected around {format_date_for_display(date)}, indicating planting between {format_date_for_display(p_start)} and {format_date_for_display(p_end)}."
            res = {"emergenceDate": date, "plantingWindowStart": p_start, "plantingWindowEnd": p_end, "confidence": conf, "message": msg, "primary_emergence": True, "detection_method": "wheat_winter_detector"}
            res.update(meta)
            return res
            
    sorted_ndvi = sorted(ndvi_data, key=lambda x: x['date'])
    emergence_date, emergence_index = None, -1
    for i in range(len(sorted_ndvi) - 1):
        if sorted_ndvi[i]['ndvi'] < EMERGENCE_THRESHOLD and sorted_ndvi[i+1]['ndvi'] >= EMERGENCE_THRESHOLD:
            emergence_date, emergence_index = sorted_ndvi[i+1]['date'], i+1
            break
    if not emergence_date:
        fb = detect_significant_rise_fallback(sorted_ndvi)
        if fb: emergence_date, emergence_index = fb['date'], [n['date'] for n in sorted_ndvi].index(fb['date'])
    
    if not emergence_date and sorted_ndvi and sorted_ndvi[0]['ndvi'] >= EMERGENCE_THRESHOLD:
        high = [n['ndvi'] for n in sorted_ndvi if n['ndvi'] >= EMERGENCE_THRESHOLD]
        if len(high) >= len(sorted_ndvi) * 0.8:
            return {"emergenceDate": None, "preEstablished": True, "confidence": "high", "message": "Crop was already established.", "primary_emergence": False}
            
    if not emergence_date:
        if irrigated == "No" and rainfall_data:
            rf = detect_rainfall_without_emergence(ndvi_data, rainfall_data)
            if rf: return {"emergenceDate": None, "confidence": "medium", "message": rf['message'], "no_planting_detected": True, "primary_emergence": False}
        return {"emergenceDate": None, "confidence": "high", "message": "No planting detected.", "no_planting_detected": True, "primary_emergence": False}
        
    win = EMERGENCE_WINDOWS.get(crop_type, DEFAULT_EMERGENCE_WINDOW)
    p_end = (datetime.strptime(emergence_date, '%Y-%m-%d') - timedelta(days=win[0])).strftime('%Y-%m-%d')
    p_start = (datetime.strptime(emergence_date, '%Y-%m-%d') - timedelta(days=win[1])).strftime('%Y-%m-%d')
    msg = f"Emergence detected around {format_date_for_display(emergence_date)}, indicating planting between {format_date_for_display(p_start)} and {format_date_for_display(p_end)}."
    return {"emergenceDate": emergence_date, "plantingWindowStart": p_start, "plantingWindowEnd": p_end, "confidence": "medium" if len(sorted_ndvi) < 6 else "high", "message": msg, "primary_emergence": True}


# --- FastAPI Async Handlers ---

@app.on_event("startup")
async def startup_event():
    log_authentication_status()
    asyncio.create_task(asyncio.to_thread(initialize_gee_at_startup))

@app.get("/")
async def root():
    return {"message": "NDVI & RGB backend with Multi-Index Support is live!"}

@app.get("/api/health")
async def health():
    if gee_state["initializing"]: return {"success": True, "message": "Starting up..."}
    if not gee_state["initialized"]: raise HTTPException(500, f"Error: {gee_state['error']}")
    return {"success": True, "initialized": True, "cache_size": len(cache)}

@app.post("/api/gee_ndvi")
async def generate_ndvi(req: NdviRequest, auth: bool = Depends(verify_auth)):
    ckey = get_cache_key(req.coordinates, req.startDate, req.endDate, "tiles", req.index_type)
    with cache_lock:
        if ckey in cache: return cache[ckey]
    
    if not gee_state["initialized"]: raise HTTPException(500, "GEE not ready")
    
    def sync_logic():
        poly = ee.Geometry.Polygon(req.coordinates)
        col, size, cloud = get_optimized_collection(poly, req.startDate, req.endDate, index_type=req.index_type)
        if not col or size == 0: raise Exception("No imagery found for path/date")
        first_image = col.first()
        image_date = first_image.date().format("YYYY-MM-dd").getInfo()
        image_time = first_image.date().format("YYYY-MM-dd HH:mm:ss").getInfo()
        
        # Safe property extraction
        is_radar = req.index_type == "RADAR"
        scene_cloud_pct = first_image.get("CLOUDY_PIXEL_PERCENTAGE").getInfo() if not is_radar else 0
        satellite_name = "Sentinel-1" if is_radar else first_image.get("SPACECRAFT_NAME").getInfo()
        
        
        img = col.median().clip(poly)
        stats_dict = {}
        
        if req.index_type == "RGB":
            vis = img.select(["B4","B3","B2"]).visualize(min=0, max=3000)
        elif req.index_type == "RADAR":
            # Engineering a professional RADAR FCC (False Color Composite)
            # ULTIMATE REFINE: High-fidelity separation for crops vs soil
            vv = img.select('VV')
            vh = img.select('VH')
            
            # Convert to power for a clean linear ratio
            vv_pwr = ee.Image(10).pow(vv.divide(10))
            vh_pwr = ee.Image(10).pow(vh.divide(10))
            ratio = vh_pwr.divide(vv_pwr).rename('ratio')
            
            # Recalibrated unitScales for agricultural "pop":
            # R (VV): Sensitivity to urban/soil structure (dB: -18 to -2)
            # G (VH): Sensitivity to crop volume scattering (dB: -24 to -10)
            # B (Ratio): Sensitivity to water/moisture (linear: 0.2 to 0.7)
            vis = ee.Image.rgb(
                vv.unitScale(-18, -2), 
                vh.unitScale(-24, -10), 
                ratio.unitScale(0.2, 0.7)
            ).visualize()
        else:
            idx_img = get_index(img, req.index_type)
            conf = INDEX_CONFIGS.get(req.index_type, INDEX_CONFIGS["NDVI"])
            vis = idx_img.visualize(min=conf["range"][0], max=conf["range"][1], palette=conf["palette"])
            
            # Calculate stats for the index with explicit scale
            try:
                # Optimized scale for field-level analysis
                stats = idx_img.reduceRegion(
                    reducer=ee.Reducer.mean().combine(ee.Reducer.minMax(), "", True),
                    geometry=poly,
                    scale=20, # Increased to 20m for significant speed gain without quality loss for stats
                    maxPixels=1e9
                ).getInfo()
                stats_dict = stats
                logger.info(f"Stats Result for {req.index_type}: {stats_dict}")
            except Exception as e:
                logger.error(f"Stats Error: {e}")

        mid = ee.data.getMapId({"image": vis})
        tile_url = f"https://earthengine.googleapis.com/v1alpha/{mid['mapid']}/tiles/{{z}}/{{x}}/{{y}}"
        
        res = {
            "success": True, 
            "index": req.index_type, 
            "tile_url": tile_url, 
            "cloud_cover": cloud if cloud is not None else scene_cloud_pct, 
            "collection_size": size, 
            "image_date": image_date,
            "scene_cloud_percentage": scene_cloud_pct,
            "satellite": {
                "name": satellite_name,
                "sensor": "SAR" if is_radar else "MSI",
                "processing_level": "Level-1" if is_radar else "Level-2A",
                "acquisition_time": image_time,
                "resolution": "10m",
                "platform": "Copernicus Sentinel-1" if is_radar else "Copernicus Sentinel-2"
            }
        }
        
        if req.index_type != "RGB":
            # Priority 1: Specific combined names (Index_mean, Index_min, Index_max)
            res["mean"] = stats_dict.get(f"{req.index_type}_mean")
            res["min"] = stats_dict.get(f"{req.index_type}_min")
            res["max"] = stats_dict.get(f"{req.index_type}_max")
            
            # Priority 2: Standard band names
            if res["mean"] is None: res["mean"] = stats_dict.get(req.index_type)
            if res["min"] is None: res["min"] = stats_dict.get("min")
            if res["max"] is None: res["max"] = stats_dict.get("max")
            
            # Quality control: strictly clamp [0, 1] for dashboard
            if res["mean"] is not None: res["mean"] = max(0.0, min(1.0, float(res["mean"])))
            if res["min"] is not None: res["min"] = max(0.0, min(1.0, float(res["min"])))
            if res["max"] is not None: res["max"] = max(0.0, min(1.0, float(res["max"])))
            
            # Final fallback for empty/failed keys
            if res["mean"] is None and len(stats_dict) > 0:
                res["mean"] = max(0.0, min(1.0, float(next(iter(stats_dict.values())))))
            
        return res

    try:
        res = await asyncio.to_thread(sync_logic)
        with cache_lock: cache[ckey] = res
        return res
    except Exception as e: raise HTTPException(500, str(e))

@app.post("/api/gee_ndvi_timeseries")
async def generate_timeseries(req: TimeSeriesRequest, auth: bool = Depends(verify_auth)):
    ckey = get_cache_key(req.coordinates, req.startDate, req.endDate, "ts", req.index_type)
    with cache_lock:
        if ckey in cache: return cache[ckey]

    def sync_logic():
        poly = ee.Geometry.Polygon(req.coordinates)
        col, size, _ = get_optimized_collection(poly, req.startDate, req.endDate, limit_images=False, index_type=req.index_type)
        if not col or size == 0: raise Exception("No imagery")
        
        def add_stats(img):
            clipped = img.clip(poly)
            val = get_index(clipped, req.index_type).reduceRegion(ee.Reducer.mean(), poly, 10).get(req.index_type)
            return img.set('val', val, 'date', img.date().format('YYYY-MM-dd'))
        
        data = col.map(add_stats).aggregate_array('val').getInfo()
        dates = col.map(add_stats).aggregate_array('date').getInfo()
        series = [{"date": dates[i], "ndvi": data[i]} for i in range(len(dates)) if data[i] is not None]
        series.sort(key=lambda x: x['date'])
        
        res = {"success": True, "time_series": series, "collection_size": size}
        if req.index_type == "NDVI" and req.crop:
            edate, conf, meta = detect_wheat_winter_emergence(series, req.coordinates, req.forceWinterDetector) if req.crop.lower() == 'wheat' else (None, None, {})
            if edate: res.update({"emergence_date": edate, "emergence_confidence": conf, **meta})
        return res

    try:
        res = await asyncio.to_thread(sync_logic)
        with cache_lock: cache[ckey] = res
        return res
    except Exception as e: raise HTTPException(500, str(e))

@app.post("/api/agronomic_insight")
async def agronomic_insight(req: AgronomicRequest, auth: bool = Depends(verify_auth)):
    if not gee_state["initialized"]: raise HTTPException(500, "GEE not ready")
    
    def sync_logic():
        # --- Start of Complex Formatting Logic from Backup ---
        # Calculate average cloud cover if available
        avg_cloud_cover = None
        if req.ndvi_data:
            cloud_percentages = [item.get("cloud_percentage") for item in req.ndvi_data if item.get("cloud_percentage") is not None]
            if cloud_percentages: avg_cloud_cover = sum(cloud_percentages) / len(cloud_percentages)
        
        # Format NDVI data
        ndvi_formatted = ", ".join([f"{item['date']}: {item['ndvi']:.2f}" for item in req.ndvi_data[:10]]) if req.ndvi_data else "No data"
        if len(req.ndvi_data) > 10: ndvi_formatted += f" (+ {len(req.ndvi_data) - 10} more readings)"
        
        # Process rainfall data
        weekly_rainfall = {}
        if req.irrigated: rainfall_formatted = "Not applicable for irrigated fields"
        elif req.rainfall_data:
            for item in req.rainfall_data:
                date = item.get('date')
                if date:
                    week_key = date[:7] + "-W" + str((int(date[8:10]) - 1) // 7 + 1)
                    weekly_rainfall[week_key] = weekly_rainfall.get(week_key, 0) + item.get('rainfall', 0)
            rainfall_formatted = ", ".join([f"{week}: {total:.1f}mm" for week, total in weekly_rainfall.items()])
        else: rainfall_formatted = "No data"
        
        # Format temperature and GDD
        temp_formatted = "No data"
        if req.temperature_data:
            avg_min = sum(item["min"] for item in req.temperature_data) / len(req.temperature_data)
            avg_max = sum(item["max"] for item in req.temperature_data) / len(req.temperature_data)
            temp_formatted = f"Avg min: {avg_min:.1f}°C, Avg max: {avg_max:.1f}°C"
        
        # Detailed Emergence Detection
        primary_res = detect_primary_emergence_and_planting(req.ndvi_data, req.crop, "Yes" if req.irrigated else "No", req.rainfall_data, req.coordinates, req.forceWinterDetector)
        tillage_res = detect_tillage_replanting_events(req.ndvi_data, primary_res.get("emergenceDate"))
        
        planting_text = primary_res["message"]
        if not primary_res.get("no_planting_detected") and tillage_res["tillage_detected"]:
            planting_text += " " + tillage_res["message"]
            
        confidence = primary_res.get("confidence", "medium")
        if confidence != "high" and req.ndvi_data and len(req.ndvi_data) >= 10:
            if avg_cloud_cover is not None and avg_cloud_cover < 20: confidence = "high"
            
        # OpenAI Prompting
        if primary_res.get("no_planting_detected"):
            prompt = f"Field: {req.field_name} - {req.crop}. Result: {planting_text}. Write a 2-3 sentence professional advising the farmer that NO PLANTING was seen."
        else:
            prompt = f"Field: {req.field_name} - {req.crop}. Status: {planting_text}. NDVI: {ndvi_formatted[:200]}. Rainfall: {rainfall_formatted[:100]}. Write 2-3 sentences of farming advice."
            
        ai_res = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "Professional Farm Advisor"},{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=150
        )
        
        return {
            "success": True,
            "insight": ai_res.choices[0].message.content.strip(),
            "confidence_level": confidence,
            "tillage_detected": tillage_res["tillage_detected"],
            "primary_emergence_detected": primary_res.get("primary_emergence", False),
            "planting_date_estimation": {
                **primary_res,
                "formatted_planting_window": planting_text,
                "tillage_event": tillage_res if tillage_res["tillage_detected"] else None
            }
        }

    try: return await asyncio.to_thread(sync_logic)
    except Exception as e:
        logger.error(f"Insight Error: {e}")
        raise HTTPException(500, str(e))


@app.post("/api/advanced-report")
async def advanced_report(req: AdvancedReportRequest, auth: bool = Depends(verify_auth)):
    def sync_logic():
        poly = ee.Geometry.Polygon(req.coordinates["coordinates"] if isinstance(req.coordinates, dict) else req.coordinates)
        col, size, cloud = get_optimized_collection(poly, req.start_date, req.end_date, False)
        if not col or size == 0:
            return {"success": False, "message": "No imagery available for this field in the selected date range."}
            
        latest = col.sort("system:time_start", False).first()
        idate = datetime.fromtimestamp(latest.get("system:time_start").getInfo()/1000).strftime("%Y-%m-%d")
        
        indices = {}
        for idx in ["NDVI","EVI","SAVI","NDMI","NDWI"]:
            try:
                val = get_index(latest, idx).reduceRegion(ee.Reducer.mean(), poly, 20).get(idx).getInfo()
                indices[idx] = val
            except:
                indices[idx] = 0
                
        growth = analyze_growth_stage(req.crop, req.planting_date, idate)
        
        # Add rich observation metadata for the PDF
        obs_meta = {
            "satellite_observation_date": idate,
            "date_range_start": req.start_date,
            "date_range_end": req.end_date,
            "data_source": "Sentinel-2 L2A (Multispectral)"
        }
        
        v_indices = validate_indices(indices)
        v_indices["cloud_cover"] = cloud if cloud is not None else 0
        
        report = build_report_structure({"field_name":req.field_name,"crop":req.crop,"area":req.area}, growth, v_indices)
        report["observation_metadata"] = obs_meta
        
        ai = generate_ai_analysis(report)
        final = build_report_structure({"field_name":req.field_name,"crop":req.crop,"area":req.area}, growth, v_indices, ai)
        final["observation_metadata"] = obs_meta
        
        pdf = generate_pdf_report(final)
        return {"success": True, "report": final, "pdf": {"base64": base64.b64encode(pdf.getvalue()).decode('utf-8')}}

    try: return await asyncio.to_thread(sync_logic)
    except Exception as e: raise HTTPException(500, str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
