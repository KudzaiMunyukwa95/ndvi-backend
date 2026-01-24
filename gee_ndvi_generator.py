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
        "range": [-0.2, 0.6],
        "palette": ["#fdae61", "#ffffbf", "#a6d96a", "#1a9850"],
        "explanation": "Canopy moisture index: yellow = dry canopy, light green = moderate moisture, dark green = moist/saturated canopy."
    },
    "NDWI": {
        "range": [0.05, 0.4],
        "palette": ["#fff7bc", "#c7e9b4", "#7fcdbb", "#41b6c4", "#1d91c0", "#0c2c84"],
        "explanation": "Water detection index: yellow/green = dry or vegetated areas, blue = water bodies or very high moisture."
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

# --- Core Earth Engine Logic (Ported from Flask) ---

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
        return ndvi.where(ndvi.lt(0), 0)
    elif index_type == "EVI":
        nir = image.select('B8').divide(10000)
        red = image.select('B4').divide(10000)
        blue = image.select('B2').divide(10000)
        evi = image.expression('2.5 * ((NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1))', {'NIR': nir, 'RED': red, 'BLUE': blue}).rename('EVI')
        return evi.where(evi.gt(1), 1).where(evi.lt(0), 0)
    elif index_type == "SAVI":
        nir = image.select('B8').divide(10000)
        red = image.select('B4').divide(10000)
        savi = image.expression('((NIR - RED) * (1 + L)) / (NIR + RED + L)', {'NIR': nir, 'RED': red, 'L': 0.5}).rename('SAVI')
        return savi.where(savi.lt(0), 0)
    elif index_type == "NDMI":
        return image.normalizedDifference(['B8', 'B11']).rename('NDMI')
    elif index_type == "NDWI":
        ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
        config = INDEX_CONFIGS["NDWI"]
        return ndwi.where(ndwi.lt(config["range"][0]), config["range"][0]).where(ndwi.gt(config["range"][1]), config["range"][1])
    return image.normalizedDifference(['B8', 'B4']).rename('NDVI')

def smooth_ndvi_series(ndvi_data, window=SMOOTH_WINDOW):
    if len(ndvi_data) < window: return ndvi_data
    smoothed_data = []
    sorted_data = sorted(ndvi_data, key=lambda x: x['date'])
    for i in range(len(sorted_data)):
        if i == 0 or i == len(sorted_data) - 1:
            smoothed_data.append(sorted_data[i].copy())
        else:
            window_values = [sorted_data[i-1]['ndvi'], sorted_data[i]['ndvi'], sorted_data[i+1]['ndvi']]
            smoothed_point = sorted_data[i].copy()
            smoothed_point['ndvi'] = statistics.median(window_values)
            smoothed_data.append(smoothed_point)
    return smoothed_data

def is_winter_season(start_date, end_date, coordinates):
    try:
        start_obj = datetime.strptime(start_date, '%Y-%m-%d')
        end_obj = datetime.strptime(end_date, '%Y-%m-%d')
        lat = coordinates[0][0][1] if coordinates and len(coordinates) > 0 else -19.0
        winter_months = [4, 5, 6, 7, 8] if lat < 0 else [11, 12, 1, 2, 3]
        curr = start_obj
        while curr <= end_obj:
            if curr.month in winter_months: return True
            curr = (curr.replace(day=1) + timedelta(days=32)).replace(day=1)
        return False
    except: return False

def get_geohash_key(coords, crop, year):
    try:
        lats = [c[1] for c in coords[0]]
        lons = [c[0] for c in coords[0]]
        gh = geohash2.encode(sum(lats)/len(lats), sum(lons)/len(lons), precision=5)
        return gh, f"{gh}_{crop}_{year}"
    except: return None, None

def detect_wheat_winter_emergence(ndvi_data, coordinates=None, force=False):
    if not force:
        start = min(item['date'] for item in ndvi_data)
        end = max(item['date'] for item in ndvi_data)
        if not is_winter_season(start, end, coordinates): return None, None, {}
    
    if len(ndvi_data) < 4: return None, "low", {"qa": {"valid": False, "reason": "sparse"}}
    
    smoothed = smooth_ndvi_series(ndvi_data)
    ndvi_vals = [item['ndvi'] for item in smoothed]
    amplitude = max(ndvi_vals) - min(ndvi_vals)
    if amplitude < NDVI_AMPLITUDE_MIN: return None, "low", {"qa": {"valid": False, "reason": "low_signal"}}
    
    sorted_data = sorted(smoothed, key=lambda x: x['date'])
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
                
    if not candidates: return None, "low", {"qa": {"valid": True}}
    candidates.sort(key=lambda x: x['date'])
    best = candidates[0]
    return best['date'], best['confidence'], {"detection_method": best['method'], "cloud_at_emergence_pct": best['cloud_pct']}

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

def get_optimized_collection(polygon, start, end, limit_images=True):
    base = ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filterBounds(polygon).filterDate(start, end)
    size = base.size().getInfo()
    if size == 0: return None, 0, None
    threshold = 10 if size > 50 else (20 if size > 20 else 80)
    max_img = 15 if size > 50 else (20 if size > 20 else size)
    col = base.filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", threshold)).sort("CLOUDY_PIXEL_PERCENTAGE")
    if limit_images: col = col.limit(max_img)
    final_size = col.size().getInfo()
    avg_cloud = calculate_collection_cloud_cover(col, polygon, start, end)
    return col, final_size, avg_cloud.getInfo() if avg_cloud else None

def format_date_for_display(date_str):
    try: return datetime.strptime(date_str, '%Y-%m-%d').strftime('%B %d')
    except: return date_str

def detect_primary_emergence_and_planting(ndvi_data, crop_type, irrigated, rainfall_data=None, coords=None, force=False):
    if crop_type.lower() == 'wheat':
        date, conf, meta = detect_wheat_winter_emergence(ndvi_data, coords, force)
        if date:
            win = EMERGENCE_WINDOWS.get('Wheat', (3, 6))
            p_end = (datetime.strptime(date, '%Y-%m-%d') - timedelta(days=win[0])).strftime('%Y-%m-%d')
            p_start = (datetime.strptime(date, '%Y-%m-%d') - timedelta(days=win[1])).strftime('%Y-%m-%d')
            msg = f"Winter wheat emergence detected near {format_date_for_display(date)}, planting likely between {format_date_for_display(p_start)} and {format_date_for_display(p_end)}."
            res = {"emergenceDate": date, "plantingWindowStart": p_start, "plantingWindowEnd": p_end, "confidence": conf, "message": msg, "primary_emergence": True, "detection_method": "wheat_winter_detector"}
            res.update(meta)
            return res

    sorted_ndvi = sorted(ndvi_data, key=lambda x: x['date'])
    date, idx = None, -1
    for i in range(len(sorted_ndvi) - 1):
        if sorted_ndvi[i]['ndvi'] < EMERGENCE_THRESHOLD and sorted_ndvi[i+1]['ndvi'] >= EMERGENCE_THRESHOLD:
            date, idx = sorted_ndvi[i+1]['date'], i+1
            break
    if not date:
        for i in range(len(sorted_ndvi) - 1):
            if sorted_ndvi[i]['ndvi'] < 0.15 and sorted_ndvi[i+1]['ndvi'] > sorted_ndvi[i]['ndvi'] + 0.05:
                date, idx = sorted_ndvi[i+1]['date'], i+1
                break
    if not date: return {"emergenceDate": None, "no_planting_detected": True, "message": "No planting detected.", "confidence": "high"}
    
    win = EMERGENCE_WINDOWS.get(crop_type, DEFAULT_EMERGENCE_WINDOW)
    p_end = (datetime.strptime(date, '%Y-%m-%d') - timedelta(days=win[0])).strftime('%Y-%m-%d')
    p_start = (datetime.strptime(date, '%Y-%m-%d') - timedelta(days=win[1])).strftime('%Y-%m-%d')
    msg = f"Primary emergence detected near {format_date_for_display(date)}, planting likely between {format_date_for_display(p_start)} and {format_date_for_display(p_end)}."
    return {"emergenceDate": date, "plantingWindowStart": p_start, "plantingWindowEnd": p_end, "message": msg, "confidence": "medium" if len(sorted_ndvi) < 6 else "high", "primary_emergence": True}

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
        col, size, cloud = get_optimized_collection(poly, req.startDate, req.endDate)
        if not col or size == 0: raise Exception("No imagery found")
        img = col.median().clip(poly)
        if req.index_type == "RGB":
            vis = img.select(["B4","B3","B2"]).visualize(min=0, max=3000).reproject(crs='EPSG:3857', scale=30)
        else:
            idx_img = get_index(img, req.index_type)
            conf = INDEX_CONFIGS[req.index_type]
            vis = idx_img.visualize(min=conf["range"][0], max=conf["range"][1], palette=conf["palette"]).reproject(crs='EPSG:3857', scale=10)
        
        mid = ee.data.getMapId({"image": vis})
        tile_url = f"https://earthengine.googleapis.com/v1alpha/{mid['mapid']}/tiles/{{z}}/{{x}}/{{y}}"
        res = {"success": True, "index": req.index_type, "tile_url": tile_url, "cloud_cover": cloud, "collection_size": size, "image_date": col.first().date().format("YYYY-MM-dd").getInfo()}
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
        col, size, _ = get_optimized_collection(poly, req.startDate, req.endDate, limit_images=False)
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
        res = detect_primary_emergence_and_planting(req.ndvi_data, req.crop, "Yes" if req.irrigated else "No", req.rainfall_data, req.coordinates, req.forceWinterDetector)
        prompt = f"Field: {req.field_name}, Crop: {req.crop}, Result: {res['message']}. Write 2 sentences of farm advice."
        ai_res = openai_client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"system","content":"Advisor"},{"role":"user","content":prompt}], temperature=0.1)
        return {"success": True, "insight": ai_res.choices[0].message.content, "planting_date_estimation": res}

    try: return await asyncio.to_thread(sync_logic)
    except Exception as e: raise HTTPException(500, str(e))

@app.post("/api/advanced-report")
async def advanced_report(req: AdvancedReportRequest, auth: bool = Depends(verify_auth)):
    def sync_logic():
        poly = ee.Geometry.Polygon(req.coordinates["coordinates"] if isinstance(req.coordinates, dict) else req.coordinates)
        col, size, cloud = get_optimized_collection(poly, req.start_date, req.end_date, False)
        latest = col.sort("system:time_start", False).first()
        idate = datetime.fromtimestamp(latest.get("system:time_start").getInfo()/1000).strftime("%Y-%m-%d")
        
        indices = {idx: get_index(latest, idx).reduceRegion(ee.Reducer.mean(), poly, 20).get(idx).getInfo() for idx in ["NDVI","EVI","SAVI","NDMI","NDWI"]}
        growth = analyze_growth_stage(req.crop, req.planting_date, idate)
        report = build_report_structure({"field_name":req.field_name,"crop":req.crop,"area":req.area}, growth, validate_indices(indices))
        ai = generate_ai_analysis(report)
        final = build_report_structure({"field_name":req.field_name,"crop":req.crop,"area":req.area}, growth, validate_indices(indices), ai)
        pdf = generate_pdf_report(final)
        return {"success": True, "report": final, "pdf": {"base64": base64.b64encode(pdf.getvalue()).decode('utf-8')}}

    try: return await asyncio.to_thread(sync_logic)
    except Exception as e: raise HTTPException(500, str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
