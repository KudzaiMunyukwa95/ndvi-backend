
import ee
import logging
import json
from datetime import datetime, timedelta

# Core configuration for Sentinel-1
# We use Interferometric Wide Swath (IW) mode for land monitoring
# GRD: Ground Range Detected (Standard Product)
S1_COLLECTION = "COPERNICUS/S1_GRD"

logger = logging.getLogger(__name__)

def get_s1_collection(geometry, start_date, end_date, orbit_pass=None):
    """
    Fetch filtered Sentinel-1 collection for visualization.
    
    Args:
        geometry: ee.Geometry polygon
        start_date: string 'YYYY-MM-DD'
        end_date: string 'YYYY-MM-DD'
        orbit_pass: 'ASCENDING' or 'DESCENDING' (Optional, default None = Any)
        
    Returns:
        ee.ImageCollection
    """
    try:
        # Load Sentinel-1 Collection
        s1 = ee.ImageCollection(S1_COLLECTION) \
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
            .filter(ee.Filter.eq('instrumentMode', 'IW')) \
            .filterBounds(geometry) \
            .filterDate(start_date, end_date)
            
        # Filter by orbit pass if specified
        if orbit_pass:
            s1 = s1.filter(ee.Filter.eq('orbitProperties_pass', orbit_pass))
            
        return s1
        
    except Exception as e:
        logger.error(f"Error fetching Sentinel-1 collection: {e}")
        return None

def calculate_radar_visualization(image):
    """
    Create a False-Color Composite from Radar Backscatter.
    
    Physics:
    - VV: Vertical Structure (Urban, Stems) -> RED
    - VH: Volume Scattering (Canopy Biomass) -> GREEN
    - VV/VH: Ratio (Texture/Moisture) -> BLUE
    
    This creates an image where:
    - Green = Healthy Crops
    - Red/Pink = Urban or Bare Tilled Soil
    - Blue/Black = Water
    """
    # Convert Linear to Decibels (dB)
    # 10 * log10(x)
    # Clamp to avoid -Infinity at 0
    image_db = image.clamp(0.0005, 100).log10().multiply(10.0)
    
    # Select bands (Now in Decibels)
    vv = image_db.select('VV')
    vh = image_db.select('VH')
    
    # Create the Ratio Band (VV / VH)
    # Ratio in dB = VV_dB - VH_dB (Log rules)
    ratio = vv.subtract(vh).rename('Ratio')
    
    # Create Composite
    return image_db.addBands(ratio).select(['VV', 'VH', 'Ratio'])

def get_radar_visualization_url(geometry, start_date, end_date):
    """
    Get a tile URL for the Radar Visualization Layer.
    """
    try:
        # Get collection
        collection = get_s1_collection(geometry, start_date, end_date)
        
        if collection.size().getInfo() == 0:
            logger.warning("No Sentinel-1 images found for this period")
            return None, 0
            
        # Mosaic logic: Reduce to a single image (median to remove speckle)
        mosaic = collection.median().clip(geometry)
        
        # Single-pass speckle filtering (fast but effective)
        filtered = mosaic.focalMedian(radius=1.5, kernelType='square', units='pixels')
        
        # Extract polarizations
        vv = filtered.select('VV')
        vh = filtered.select('VH')
        
        # Calculate VV/VH ratio for water/soil/vegetation distinction
        ratio = vv.divide(vh).rename('ratio')
        
        # ORIGINAL WORKING RGB - MINIMAL FINE-TUNING
        # This was the version that worked well, just slight adjustments
        
        # VV (Red): Soil - standard range
        vv_norm = vv.unitScale(-20, -5).multiply(255)
        
        # VH (Green): Vegetation - standard range (prevents green on bare soil)
        vh_norm = vh.unitScale(-28, -12).multiply(255)
        
        # Ratio (Blue): Water - slightly boosted for vivid blue
        ratio_norm = ratio.unitScale(0.3, 3).multiply(280).clamp(0, 255)
        
        # Original RGB composite that worked
        rgb_image = ee.Image.rgb(
            vv_norm,      # Red: Bare soil (brown)
            vh_norm,      # Green: Vegetation only
            ratio_norm    # Blue: Water (slightly enhanced)
        ).byte()
        
        logger.info(f"[RADAR] Original RGB with enhanced blue water")
        
        # Skip RVI metrics for performance
        mean_rvi = None
        min_rvi = None
        max_rvi = None
        health_score = None
        
        # Extract Sentinel-1 metadata from first image
        first_image = ee.Image(collection.first())
        satellite_name = first_image.get("platform_number").getInfo()  # "A" or "B"
        orbit_direction = first_image.get("orbitProperties_pass").getInfo()  # "ASCENDING" or "DESCENDING"
        instrument_mode = first_image.get("instrumentMode").getInfo()  # "IW"
        acquisition_time = first_image.date().format("YYYY-MM-dd HH:mm:ss").getInfo()
        
        logger.info(f"[RADAR METADATA] Sentinel-1{satellite_name}, Orbit: {orbit_direction}, Mode: {instrument_mode}")
        
        # Get MapID using new GEE API format
        map_id = rgb_image.getMapId()
        
        # Construct tile URL (FIXED METHOD - same as optical imagery)
        base_url = "https://earthengine.googleapis.com/v1alpha"
        mapid = map_id.get('mapid')
        tile_url = f"{base_url}/{mapid}/tiles/{{z}}/{{x}}/{{y}}"
        
        logger.info(f"[RADAR] Generated tile URL: {tile_url}")
        
        # Create metadata dictionary with RVI metrics
        metadata = {
            "name": f"Sentinel-1{satellite_name}",
            "sensor": "C-SAR",
            "orbit_direction": orbit_direction,
            "instrument_mode": instrument_mode,
            "polarization": "VV+VH",
            "acquisition_time": acquisition_time,
            "resolution": "10m",
            "platform": "Copernicus Sentinel-1",
            "processing": "Multi-band RGB with RVI metrics"
        }
        
        # Create RVI metrics dictionary for insurance decisions
        rvi_metrics = {
            "mean_rvi": round(mean_rvi, 3) if mean_rvi is not None else None,
            "min_rvi": round(min_rvi, 3) if min_rvi is not None else None,
            "max_rvi": round(max_rvi, 3) if max_rvi is not None else None,
            "health_score": health_score
        }
        
        return tile_url, collection.size().getInfo(), metadata, rvi_metrics
        
    except Exception as e:
        logger.error(f"Error generating Radar URL: {e}")
        return None, 0, None, None

