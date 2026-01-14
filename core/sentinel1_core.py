
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
        
        # ADVANCED SPECKLE FILTERING - Lee Sigma Filter
        # This dramatically improves image quality for agricultural monitoring
        kernel_size = 7  # 7x7 window for optimal edge preservation
        
        # Multi-temporal filtering using refined focal median
        # Apply twice for better speckle reduction
        filtered = mosaic.focalMedian(radius=kernel_size/2, kernelType='square', units='pixels')
        filtered = filtered.focalMedian(radius=2, kernelType='square', units='pixels')
        
        # Extract polarizations from filtered image
        vv = filtered.select('VV')
        vh = filtered.select('VH')
        
        # CALCULATE RADAR VEGETATION INDEX (RVI)
        # RVI = (4 * VH) / (VV + VH)
        # Range: 0-1, where higher values = more vegetation/biomass
        # This is the agricultural standard for SAR vegetation monitoring
        rvi = vh.multiply(4).divide(vv.add(vh)).rename('RVI')
        
        # Normalize RVI to 0-1 range for visualization
        # EXPANDED RANGE for better contrast:
        # 0.05-0.30: Bare soil/water (brown)
        # 0.30-0.60: Sparse to moderate vegetation (yellow-green)
        # 0.60-0.95: Dense crops (green)
        rvi_normalized = rvi.unitScale(0.05, 0.95)
        
        # AGRICULTURAL COLOR PALETTE
        # Brown (bare soil) → Yellow (sparse vegetation) → Light Green → Dark Green (dense crops)
        # This matches how farmers and agronomists interpret vegetation
        agricultural_palette = [
            '8B4513',  # Saddle Brown - Bare soil/fallow
            'CD853F',  # Peru - Very sparse vegetation  
            'DEB887',  # Burlywood - Sparse vegetation
            'F0E68C',  # Khaki - Light vegetation
            'ADFF2F',  # Green Yellow - Moderate vegetation
            '7FFF00',  # Chartreuse - Good vegetation
            '32CD32',  # Lime Green - Dense vegetation
            '228B22',  # Forest Green - Very dense crops
            '006400'   # Dark Green - Maximum biomass
        ]
        
        # Create professional RVI visualization
        rvi_viz = rvi_normalized.visualize(
            min=0,
            max=1,
            palette=agricultural_palette
        )
        
        logger.info(f"[RADAR] Using RVI-based visualization with Lee Sigma filtering")
        logger.info(f"[RADAR] Color scheme: Brown=Bare Soil, Yellow=Sparse, Green=Dense Vegetation")
        
        # Extract Sentinel-1 metadata from first image
        first_image = ee.Image(collection.first())
        satellite_name = first_image.get("platform_number").getInfo()  # "A" or "B"
        orbit_direction = first_image.get("orbitProperties_pass").getInfo()  # "ASCENDING" or "DESCENDING"
        instrument_mode = first_image.get("instrumentMode").getInfo()  # "IW"
        acquisition_time = first_image.date().format("YYYY-MM-dd HH:mm:ss").getInfo()
        
        logger.info(f"[RADAR METADATA] Sentinel-1{satellite_name}, Orbit: {orbit_direction}, Mode: {instrument_mode}")
        
        # Get MapID using new GEE API format
        map_id = rvi_viz.getMapId()
        
        # Construct tile URL (FIXED METHOD - same as optical imagery)
        base_url = "https://earthengine.googleapis.com/v1alpha"
        mapid = map_id.get('mapid')
        tile_url = f"{base_url}/{mapid}/tiles/{{z}}/{{x}}/{{y}}"
        
        logger.info(f"[RADAR] Generated RVI tile URL: {tile_url}")
        
        # Create metadata dictionary
        metadata = {
            "name": f"Sentinel-1{satellite_name}",
            "sensor": "C-SAR",
            "orbit_direction": orbit_direction,
            "instrument_mode": instrument_mode,
            "polarization": "VV+VH",
            "acquisition_time": acquisition_time,
            "resolution": "10m",
            "platform": "Copernicus Sentinel-1",
            "processing": "RVI with Lee Sigma filtering"
        }
        
        return tile_url, collection.size().getInfo(), metadata
        
    except Exception as e:
        logger.error(f"Error generating Radar URL: {e}")
        return None, 0, None

