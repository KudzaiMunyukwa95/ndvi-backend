
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
        
        # Simple VV-only visualization (more reliable than false-color)
        vv = mosaic.select('VV')
        
        # Convert to power scale for better visualization
        vv_power = ee.Image(10).pow(vv.divide(10))
        
        # Stretch to 0-255 for visualization
        vv_stretched = vv_power.unitScale(0.001, 0.5).multiply(255).byte()
        
        # Create RGB by duplicating VV to all channels (grayscale)
        rgb_image = ee.Image.rgb(vv_stretched, vv_stretched, vv_stretched)
        
        logger.info(f"[RADAR] Using simplified VV grayscale visualization")
        
        # Get MapID using new GEE API format
        map_id = rgb_image.getMapId()
        
        # Construct tile URL (FIXED METHOD - same as optical imagery)
        base_url = "https://earthengine.googleapis.com/v1alpha"
        mapid = map_id.get('mapid')
        tile_url = f"{base_url}/{mapid}/tiles/{{z}}/{{x}}/{{y}}"
        
        logger.info(f"[RADAR] Generated tile URL: {tile_url}")
        
        return tile_url, collection.size().getInfo()
        
    except Exception as e:
        logger.error(f"Error generating Radar URL: {e}")
        return None, 0

