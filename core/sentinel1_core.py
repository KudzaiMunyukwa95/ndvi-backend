
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
    # Select bands (Decibels)
    vv = image.select('VV')
    vh = image.select('VH')
    
    # Create the Ratio Band (VV / VH)
    # Note: In dB, division is subtraction
    ratio = vv.subtract(vh).rename('Ratio')
    
    # Create Composite
    # Range is roughly [-20, 0] for VV/VH in dB
    # We clamp and normalize for visualization
    return image.addBands(ratio).select(['VV', 'VH', 'Ratio'])

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
            
        # Mosaic logic: Reduce to a single image (mean or median)
        # We use median to remove speckle noise
        mosaic = collection.median().clip(geometry)
        
        # Calculate visualization bands
        vis_image = calculate_radar_visualization(mosaic)
        
        # Visualization Parameters (Min/Max in dB)
        vis_params = {
            'bands': ['VV', 'VH', 'Ratio'],
            'min': [-20, -25, 1],   # VV, VH, Ratio
            'max': [0, -5, 15],     # VV, VH, Ratio
            'gamma': 1.6            # Gamma correction for contrast
        }
        
        # Explicitly visualize to creating 8-bit RGB image
        # This fixes transparency issues by forcing server-side rendering
        rgb_image = vis_image.visualize(**vis_params)
        
        # Get MapID/TileURL
        map_id_dict = rgb_image.getMapId()
        tile_url = map_id_dict['tile_fetcher'].url_format
        
        return tile_url, collection.size().getInfo()
        
    except Exception as e:
        logger.error(f"Error generating Radar URL: {e}")
        return None, 0
