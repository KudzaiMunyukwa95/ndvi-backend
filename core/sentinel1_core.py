import ee
import logging
import json
from datetime import datetime, timedelta

# Core configuration for Sentinel-1
S1_COLLECTION = "COPERNICUS/S1_GRD"

logger = logging.getLogger(__name__)

def get_s1_collection(geometry, start_date, end_date, orbit_pass=None):
    """
    Fetch filtered Sentinel-1 collection for visualization.
    
    Args:
        geometry: ee.Geometry polygon
        start_date: string 'YYYY-MM-DD'
        end_date: string 'YYYY-MM-DD'
        orbit_pass: 'ASCENDING' or 'DESCENDING' (Optional)
        
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


def apply_lee_sigma_filter(image, window_size=5):
    """
    Apply Lee Sigma filter for speckle reduction.
    Preserves edges while reducing noise.
    
    Args:
        image: ee.Image with VV and VH bands
        window_size: kernel size (5, 7, or 9 recommended)
    
    Returns:
        Filtered ee.Image
    """
    # Apply focal median filter twice for aggressive speckle reduction
    # This is computationally efficient in GEE
    filtered = image.focalMedian(
        radius=window_size/2,
        kernelType='square',
        units='pixels'
    )
    filtered = filtered.focalMedian(
        radius=2,
        kernelType='square',
        units='pixels'
    )
    
    return filtered


def calculate_rvi(image):
    """
    Calculate Radar Vegetation Index (RVI).
    
    RVI = 4 * VH / (VV + VH)
    
    - Ranges from 0 to 1
    - Higher values = denser vegetation
    - VH: sensitive to vegetation structure
    - VV: sensitive to surface moisture
    
    Args:
        image: ee.Image with VV and VH bands (linear scale)
    
    Returns:
        RVI as ee.Image (0-1 range)
    """
    vv = image.select('VV')
    vh = image.select('VH')
    
    # RVI formula
    rvi = vh.multiply(4).divide(vv.add(vh)).rename('RVI')
    
    # Clip to 0-1 range
    rvi = rvi.max(0).min(1)
    
    return rvi


def get_radar_visualization_url(geometry, start_date, end_date):
    """
    Generate RVI visualization tile URL for mapping.
    
    Returns:
        tuple: (tile_url, image_count, metadata)
    """
    try:
        # Get collection
        collection = get_s1_collection(geometry, start_date, end_date)
        
        if collection.size().getInfo() == 0:
            logger.warning("No Sentinel-1 images found for this period")
            return None, 0, None
        
        logger.info(f"Found {collection.size().getInfo()} Sentinel-1 images")
        
        # Mosaic (median to reduce speckle naturally)
        mosaic = collection.median().clip(geometry)
        
        # Apply speckle filtering
        logger.info("Applying Lee Sigma filter (7x7 window)...")
        filtered = apply_lee_sigma_filter(mosaic, window_size=7)
        
        # Calculate RVI
        logger.info("Calculating Radar Vegetation Index (RVI)...")
        rvi = calculate_rvi(filtered)
        
        # Get additional indices for context
        vv = filtered.select('VV')
        vh = filtered.select('VH')
        ratio = vv.divide(vh).rename('ratio')
        
        # ============================================================
        # VISUALIZATION: RVI as Single-Band Heatmap
        # Green (0.6-1.0) = Dense vegetation
        # Yellow (0.4-0.6) = Good vegetation
        # Orange (0.2-0.4) = Moderate vegetation
        # Red (0.0-0.2) = Sparse/bare soil
        # ============================================================
        
        # Visualize RVI with a green-to-red palette for agricultural interpretation
        rvi_viz = rvi.visualize(
            min=0.0,
            max=1.0,
            palette=[
                '#8B4513',  # Brown (0.0) - Bare soil
                '#FF4500',  # Red-orange (0.2) - Sparse
                '#FFD700',  # Gold (0.4) - Moderate
                '#90EE90',  # Light green (0.6) - Good
                '#006400'   # Dark green (1.0) - Dense
            ]
        )
        
        logger.info("[VISUALIZATION] RVI Heatmap: Brown→Red→Yellow→Light Green→Dark Green")
        logger.info("Interpretation: Brown=Bare, Red=Sparse, Yellow=Moderate, Green=Good/Dense")
        
        # ============================================================
        # ALTERNATIVE: Multi-band RGB Composite
        # ============================================================
        # If you want to see VV, VH, Ratio together:
        # Create normalized bands for RGB display
        
        vv_db = vv.clamp(0.0005, 100).log10().multiply(10.0)  # Convert to dB
        vh_db = vh.clamp(0.0005, 100).log10().multiply(10.0)
        
        vv_norm = vv_db.unitScale(-20, -5).multiply(255)
        vh_norm = vh_db.unitScale(-28, -12).multiply(255)
        ratio_norm = ratio.unitScale(0.3, 3).multiply(255)
        
        # Create RGB: Red=VV (soil), Green=VH (vegetation), Blue=Ratio (water/structure)
        rgb_composite = ee.Image([vv_norm, vh_norm, ratio_norm]).rename(['VV', 'VH', 'Ratio'])
        
        rgb_viz = rgb_composite.visualize(
            min=[0, 0, 0],
            max=[255, 255, 255],
            gamma=[1.1, 1.0, 1.1]
        )
        
        logger.info("[ALTERNATIVE] RGB Composite: Red=Soil Moisture, Green=Vegetation, Blue=Water/Structure")
        
        # ============================================================
        # USE RVI FOR PRIMARY VISUALIZATION (Recommended for Insurance)
        # ============================================================
        
        # Get MapID using GEE API v1alpha
        map_id = rvi_viz.getMapId()
        
        # Construct tile URL
        base_url = "https://earthengine.googleapis.com/v1alpha"
        mapid = map_id.get('mapid')
        tile_url = f"{base_url}/{mapid}/tiles/{{z}}/{{x}}/{{y}}"
        
        logger.info(f"[SUCCESS] Generated RVI tile URL")
        logger.info(f"Tile URL: {tile_url}")
        
        # Extract metadata from first image
        first_image = ee.Image(collection.first())
        satellite_name = first_image.get("platform_number").getInfo()
        orbit_direction = first_image.get("orbitProperties_pass").getInfo()
        instrument_mode = first_image.get("instrumentMode").getInfo()
        acquisition_time = first_image.date().format("YYYY-MM-dd HH:mm:ss").getInfo()
        
        logger.info(f"[METADATA] Sentinel-1{satellite_name} | Orbit: {orbit_direction} | Mode: {instrument_mode}")
        
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
            "processing": "RVI with Lee Sigma filtering",
            "index": "Radar Vegetation Index (RVI)",
            "rvi_formula": "4 * VH / (VV + VH)",
            "interpretation": {
                "0.0-0.2": "Bare soil / Water",
                "0.2-0.4": "Sparse vegetation",
                "0.4-0.6": "Moderate vegetation",
                "0.6-1.0": "Dense vegetation"
            }
        }
        
        image_count = collection.size().getInfo()
        
        return tile_url, image_count, metadata
        
    except Exception as e:
        logger.error(f"Error generating Radar URL: {e}")
        import traceback
        traceback.print_exc()
        return None, 0, None


# ============================================================
# BONUS: Calculate statistics for a specific field
# ============================================================

def extract_rvi_statistics(geometry, start_date, end_date):
    """
    Extract RVI statistics for a field polygon.
    Useful for insurance assessment.
    
    Args:
        geometry: ee.Geometry (field boundary)
        start_date: 'YYYY-MM-DD'
        end_date: 'YYYY-MM-DD'
    
    Returns:
        dict with RVI stats
    """
    try:
        collection = get_s1_collection(geometry, start_date, end_date)
        
        if collection.size().getInfo() == 0:
            return None
        
        mosaic = collection.median().clip(geometry)
        filtered = apply_lee_sigma_filter(mosaic, window_size=7)
        rvi = calculate_rvi(filtered)
        
        # Calculate statistics
        stats = rvi.reduceRegion(
            reducer=ee.Reducer.mean().combine(
                ee.Reducer.stdDev(), None, None
            ).combine(
                ee.Reducer.percentile([10, 25, 50, 75, 90]), None, None
            ),
            geometry=geometry,
            scale=10  # 10m resolution
        ).getInfo()
        
        return {
            'mean_rvi': stats.get('RVI_mean', 0),
            'std_rvi': stats.get('RVI_stdDev', 0),
            'percentile_10': stats.get('RVI_p10', 0),
            'percentile_50': stats.get('RVI_p50', 0),  # Median
            'percentile_90': stats.get('RVI_p90', 0),
            'min': stats.get('RVI_p10', 0),
            'max': stats.get('RVI_p90', 0)
        }
        
    except Exception as e:
        logger.error(f"Error extracting RVI statistics: {e}")
        return None


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    # Initialize Earth Engine
    ee.Authenticate()  # Only needed first time
    ee.Initialize()
    
    # Define geometry (example: Chivero, Zimbabwe)
    geometry = ee.Geometry.Polygon([
        [30.5, -17.8],
        [31.0, -17.8],
        [31.0, -17.3],
        [30.5, -17.3],
        [30.5, -17.8]
    ])
    
    # Date range
    start_date = '2025-01-01'
    end_date = '2025-01-14'
    
    # Generate visualization
    print("Generating RVI visualization...")
    tile_url, count, metadata = get_radar_visualization_url(geometry, start_date, end_date)
    
    if tile_url:
        print(f"\n✓ Success!")
        print(f"Images found: {count}")
        print(f"Tile URL: {tile_url}")
        print(f"\nMetadata: {json.dumps(metadata, indent=2)}")
        
        # Get field statistics
        print("\nExtracting field statistics...")
        stats = extract_rvi_statistics(geometry, start_date, end_date)
        if stats:
            print(f"Mean RVI: {stats['mean_rvi']:.3f}")
            print(f"Std Dev: {stats['std_rvi']:.3f}")
            print(f"Range: {stats['min']:.3f} - {stats['max']:.3f}")
    else:
        print("✗ Failed to generate visualization")
