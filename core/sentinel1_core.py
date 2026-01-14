import ee
import logging
import json
from datetime import datetime
import numpy as np

S1_COLLECTION = "COPERNICUS/S1_GRD"
logger = logging.getLogger(__name__)

def get_s1_collection(geometry, start_date, end_date, orbit_pass=None):
    """Fetch filtered Sentinel-1 collection"""
    try:
        s1 = ee.ImageCollection(S1_COLLECTION) \
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
            .filter(ee.Filter.eq('instrumentMode', 'IW')) \
            .filterBounds(geometry) \
            .filterDate(start_date, end_date)
        
        if orbit_pass:
            s1 = s1.filter(ee.Filter.eq('orbitProperties_pass', orbit_pass))
        
        return s1
    except Exception as e:
        logger.error(f"Error fetching Sentinel-1: {e}")
        return None

def apply_speckle_filter(image, window_size=5):
    """
    Apply speckle reduction with edge preservation.
    
    This is the Lee Sigma filter equivalent in GEE:
    - Reduces multiplicative speckle noise
    - Preserves agricultural field boundaries
    - Essential for accurate RVI calculation
    """
    filtered = image.focalMedian(
        radius=window_size/2,
        kernelType='square',
        units='pixels'
    )
    # Second pass for stronger noise reduction
    filtered = filtered.focalMedian(radius=2, kernelType='square', units='pixels')
    return filtered

def calculate_rvi(image):
    """
    Calculate Radar Vegetation Index = 4*VH/(VV+VH)
    
    Range: 0 to 1
    - Low values (0.0-0.3): Bare soil, water, sparse vegetation
    - Mid values (0.3-0.6): Growing crops, moderate vegetation
    - High values (0.6-1.0): Dense vegetation, mature crops
    
    This is the PRIMARY metric for agricultural monitoring.
    """
    vv = image.select('VV')
    vh = image.select('VH')
    
    # RVI formula with small epsilon to avoid division by zero
    rvi = vh.multiply(4).divide(vv.add(vh).add(1e-6))
    
    # Clip to valid range [0, 1]
    return rvi.max(0).min(1).rename('RVI')

def get_rvi_visualization(geometry, start_date, end_date):
    """
    Generate RVI heatmap visualization with professional color palette.
    
    Returns:
        tile_url: URL for mapping library
        metadata: Comprehensive metadata including RVI statistics
    """
    try:
        # ============================================================
        # 1. DATA ACQUISITION & QUALITY CHECK
        # ============================================================
        
        collection = get_s1_collection(geometry, start_date, end_date)
        image_count = collection.size().getInfo()
        
        if image_count == 0:
            logger.warning("No Sentinel-1 images found")
            return None, None, {
                'status': 'ERROR',
                'message': 'No Sentinel-1 data available for this period',
                'data_quality': 'INSUFFICIENT'
            }
        
        logger.info(f"Found {image_count} Sentinel-1 images")
        
        # ============================================================
        # 2. MOSAIC & FILTERING
        # ============================================================
        
        # Use median composite to reduce speckle naturally
        mosaic = collection.median().clip(geometry)
        
        # Apply professional speckle filtering
        filtered = apply_speckle_filter(mosaic, window_size=5)
        
        # ============================================================
        # 3. RVI CALCULATION
        # ============================================================
        
        rvi = calculate_rvi(filtered)
        
        # ============================================================
        # 4. EXTRACT STATISTICS (Critical for Insurance)
        # ============================================================
        
        stats_dict = rvi.reduceRegion(
            reducer=ee.Reducer.mean() \
                .combine(ee.Reducer.stdDev(), None, None) \
                .combine(ee.Reducer.minMax(), None, None) \
                .combine(ee.Reducer.percentile([10, 25, 50, 75, 90]), None, None) \
                .combine(ee.Reducer.count(), None, None),
            geometry=geometry,
            scale=10
        ).getInfo()
        
        # Parse statistics
        mean_rvi = float(stats_dict.get('RVI_mean', 0))
        std_rvi = float(stats_dict.get('RVI_stdDev', 0))
        min_rvi = float(stats_dict.get('RVI_min', 0))
        max_rvi = float(stats_dict.get('RVI_max', 0))
        median_rvi = float(stats_dict.get('RVI_p50', 0))
        p10 = float(stats_dict.get('RVI_p10', 0))
        p90 = float(stats_dict.get('RVI_p90', 0))
        pixel_count = int(stats_dict.get('RVI_count', 0))
        
        logger.info(f"RVI Stats: mean={mean_rvi:.3f}, std={std_rvi:.3f}, range=[{min_rvi:.3f}, {max_rvi:.3f}]")
        
        # ============================================================
        # 5. PROFESSIONAL VISUALIZATION
        # ============================================================
        
        # RVI Heatmap with agricultural color palette
        # Brown → Red → Orange → Yellow → Light Green → Dark Green
        rvi_viz = rvi.visualize(
            min=0.0,
            max=1.0,
            palette=[
                '#8B4513',  # Brown (0.0) - Bare soil
                '#CD5C5C',  # Indian Red (0.2) - Sparse vegetation
                '#FF8C00',  # Dark Orange (0.4) - Moderate vegetation
                '#FFD700',  # Gold (0.6) - Good vegetation
                '#90EE90',  # Light Green (0.8) - Dense vegetation
                '#006400'   # Dark Green (1.0) - Very dense vegetation
            ]
        )
        
        logger.info("RVI Visualization: Brown→Red→Orange→Yellow→Light Green→Dark Green")
        
        # ============================================================
        # 6. GENERATE TILE URL
        # ============================================================
        
        map_id = rvi_viz.getMapId()
        base_url = "https://earthengine.googleapis.com/v1alpha"
        mapid = map_id.get('mapid')
        tile_url = f"{base_url}/{mapid}/tiles/{{z}}/{{x}}/{{y}}"
        
        # ============================================================
        # 7. EXTRACT METADATA
        # ============================================================
        
        first_image = ee.Image(collection.first())
        satellite_name = first_image.get("platform_number").getInfo()
        orbit_direction = first_image.get("orbitProperties_pass").getInfo()
        acquisition_time = first_image.date().format("YYYY-MM-dd HH:mm:ss").getInfo()
        
        # ============================================================
        # 8. CALCULATE HEALTH SCORE
        # ============================================================
        
        health_score = calculate_health_score(mean_rvi, std_rvi, p10)
        risk_level = get_risk_level(health_score)
        
        # ============================================================
        # 9. COMPILE METADATA
        # ============================================================
        
        metadata = {
            'status': 'SUCCESS',
            'timestamp': datetime.now().isoformat(),
            'tile_url': tile_url,
            'data_source': {
                'platform': f'Sentinel-1{satellite_name}',
                'sensor': 'C-SAR',
                'orbit_direction': orbit_direction,
                'mode': 'IW (Interferometric Wide Swath)',
                'resolution': '10m',
                'polarization': 'VV + VH (Dual-polarization)',
                'acquisition_time': acquisition_time,
                'images_in_composite': image_count
            },
            'analysis_period': {
                'start_date': start_date,
                'end_date': end_date,
                'days': (datetime.fromisoformat(end_date) - datetime.fromisoformat(start_date)).days
            },
            'rvi_statistics': {
                'mean': round(mean_rvi, 4),
                'median': round(median_rvi, 4),
                'std_dev': round(std_rvi, 4),
                'min': round(min_rvi, 4),
                'max': round(max_rvi, 4),
                'p10': round(p10, 4),
                'p90': round(p90, 4),
                'pixel_count': pixel_count,
                'interpretation': {
                    'mean_class': classify_rvi_value(mean_rvi),
                    'coverage_quality': 'Good' if p10 > 0.2 else 'Fair' if p10 > 0.1 else 'Poor'
                }
            },
            'health_assessment': {
                'health_score': round(health_score, 1),
                'risk_level': risk_level,
                'expected_yield_percent': get_expected_yield(health_score),
                'premium_factor': get_premium_factor(health_score)
            },
            'data_quality': {
                'status': 'HIGH' if image_count >= 3 else 'MEDIUM' if image_count >= 1 else 'LOW',
                'confidence': 'HIGH' if (image_count >= 3 and std_rvi < 0.15) else 'MEDIUM' if image_count >= 1 else 'LOW',
                'warning': get_quality_warning(image_count, std_rvi)
            },
            'visualization': {
                'palette': 'RVI Heatmap (Brown→Dark Green)',
                'min_value': 0.0,
                'max_value': 1.0,
                'color_interpretation': {
                    '0.0-0.2': 'Bare/Water (Brown)',
                    '0.2-0.4': 'Sparse Vegetation (Red-Orange)',
                    '0.4-0.6': 'Moderate Vegetation (Orange-Yellow)',
                    '0.6-0.8': 'Good Vegetation (Light Green)',
                    '0.8-1.0': 'Dense Vegetation (Dark Green)'
                }
            }
        }
        
        logger.info(f"[SUCCESS] Health Score: {health_score:.1f}, Risk: {risk_level}")
        
        return tile_url, metadata, None  # No error
        
    except Exception as e:
        logger.error(f"Error in RVI visualization: {e}")
        return None, None, {
            'status': 'ERROR',
            'message': str(e),
            'data_quality': 'FAILED'
        }

def calculate_health_score(mean_rvi, std_rvi, p10):
    """
    Convert RVI statistics to insurance health score (0-100).
    
    Scoring factors:
    - 50% weight: Mean RVI (primary indicator)
    - 30% weight: Consistency (low std dev = better)
    - 20% weight: Minimum pixels (p10 = worst 10%, indicates patches)
    """
    mean_component = (mean_rvi / 0.6) * 50
    consistency_component = max(0, 30 - std_rvi * 100)
    coverage_component = (p10 / 0.3) * 20
    
    score = np.clip(mean_component + consistency_component + coverage_component, 0, 100)
    return float(score)

def get_risk_level(score):
    """Map health score to risk category"""
    if score >= 80: return 'LOW'
    elif score >= 60: return 'MEDIUM'
    elif score >= 40: return 'MEDIUM_HIGH'
    elif score >= 20: return 'HIGH'
    else: return 'CRITICAL'

def classify_rvi_value(rvi):
    """Classify single RVI value"""
    if rvi < 0.2: return 'Bare/Water'
    elif rvi < 0.4: return 'Sparse'
    elif rvi < 0.6: return 'Moderate'
    elif rvi < 0.8: return 'Good'
    else: return 'Dense'

def get_expected_yield(score):
    """Expected yield as % of normal for given health score"""
    if score >= 80: return '110%'
    elif score >= 60: return '90%'
    elif score >= 40: return '65%'
    elif score >= 20: return '30%'
    else: return '<10%'

def get_premium_factor(score):
    """Insurance premium adjustment"""
    if score >= 80: return '0.85x (15% discount)'
    elif score >= 60: return '1.00x (standard)'
    elif score >= 40: return '1.15x (15% surcharge)'
    elif score >= 20: return '1.35x (35% surcharge)'
    else: return 'Not insurable'

def get_quality_warning(image_count, std_rvi):
    """Generate data quality warnings for underwriter"""
    warnings = []
    if image_count < 2:
        warnings.append('Only 1 image: Limited temporal confidence')
    if std_rvi > 0.2:
        warnings.append('High variability: Field may have mixed conditions')
    return warnings if warnings else None

# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    ee.Initialize()
    
    # Example: New Beginnings field
    geometry = ee.Geometry.Polygon([
        [30.5, -17.8],
        [31.0, -17.8],
        [31.0, -17.3],
        [30.5, -17.3],
        [30.5, -17.8]
    ])
    
    print("Generating RVI visualization...")
    tile_url, metadata, error = get_rvi_visualization(
        geometry,
        '2025-01-01',
        '2025-01-14'
    )
    
    if not error:
        print("\n✓ SUCCESS")
        print(json.dumps(metadata, indent=2))
    else:
        print(f"\n✗ ERROR: {error['message']}")
