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

def apply_speckle_filter(image):
    """Multi-pass speckle filtering for clean visualization"""
    filtered = image.focalMedian(radius=3.5, kernelType='square', units='pixels')
    filtered = filtered.focalMedian(radius=2, kernelType='square', units='pixels')
    return filtered

def calculate_rvi(image):
    """RVI = 4*VH/(VV+VH)"""
    vv = image.select('VV')
    vh = image.select('VH')
    rvi = vh.multiply(4).divide(vv.add(vh).add(1e-6))
    return rvi.max(0).min(1).rename('RVI')

def calculate_water_index(image):
    """
    Water Detection Index
    Water has LOW VH and LOW VV (smooth surface)
    Higher values = more water-like
    
    Index = 1 - (VH + VV) / max(VH + VV)
    """
    vv = image.select('VV')
    vh = image.select('VH')
    
    # Sum of polarizations
    total_scatter = vv.add(vh)
    
    # Normalize to 0-1
    # Water: low values → high index
    water_index = total_scatter.unitScale(-40, -10).multiply(-1).add(1)
    water_index = water_index.max(0).min(1).rename('WATER')
    
    return water_index

def detect_field_anomalies(rvi, threshold_std=1.5):
    """
    Detect stressed/anomalous areas within field.
    Returns binary mask where 1 = anomaly
    """
    # Get field mean and std
    field_stats = rvi.reduceRegion(
        reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), None, None),
        scale=10
    ).getInfo()
    
    mean_val = float(field_stats.get('RVI_mean', 0.5))
    std_val = float(field_stats.get('RVI_stdDev', 0.1))
    
    # Threshold: mean - 1.5*std
    threshold = mean_val - (threshold_std * std_val)
    
    # Anomaly mask: pixels below threshold
    anomaly = rvi.lt(threshold).rename('ANOMALY')
    
    return anomaly, threshold, mean_val, std_val

def get_enhanced_rvi_visualization(geometry, start_date, end_date):
    """
    Generate enhanced RVI visualization with anomaly overlay.
    
    Returns:
        - Primary layer: RVI heatmap (for mapping)
        - Metadata: Full statistics for insurance
    """
    try:
        # ============================================================
        # DATA ACQUISITION
        # ============================================================
        
        collection = get_s1_collection(geometry, start_date, end_date)
        image_count = collection.size().getInfo()
        
        if image_count == 0:
            logger.warning("No Sentinel-1 images found")
            return None, None, {'status': 'ERROR', 'message': 'No data available'}
        
        logger.info(f"Found {image_count} Sentinel-1 images")
        
        # ============================================================
        # PROCESSING
        # ============================================================
        
        mosaic = collection.median().clip(geometry)
        filtered = apply_speckle_filter(mosaic)
        
        rvi = calculate_rvi(filtered)
        water = calculate_water_index(filtered)
        
        # ============================================================
        # ANOMALY DETECTION
        # ============================================================
        
        anomaly, anomaly_threshold, mean_rvi, std_rvi = detect_field_anomalies(rvi)
        
        logger.info(f"Anomaly threshold: {anomaly_threshold:.3f}, Mean: {mean_rvi:.3f}, Std: {std_rvi:.3f}")
        
        # ============================================================
        # EXTRACT FULL STATISTICS
        # ============================================================
        
        stats_dict = rvi.reduceRegion(
            reducer=ee.Reducer.mean() \
                .combine(ee.Reducer.stdDev(), None, None) \
                .combine(ee.Reducer.minMax(), None, None) \
                .combine(ee.Reducer.percentile([10, 25, 50, 75, 90]), None, None),
            geometry=geometry,
            scale=10
        ).getInfo()
        
        mean_rvi_final = float(stats_dict.get('RVI_mean', 0))
        median_rvi = float(stats_dict.get('RVI_p50', 0))
        p10 = float(stats_dict.get('RVI_p10', 0))
        p90 = float(stats_dict.get('RVI_p90', 0))
        min_rvi = float(stats_dict.get('RVI_min', 0))
        max_rvi = float(stats_dict.get('RVI_max', 0))
        std_rvi_final = float(stats_dict.get('RVI_stdDev', 0))
        
        # ============================================================
        # ANOMALY AREA CALCULATION
        # ============================================================
        
        anomaly_area_result = anomaly.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=geometry,
            scale=10
        ).getInfo()
        
        anomaly_pixels = float(anomaly_area_result.get('ANOMALY', 0))
        anomaly_area_ha = (anomaly_pixels * 100) / 10000  # Convert to hectares
        
        logger.info(f"RVI Range: {min_rvi:.3f} - {max_rvi:.3f}, Anomalies: {anomaly_area_ha:.1f} ha")
        
        # ============================================================
        # VISUALIZATION STRATEGY
        # ============================================================
        
        # KEY FIX: Use dynamic scaling based on ACTUAL data range, not fixed 0.2-0.8
        # This stretches the palette across actual RVI values in the field
        
        viz_min = max(0.0, min_rvi - 0.1)  # Add padding below minimum
        viz_max = min(1.0, max_rvi + 0.1)  # Add padding above maximum
        
        logger.info(f"Visualization scaling: min={viz_min:.3f}, max={viz_max:.3f}")
        
        # Professional agricultural palette with FULL range
        agricultural_palette = [
            '8B4513',  # Brown - Bare soil (0.0)
            'CD853F',  # Peru - Very sparse (0.17)
            'FF8C00',  # Dark orange - Sparse (0.33)
            'FFD700',  # Gold - Moderate (0.50)
            '90EE90',  # Light green - Good (0.67)
            '228B22'   # Forest green - Dense (0.83)
        ]
        
        # PRIMARY VISUALIZATION: RVI Heatmap
        # Stretches to actual data range for maximum differentiation
        rvi_viz = rvi.visualize(
            min=viz_min,
            max=viz_max,
            palette=agricultural_palette
        )
        
        logger.info(f"RVI visualization: stretching {viz_min:.3f}-{viz_max:.3f} across palette")
        
        # ============================================================
        # ALTERNATIVE VISUALIZATION: RGB Composite
        # For direct interpretation without color mapping
        # ============================================================
        
        # Convert to dB for RGB (optional, for export)
        vv = filtered.select('VV')
        vh = filtered.select('VH')
        
        vv_db = vv.clamp(0.0005, 100).log10().multiply(10.0)
        vh_db = vh.clamp(0.0005, 100).log10().multiply(10.0)
        
        # Normalize each band independently to 0-255
        vv_norm = vv_db.unitScale(-20, -5).multiply(255)
        vh_norm = vh_db.unitScale(-28, -12).multiply(255)
        
        # Create RGB: Red=VV (soil), Green=VH (vegetation), Blue=Water index
        water_norm = water.multiply(255)
        
        rgb_composite = ee.Image([vv_norm, vh_norm, water_norm]).byte()
        
        rgb_viz = rgb_composite.visualize(
            min=[0, 0, 0],
            max=[255, 255, 255]
        )
        
        logger.info("RGB composite: Red=Soil, Green=Vegetation, Blue=Water")
        
        # ============================================================
        # GENERATE TILE URLs
        # ============================================================
        
        # Primary: RVI heatmap (recommended for insurance)
        map_id_rvi = rvi_viz.getMapId()
        mapid_rvi = map_id_rvi.get('mapid')
        tile_url_rvi = f"https://earthengine.googleapis.com/v1alpha/{mapid_rvi}/tiles/{{z}}/{{x}}/{{y}}"
        
        # Alternative: RGB composite
        map_id_rgb = rgb_viz.getMapId()
        mapid_rgb = map_id_rgb.get('mapid')
        tile_url_rgb = f"https://earthengine.googleapis.com/v1alpha/{mapid_rgb}/tiles/{{z}}/{{x}}/{{y}}"
        
        # ============================================================
        # METADATA & HEALTH SCORE
        # ============================================================
        
        health_score = calculate_health_score(mean_rvi_final, std_rvi_final, p10)
        risk_level = get_risk_level(health_score)
        
        # ============================================================
        # COMPILE METADATA
        # ============================================================
        
        first_image = ee.Image(collection.first())
        satellite_name = first_image.get("platform_number").getInfo()
        orbit_direction = first_image.get("orbitProperties_pass").getInfo()
        acquisition_time = first_image.date().format("YYYY-MM-dd HH:mm:ss").getInfo()
        
        metadata = {
            'status': 'SUCCESS',
            'timestamp': datetime.now().isoformat(),
            'tile_urls': {
                'rvi_heatmap': tile_url_rvi,
                'rgb_composite': tile_url_rgb,
                'recommended': 'rvi_heatmap'
            },
            'data_source': {
                'platform': f'Sentinel-1{satellite_name}',
                'sensor': 'C-SAR',
                'orbit_direction': orbit_direction,
                'mode': 'IW',
                'resolution': '10m',
                'polarization': 'VV + VH',
                'acquisition_time': acquisition_time,
                'images_in_composite': image_count
            },
            'analysis_period': {
                'start_date': start_date,
                'end_date': end_date
            },
            'rvi_statistics': {
                'mean': round(mean_rvi_final, 4),
                'median': round(median_rvi, 4),
                'std_dev': round(std_rvi_final, 4),
                'min': round(min_rvi, 4),
                'max': round(max_rvi, 4),
                'p10': round(p10, 4),
                'p90': round(p90, 4),
                'range': round(max_rvi - min_rvi, 4)
            },
            'visualization': {
                'palette': 'Agricultural Heatmap (Brown→Dark Green)',
                'min_value': round(viz_min, 4),
                'max_value': round(viz_max, 4),
                'color_scale': {
                    'brown': 'Bare soil / Fallow',
                    'orange': 'Sparse vegetation',
                    'yellow': 'Moderate vegetation',
                    'light_green': 'Good vegetation',
                    'dark_green': 'Dense crops'
                },
                'note': f'Scaled to actual data range: {viz_min:.3f}-{viz_max:.3f}'
            },
            'anomaly_detection': {
                'threshold_rvi': round(anomaly_threshold, 4),
                'anomaly_area_ha': round(anomaly_area_ha, 2),
                'anomaly_detected': anomaly_pixels > 0,
                'severity': get_anomaly_severity(anomaly_area_ha)
            },
            'health_assessment': {
                'health_score': round(health_score, 1),
                'risk_level': risk_level,
                'expected_yield': get_expected_yield(health_score),
                'premium_factor': get_premium_factor(health_score)
            },
            'data_quality': {
                'status': 'HIGH' if image_count >= 3 else 'MEDIUM',
                'confidence': 'HIGH' if (image_count >= 2 and std_rvi_final < 0.15) else 'MEDIUM'
            }
        }
        
        logger.info(f"[SUCCESS] Health Score: {health_score:.1f}, Risk: {risk_level}")
        
        return tile_url_rvi, metadata, None
        
    except Exception as e:
        logger.error(f"Error in RVI visualization: {e}")
        import traceback
        traceback.print_exc()
        return None, None, {
            'status': 'ERROR',
            'message': str(e)
        }

def calculate_health_score(mean_rvi, std_rvi, p10):
    """Health score 0-100 based on RVI statistics"""
    mean_component = (mean_rvi / 0.6) * 50
    consistency_component = max(0, 30 - std_rvi * 100)
    coverage_component = (p10 / 0.3) * 20
    score = np.clip(mean_component + consistency_component + coverage_component, 0, 100)
    return float(score)

def get_risk_level(score):
    if score >= 80: return 'LOW'
    elif score >= 60: return 'MEDIUM'
    elif score >= 40: return 'MEDIUM_HIGH'
    elif score >= 20: return 'HIGH'
    else: return 'CRITICAL'

def get_anomaly_severity(area_ha):
    if area_ha == 0: return 'NONE'
    elif area_ha < 5: return 'LOW'
    elif area_ha < 20: return 'MEDIUM'
    elif area_ha < 50: return 'HIGH'
    else: return 'CRITICAL'

def get_expected_yield(score):
    if score >= 80: return '110%'
    elif score >= 60: return '90%'
    elif score >= 40: return '65%'
    elif score >= 20: return '30%'
    else: return '<10%'

def get_premium_factor(score):
    if score >= 80: return '0.85x (15% discount)'
    elif score >= 60: return '1.00x (standard)'
    elif score >= 40: return '1.15x (15% surcharge)'
    elif score >= 20: return '1.35x (35% surcharge)'
    else: return 'Not insurable'

# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    ee.Initialize()
    
    geometry = ee.Geometry.Polygon([
        [30.5, -17.8],
        [31.0, -17.8],
        [31.0, -17.3],
        [30.5, -17.3],
        [30.5, -17.8]
    ])
    
    print("Generating enhanced RVI visualization...")
    tile_url, metadata, error = get_enhanced_rvi_visualization(
        geometry,
        '2025-01-01',
        '2025-01-14'
    )
    
    if not error:
        print("\n✓ SUCCESS")
        print(json.dumps(metadata, indent=2))
    else:
        print(f"\n✗ ERROR: {error['message']}")
