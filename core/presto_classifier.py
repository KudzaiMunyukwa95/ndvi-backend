"""
Presto-based Crop Classification Module
Uses NASA Harvest's Presto foundation model for satellite-based crop type detection
"""

import numpy as np
import ee
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class CropClassifier:
    """
    Crop classification using Presto foundation model
    """
    
    def __init__(self):
        logger.info("Initializing Crop Classifier (rule-based mode)")
        
        # Crop type mapping (Zimbabwe-specific)
        self.crop_labels = {
            0: "Maize",
            1: "Wheat", 
            2: "Tobacco",
            3: "Cotton",
            4: "Sorghum",
            5: "Soybean",
            6: "Groundnuts",
            7: "Fallow/Bare"
        }
        
        # Lazy load model (only when first prediction is made)
        self.model = None
        
    def _load_model(self):
        """Lazy load Presto model"""
        if self.model is None:
            try:
                from presto import Presto
                logger.info("Loading Presto pre-trained weights...")
                self.model = Presto.load_pretrained()
                self.model.to(self.device)
                self.model.eval()
                logger.info("Presto model loaded successfully")
            except ImportError:
                logger.warning("Presto package not available - using rule-based classifier")
                self.model = "rule_based"  # Flag to use fallback
            except Exception as e:
                logger.error(f"Failed to load Presto: {e}")
                logger.warning("Falling back to rule-based classifier")
                self.model = "rule_based"
    
    def extract_sentinel2_timeseries(self, polygon: ee.Geometry, start_date: str, end_date: str) -> np.ndarray:
        """
        Extract Sentinel-2 time series from Google Earth Engine
        
        Returns: numpy array of shape [time_steps, 10_bands]
        Bands: B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12
        """
        try:
            # Get Sentinel-2 collection
            collection = ee.ImageCollection("COPERNICUS/S2_HARMONIZED")\
                .filterBounds(polygon)\
                .filterDate(start_date, end_date)\
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))\
                .select(['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12'])\
                .sort("system:time_start")
            
            # Check if we have data
            size = collection.size().getInfo()
            if size == 0:
                raise ValueError("No Sentinel-2 imagery available for this field/date range")
            
            logger.info(f"Found {size} Sentinel-2 images")
            
            # Extract mean values for all images in the collection efficiently
            def reduce_image(image):
                stats = image.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=polygon,
                    scale=20,
                    maxPixels=1e9
                )
                
                # Create a feature with the mean values as properties
                return ee.Feature(None, stats).set("system:time_start", image.get("system:time_start"))
            
            # Use map to process all images on GEE servers
            features_collection = collection.map(reduce_image)
            
            # Get data in one single info request
            features_list = features_collection.getInfo()['features']
            
            if not features_list:
                raise ValueError("Could not extract band data from imagery")
            
            # Extract features in correct order
            bands = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12']
            timeseries = []
            
            for feat in features_list:
                props = feat['properties']
                values = [props.get(band, 0) for band in bands]
                timeseries.append(values)
            
            logger.info(f"Successfully extracted {len(timeseries)} time steps from GEE")
            
            # Convert to numpy array
            timeseries_array = np.array(timeseries, dtype=np.float32)
            
            # Normalize to [0, 1] range (Sentinel-2 values are 0-10000)
            timeseries_array = timeseries_array / 10000.0
            
            logger.info(f"Extracted timeseries shape: {timeseries_array.shape}")
            
            return timeseries_array
            
        except Exception as e:
            logger.error(f"Error extracting Sentinel-2 data: {e}")
            raise
    
    def predict(self, sentinel2_timeseries: np.ndarray, start_date: str, end_date: str) -> Dict:
        """
        Run crop classification on Sentinel-2 time series
        Currently using rule-based classifier (Presto integration coming soon)
        
        Args:
            sentinel2_timeseries: numpy array [time_steps, 10_bands]
            start_date: ISO date string
            end_date: ISO date string
            
        Returns:
            {
                "crop_type": str,
                "confidence": float,
                "alternatives": List[Dict]
            }
        """
        # For now, always use rule-based classifier
        # Presto integration will be added post-symposium
        return self._fallback_prediction(sentinel2_timeseries, start_date, end_date)
    
    def _fallback_prediction(self, timeseries: np.ndarray, start_date: str, end_date: str) -> Dict:
        """
        Seasonal rule-based classifier for Zimbabwe
        Distinguishes between crops based on NDVI magnitude, trend (slope), and season
        """
        logger.info(f"Running seasonal rule-based classifier for {start_date} to {end_date}")
        
        # Calculate NDVI from bands (B8-B4)/(B8+B4)
        nir = timeseries[:, 6]  # B8
        red = timeseries[:, 2]  # B4
        ndvi = (nir - red) / (nir + red + 1e-8)
        
        # Calculate key metrics
        max_ndvi = float(np.max(ndvi))
        mean_ndvi = float(np.mean(ndvi))
        
        # Calculate growth trend (slope)
        # Using simple linear regression slope if we have enough points, else start-end diff
        if len(ndvi) > 2:
            x = np.arange(len(ndvi))
            slope = float(np.polyfit(x, ndvi, 1)[0])
        elif len(ndvi) == 2:
            slope = float(ndvi[1] - ndvi[0])
        else:
            slope = 0.0
            
        # Parse months to determine season
        try:
            start_month = datetime.strptime(start_date, "%Y-%m-%d").month
            end_month = datetime.strptime(end_date, "%Y-%m-%d").month
        except:
            # Fallback if date parsing fails
            start_month = 1
            end_month = 12
            
        # WINTER SEASON (May - September)
        # Main crop: Wheat
        is_winter = (5 <= start_month <= 8) or (5 <= end_month <= 9)
        
        # SUMMER SEASON (November - April)
        # Main crops: Maize, Tobacco, Cotton, Soybeans
        is_summer = (start_month >= 11 or start_month <= 3)
        
        candidates = []
        
        # 1. WHEAT (Winter strictly irrigated)
        # Signature: High growth (slope > 0) or sustained high NDVI in Winter
        if is_winter:
            if slope > 0.01 and max_ndvi > 0.35:
                candidates.append({"crop": "Wheat", "confidence": 0.85 if max_ndvi > 0.6 else 0.75})
            elif max_ndvi > 0.6 and mean_ndvi > 0.4:
                candidates.append({"crop": "Wheat", "confidence": 0.70})

        # 2. MAIZE (Summer rain-fed or irrigated)
        # Signature: Very high peak NDVI (>0.75)
        if max_ndvi > 0.75:
            # If it's winter and increasing, Wheat is more likely, otherwise Maize
            conf = 0.80 if not is_winter or slope < 0 else 0.40
            candidates.append({"crop": "Maize", "confidence": conf})
        elif max_ndvi > 0.65 and mean_ndvi > 0.45:
            candidates.append({"crop": "Maize", "confidence": 0.60})

        # 3. TOBACCO
        # Signature: Medium-high steady NDVI, often distinct from fast-growing Maize
        if 0.5 < max_ndvi < 0.75 and abs(slope) < 0.02:
            candidates.append({"crop": "Tobacco", "confidence": 0.65})
        elif is_summer and max_ndvi > 0.6:
            candidates.append({"crop": "Tobacco", "confidence": 0.55})

        # 4. BARE/FALLOW
        if max_ndvi < 0.25:
            return {
                "crop_type": "Bare Soil / Fallow", 
                "confidence": 0.90, 
                "alternatives": [],
                "method": "rule_based_fallback"
            }

        # Sort candidates by confidence
        candidates.sort(key=lambda x: x["confidence"], reverse=True)
        
        if not candidates:
            # Absolute fallback
            if max_ndvi > 0.6:
                crop_res = "Maize"
                conf_res = 0.50
            else:
                crop_res = "Unknown"
                conf_res = 0.40
            
            return {
                "crop_type": crop_res,
                "confidence": conf_res,
                "alternatives": [],
                "method": "rule_based_fallback"
            }

        # Format result
        primary = candidates[0]
        alternatives = candidates[1:3] if len(candidates) > 1 else []
        
        return {
            "crop_type": primary["crop"],
            "confidence": primary["confidence"],
            "alternatives": alternatives,
            "method": "rule_based_fallback"
        }
    
    def classify_field(self, coordinates: List, start_date: str, end_date: str) -> Dict:
        """
        End-to-end classification from coordinates
        
        Args:
            coordinates: GeoJSON polygon coordinates
            start_date: ISO date string (YYYY-MM-DD)
            end_date: ISO date string (YYYY-MM-DD)
            
        Returns:
            Classification result dictionary
        """
        try:
            # Create EE polygon
            polygon = ee.Geometry.Polygon(coordinates)
            
            # Extract Sentinel-2 time series
            timeseries = self.extract_sentinel2_timeseries(polygon, start_date, end_date)
            
            # Run prediction
            result = self.predict(timeseries, start_date, end_date)
            
            return {
                "success": True,
                **result,
                "date_range": f"{start_date} to {end_date}",
                "images_analyzed": len(timeseries)
            }
            
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "crop_type": "Error",
                "confidence": 0.0
            }
