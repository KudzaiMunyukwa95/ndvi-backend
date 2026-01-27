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
    
    def predict(self, sentinel2_timeseries: np.ndarray) -> Dict:
        """
        Run crop classification on Sentinel-2 time series
        Currently using rule-based classifier (Presto integration coming soon)
        
        Args:
            sentinel2_timeseries: numpy array [time_steps, 10_bands]
            
        Returns:
            {
                "crop_type": str,
                "confidence": float,
                "alternatives": List[Dict]
            }
        """
        # For now, always use rule-based classifier
        # Presto integration will be added post-symposium
        return self._fallback_prediction(sentinel2_timeseries)
    
    def _fallback_prediction(self, timeseries: np.ndarray) -> Dict:
        """
        Simple rule-based fallback if Presto fails
        Uses NDVI patterns to classify crops
        """
        logger.warning("Using fallback rule-based classifier")
        
        # Calculate NDVI from bands (B8-B4)/(B8+B4)
        nir = timeseries[:, 6]  # B8
        red = timeseries[:, 2]  # B4
        ndvi = (nir - red) / (nir + red + 1e-8)
        
        # Simple rules
        max_ndvi = np.max(ndvi)
        mean_ndvi = np.mean(ndvi)
        
        if max_ndvi > 0.7 and mean_ndvi > 0.5:
            crop = "Maize"
            conf = 0.75
        elif max_ndvi < 0.6 and mean_ndvi < 0.4:
            crop = "Wheat"
            conf = 0.70
        elif max_ndvi > 0.65:
            crop = "Tobacco"
            conf = 0.65
        else:
            crop = "Unknown"
            conf = 0.50
        
        return {
            "crop_type": crop,
            "confidence": conf,
            "alternatives": [],
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
            result = self.predict(timeseries)
            
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
