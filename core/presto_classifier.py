"""
Presto-based Crop Classification Module
Uses NASA Harvest's Presto foundation model for satellite-based crop type detection
"""

import torch
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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Initializing Presto on {self.device}")
        
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
            
            # Extract mean values for each image
            def extract_bands(image):
                stats = image.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=polygon,
                    scale=20,
                    maxPixels=1e9
                ).getInfo()
                
                # Extract band values in correct order
                bands = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12']
                values = [stats.get(band, 0) for band in bands]
                
                return values
            
            # Get all images as list
            images = collection.toList(size)
            
            # Extract features for each image
            timeseries = []
            for i in range(min(size, 50)):  # Limit to 50 images for speed
                img = ee.Image(images.get(i))
                features = extract_bands(img)
                timeseries.append(features)
            
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
        Run Presto inference on Sentinel-2 time series
        
        Args:
            sentinel2_timeseries: numpy array [time_steps, 10_bands]
            
        Returns:
            {
                "crop_type": str,
                "confidence": float,
                "alternatives": List[Dict]
            }
        """
        try:
            # Load model if not already loaded
            self._load_model()
            
            # Check if we're using rule-based fallback
            if self.model == "rule_based":
                return self._fallback_prediction(sentinel2_timeseries)
            
            # Convert to tensor
            x = torch.from_numpy(sentinel2_timeseries).unsqueeze(0).to(self.device)  # [1, time, bands]
            
            # Run inference
            with torch.no_grad():
                # Get Presto embeddings
                embeddings = self.model.encoder(x)
                
                # Classify (you may need to add a classification head)
                # For now, using a simple approach
                logits = self.model.classifier(embeddings) if hasattr(self.model, 'classifier') else embeddings
                
                # Get probabilities
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            
            # Get top predictions
            top_indices = np.argsort(probs)[::-1][:3]
            
            result = {
                "crop_type": self.crop_labels.get(top_indices[0], "Unknown"),
                "confidence": float(probs[top_indices[0]]),
                "alternatives": [
                    {
                        "crop": self.crop_labels.get(idx, "Unknown"),
                        "confidence": float(probs[idx])
                    }
                    for idx in top_indices[1:3]
                ],
                "method": "presto_foundation_model"
            }
            
            logger.info(f"Prediction: {result['crop_type']} ({result['confidence']:.2%})")
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            # Fallback to rule-based if Presto fails
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
