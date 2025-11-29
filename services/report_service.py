import datetime
import logging

logger = logging.getLogger(__name__)

# Comprehensive crop profiles with durations
CROP_PROFILES = {
    "Maize": {
        "duration_days": 135,
        "emergence": (0, 10),
        "vegetative": (11, 60),
        "reproductive": (61, 100),
        "maturity": (101, 135)
    },
    "Soyabeans": {
        "duration_days": 110,
        "emergence": (0, 10),
        "vegetative": (11, 45),
        "reproductive": (46, 90),
        "maturity": (91, 110)
    },
    "Wheat": {
        "duration_days": 120,
        "emergence": (0, 10),
        "vegetative": (11, 60),
        "reproductive": (61, 100),
        "maturity": (101, 120)
    },
    "Sorghum": {
        "duration_days": 100,
        "emergence": (0, 8),
        "vegetative": (9, 50),
        "reproductive": (51, 85),
        "maturity": (86, 100)
    },
    "Cotton": {
        "duration_days": 140,
        "emergence": (0, 10),
        "vegetative": (11, 70),
        "reproductive": (71, 120),
        "maturity": (121, 140)
    },
    "Groundnuts": {
        "duration_days": 115,
        "emergence": (0, 10),
        "vegetative": (11, 50),
        "reproductive": (51, 95),
        "maturity": (96, 115)
    },
    "Barley": {
        "duration_days": 90,
        "emergence": (0, 8),
        "vegetative": (9, 45),
        "reproductive": (46, 75),
        "maturity": (76, 90)
    },
    "Millet": {
        "duration_days": 100,
        "emergence": (0, 8),
        "vegetative": (9, 45),
        "reproductive": (46, 80),
        "maturity": (81, 100)
    },
    "Tobacco": {
        "duration_days": 130,
        "emergence": (0, 10),
        "vegetative": (11, 60),
        "reproductive": (61, 110),
        "maturity": (111, 130)
    },
    # Default fallback
    "default": {
        "duration_days": 120,
        "emergence": (0, 10),
        "vegetative": (11, 50),
        "reproductive": (51, 90),
        "maturity": (91, 120)
    }
}

# Harvest detection threshold (days beyond crop duration)
HARVEST_THRESHOLD_DAYS = 30

def analyze_growth_stage(crop, planting_date, current_date_str=None):
    """
    Determine growth stage and days since planting with harvest detection.
    """
    if not planting_date:
        return {
            "stage": "unknown",
            "stage_description": "Planting date not provided",
            "days_since_planting": None,
            "critical_factors": [],
            "is_harvested": False
        }

    try:
        p_date = datetime.datetime.strptime(planting_date, "%Y-%m-%d").date()
        if current_date_str:
            c_date = datetime.datetime.strptime(current_date_str, "%Y-%m-%d").date()
        else:
            c_date = datetime.date.today()

        days = (c_date - p_date).days

        if days < 0:
            return {
                "stage": "planned",
                "stage_description": f"Planting planned in {abs(days)} days",
                "days_since_planting": days,
                "critical_factors": ["Soil preparation", "Seed acquisition"],
                "is_harvested": False
            }

        # Get crop profile
        profile = CROP_PROFILES.get(crop, CROP_PROFILES["default"])
        duration = profile["duration_days"]
        
        # Check if crop has been harvested (days > duration + threshold)
        if days > (duration + HARVEST_THRESHOLD_DAYS):
            return {
                "stage": "harvested",
                "stage_description": f"Crop harvested (expected duration: {duration} days). Observed vegetation likely from regrowth, weeds, or volunteer plants.",
                "days_since_planting": days,
                "critical_factors": ["Land preparation", "Residue management", "Next season planning"],
                "is_harvested": True,
                "crop_duration": duration
            }
        
        # Determine current stage
        stage = "unknown"
        description = "Unknown stage"
        
        if days <= profile["emergence"][1]:
            stage = "emergence"
            description = "Germination and early seedling establishment"
        elif days <= profile["vegetative"][1]:
            stage = "vegetative"
            description = "Active leaf development and canopy expansion"
        elif days <= profile["reproductive"][1]:
            stage = "reproductive"
            description = "Flowering, pollination, and grain/fruit filling"
        elif days <= profile["maturity"][1]:
            stage = "maturity"
            description = "Physiological maturity and senescence"
        else:
            stage = "late_maturity"
            description = f"Late maturity or post-harvest (within {HARVEST_THRESHOLD_DAYS} days of expected harvest)"

        return {
            "stage": stage,
            "stage_description": description,
            "days_since_planting": days,
            "critical_factors": _get_critical_factors(stage),
            "is_harvested": False,
            "crop_duration": duration,
            "days_to_harvest": max(0, duration - days) if days < duration else 0
        }

    except Exception as e:
        logger.error(f"Error calculating growth stage: {e}")
        return {
            "stage": "unknown",
            "stage_description": "Error calculating stage",
            "days_since_planting": None,
            "critical_factors": [],
            "is_harvested": False
        }

def _get_critical_factors(stage):
    if stage == "emergence":
        return ["Soil moisture", "Soil temperature", "Seed quality"]
    elif stage == "vegetative":
        return ["Nitrogen availability", "Weed competition", "Water stress"]
    elif stage == "reproductive":
        return ["Water stress (critical)", "Disease pressure", "Temperature stress"]
    elif stage == "maturity":
        return ["Harvest timing", "Pest damage", "Grain moisture"]
    return []

def validate_indices(indices):
    """
    Clean and validate vegetation indices.
    """
    validated = {}
    
    # EVI sanity check [-1, 1]
    evi = indices.get("EVI")
    if evi is not None:
        validated["evi"] = max(-1.0, min(1.0, evi))
    else:
        validated["evi"] = None

    # NDVI
    ndvi = indices.get("NDVI")
    if ndvi is not None:
        validated["ndvi"] = max(-1.0, min(1.0, ndvi))
    else:
        validated["ndvi"] = None

    # SAVI
    savi = indices.get("SAVI")
    if savi is not None:
        validated["savi"] = max(-1.0, min(1.0, savi))
    else:
        validated["savi"] = None
        
    # NDMI
    ndmi = indices.get("NDMI")
    if ndmi is not None:
        validated["ndmi"] = max(-1.0, min(1.0, ndmi))
    else:
        validated["ndmi"] = None
        
    # NDWI
    ndwi = indices.get("NDWI")
    if ndwi is not None:
        validated["ndwi"] = max(-1.0, min(1.0, ndwi))
    else:
        validated["ndwi"] = None

    return validated

def build_report_structure(
    field_info,
    growth_data,
    indices_data,
    ai_analysis=None
):
    """
    Assemble the final report dictionary.
    """
    
    # Default empty analysis if AI fails or not provided yet
    if not ai_analysis:
        ai_analysis = {
            "executive_verdict": "Pending",
            "final_field_verdict": "Analysis pending...",
            "executive_summary": {
                "crop_status": "Pending",
                "field_condition": "Pending",
                "management_priority": "Pending",
                "one_line_summary": "Analysis pending..."
            },
            "cross_index_synthesis": "Analysis pending...",
            "physiological_narrative": "Analysis pending...",
            "index_interpretation": {},
            "temporal_trend": {
                "direction": "Unknown",
                "statement": "Pending analysis..."
            },
            "confidence_assessment": {
                "score": 0.0,
                "explanation": "Pending analysis..."
            },
            "farmer_guidance": {
                "immediate_actions_0_7_days": [],
                "field_checks": [],
                "harvest_or_next_season": "Pending..."
            },
            "professional_technical_notes": {
                "canopy_structure": "Pending...",
                "biomass_distribution": "Pending...",
                "moisture_stress_interaction": "Pending...",
                "senescence_quality": "Pending...",
                "spatial_heterogeneity": "Pending..."
            },
            "agronomist_notes": {
                "cause_and_effect": "Pending analysis...",
                "technical_summary": "Pending analysis...",
                "yield_implications": "Pending analysis...",
                "risk_factors": []
            },
            "historical_context": {
                "comparison_period": "Pending...",
                "seasonal_trend": "Pending...",
                "trend_description": "Pending..."
            }
        }

    report = {
        "report_date": datetime.date.today().isoformat(),
        "database_field_info": field_info,
        "growth_stage": growth_data,
        "vegetation_indices": indices_data,
        "executive_verdict": ai_analysis.get("executive_verdict", "Pending"),
        "final_field_verdict": ai_analysis.get("final_field_verdict", ""),
        "executive_summary": ai_analysis.get("executive_summary", {}),
        "cross_index_synthesis": ai_analysis.get("cross_index_synthesis", ""),
        "physiological_narrative": ai_analysis.get("physiological_narrative", ""),
        "index_interpretation": ai_analysis.get("index_interpretation", {}),
        "temporal_trend": ai_analysis.get("temporal_trend", {}),
        "confidence_assessment": ai_analysis.get("confidence_assessment", {}),
        "farmer_guidance": ai_analysis.get("farmer_guidance", {}),
        "professional_technical_notes": ai_analysis.get("professional_technical_notes", {}),
        "agronomist_notes": ai_analysis.get("agronomist_notes", {}),
        "historical_context": ai_analysis.get("historical_context", {})
    }
    
    return report
