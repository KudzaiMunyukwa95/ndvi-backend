import datetime
import logging

logger = logging.getLogger(__name__)

# Crop growth stages (approximate days)
# This can be refined with more specific crop data
CROP_STAGES = {
    "Maize": {
        "emergence": (0, 10),
        "vegetative": (11, 60),
        "reproductive": (61, 100),
        "maturity": (101, 140)
    },
    "Soyabeans": {
        "emergence": (0, 10),
        "vegetative": (11, 45),
        "reproductive": (46, 90),
        "maturity": (91, 120)
    },
    "Wheat": {
        "emergence": (0, 10),
        "vegetative": (11, 60),
        "reproductive": (61, 100),
        "maturity": (101, 130)
    },
    # Default fallback
    "default": {
        "emergence": (0, 10),
        "vegetative": (11, 50),
        "reproductive": (51, 90),
        "maturity": (91, 120)
    }
}

def analyze_growth_stage(crop, planting_date, current_date_str=None):
    """
    Determine growth stage and days since planting.
    """
    if not planting_date:
        return {
            "stage": "unknown",
            "stage_description": "Planting date not provided",
            "days_since_planting": None,
            "critical_factors": []
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
                "critical_factors": ["Soil preparation", "Seed acquisition"]
            }

        stages = CROP_STAGES.get(crop, CROP_STAGES["default"])
        
        stage = "unknown"
        description = "Unknown stage"
        
        if days <= stages["emergence"][1]:
            stage = "emergence"
            description = "Germination and early seedling growth"
        elif days <= stages["vegetative"][1]:
            stage = "vegetative"
            description = "Leaf development and stem elongation"
        elif days <= stages["reproductive"][1]:
            stage = "reproductive"
            description = "Flowering and grain filling"
        elif days <= stages["maturity"][1]:
            stage = "maturity"
            description = "Ripening and harvest readiness"
        else:
            stage = "maturity" # Or post-harvest?
            description = "Late maturity or post-harvest"

        return {
            "stage": stage,
            "stage_description": description,
            "days_since_planting": days,
            "critical_factors": _get_critical_factors(stage)
        }

    except Exception as e:
        logger.error(f"Error calculating growth stage: {e}")
        return {
            "stage": "unknown",
            "stage_description": "Error calculating stage",
            "days_since_planting": None,
            "critical_factors": []
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
            "multi_index_assessment": {
                "canopy_analysis": "Pending analysis...",
                "combined_interpretation": "Pending analysis...",
                "indices_summary": {
                    "biomass_vigor": "Pending...",
                    "canopy_development": "Pending...",
                    "moisture_status": "Pending...",
                    "stress_indicators": "Pending..."
                },
                "stress_indicators": []
            },
            "agronomist_notes": {
                "index_analysis": {},
                "technical_summary": "Pending analysis...",
                "yield_implications": "Pending analysis...",
                "risk_factors": []
            },
            "farmer_narrative": {
                "plain_language_summary": "Pending analysis...",
                "immediate_actions": [],
                "short_term_actions": [],
                "seasonal_actions": []
            },
            "risk_finance_view": {
                "risk_level": "Unknown",
                "underwriting_signals": [],
                "yield_outlook": "Unknown",
                "contract_farming_notes": "Pending...",
                "credit_implications": "Pending..."
            },
            "professional_summary": {
                "current_position": "Pending...",
                "stakeholder_guidance": "Pending...",
                "trajectory_analysis": "Pending..."
            },
             "historical_context": {
                "comparison_period": "Pending...",
                "historical_percentile": None,
                "seasonal_trend": "Pending...",
                "trend_description": "Pending..."
            }
        }

    report = {
        "report_date": datetime.date.today().isoformat(),
        "database_field_info": field_info,
        "growth_stage": growth_data,
        "vegetation_indices": indices_data,
        "multi_index_assessment": ai_analysis.get("multi_index_assessment", {}),
        "agronomist_notes": ai_analysis.get("agronomist_notes", {}),
        "farmer_narrative": ai_analysis.get("farmer_narrative", {}),
        "risk_finance_view": ai_analysis.get("risk_finance_view", {}),
        "professional_summary": ai_analysis.get("professional_summary", {}),
        "historical_context": ai_analysis.get("historical_context", {})
    }
    
    return report
