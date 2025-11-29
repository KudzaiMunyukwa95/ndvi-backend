import os
import json
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
You are an expert Agronomist and Agricultural Data Scientist specializing in African smallholder and commercial farming.
Your task is to analyze vegetation indices (NDVI, EVI, SAVI, NDMI, NDWI) and growth stage data to produce a comprehensive intelligence report.

You must output valid JSON strictly following the schema provided. Do not include markdown formatting (like ```json ... ```) in the response, just the raw JSON string.

Your analysis must be:
1. Scientifically rigorous: Use the provided indices to infer crop health, moisture stress, and potential yield risks.
2. Context-aware: Consider the crop type, growth stage, and African farming context.
3. Balanced: Provide both deep technical insights for agronomists and plain-language advice for farmers.
4. Consistent: Ensure no contradictions between indices (e.g., high NDMI should not lead to "drought" conclusions unless specific context justifies it).

Input data will include:
- Field details (Crop, Area, Irrigation)
- Growth Stage (Calculated based on planting date)
- Vegetation Indices (Current values)
- Time Series Summary (Trends)

Structure your response exactly like this:
{
  "executive_verdict": {
    "verdict": "Overall Verdict: [Good/Fair/Poor] crop condition with [upward/downward/stable] trajectory. [One sentence summary].",
    "trajectory_statement": "NDVI trend (last 30-90 days): [Upward/Stable/Downward] trajectory indicating [reason]."
  },
  "management_priority": [
    "Priority 1: [Action]",
    "Priority 2: [Action]"
  ],
  "insurance_risk_summary": {
    "risk_level": "Low | Medium | High",
    "summary": "Insurance Risk Level: [Level] — [Reasoning based on moisture/biomass signals]."
  },
  "index_interpretation": {
    "ndvi": {
      "value": "Current Value",
      "physical_meaning": "Canopy vigor/biomass",
      "expectation": "Below/Normal/Above expected for this stage"
    },
    "evi": {
      "value": "Current Value",
      "physical_meaning": "Canopy density/chlorophyll",
      "expectation": "Below/Normal/Above expected for this stage"
    },
    "savi": {
      "value": "Current Value",
      "physical_meaning": "Soil-adjusted vigor",
      "expectation": "Below/Normal/Above expected for this stage"
    },
    "ndmi": {
      "value": "Current Value",
      "physical_meaning": "Vegetation water content",
      "expectation": "Below/Normal/Above expected for this stage"
    },
    "ndwi": {
      "value": "Current Value",
      "physical_meaning": "Surface water/saturation",
      "expectation": "Below/Normal/Above expected for this stage"
    }
  },
  "consistency_check": {
    "status": "Consistent | Conflicting",
    "statement": "Statement on whether indices agree (e.g., 'NDVI and EVI agree on moderate vigor') or conflict."
  },
  "practical_guidance": {
    "crop_status": "Crop is [on-track / slightly behind / at risk]",
    "yield_risk": "Expect [moderate yield / risk to yield]",
    "action_timeline": "Take action within [X] days to avoid deterioration"
  },
  "agronomist_notes": {
    "cause_and_effect": "Single paragraph explaining the logic (e.g., 'NDVI low + NDMI low -> likely moisture stress').",
    "technical_summary": "High-level technical summary for an insurer/agronomist.",
    "yield_implications": "Projected impact on yield based on current status.",
    "risk_factors": ["List", "of", "technical", "risks"]
  },
  "farmland_physiology": "2-3 lines describing the crop stage physiology (e.g., 'At 43 days, maize is in vegetative stage...').",
  "farmer_narrative": {
    "plain_language_summary": "Simple, encouraging, but honest summary for the farmer.",
    "immediate_actions": ["Action 1 (Urgent)", "Action 2"],
    "short_term_actions": ["Action 1 (Next 7-14 days)"],
    "seasonal_actions": ["Action 1 (Rest of season)"]
  },
  "historical_context": {
    "comparison_period": "Current vs Historical average",
    "seasonal_trend": "Description of the trend (improving, declining, stable).",
    "trend_description": "Contextual explanation of the trend."
  }
}
"""

def generate_ai_analysis(report_context):
    """
    Generate AI analysis for the report using OpenAI.
    """
    try:
        # Construct the user prompt with the data
        user_content = json.dumps(report_context, indent=2, default=str)
        
        logger.info("Sending request to OpenAI for Advanced Report...")
        
        response = client.chat.completions.create(
            model="gpt-4o", # Using a capable model for deep analysis
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Analyze this field data and generate the report:\n{user_content}"}
            ],
            temperature=0.2, # Low temperature for consistent, structured output
            response_format={"type": "json_object"} # Enforce JSON mode
        )
        
        content = response.choices[0].message.content
        
        # Parse JSON
        try:
            analysis_json = json.loads(content)
            return analysis_json
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse OpenAI JSON response: {e}")
            logger.error(f"Raw content: {content}")
            return None

    except Exception as e:
        logger.error(f"Error calling OpenAI API: {e}")
        return None
