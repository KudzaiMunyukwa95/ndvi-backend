import os
import io
import logging
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

logger = logging.getLogger(__name__)

def generate_pdf_report(report_data):
    """
    Generate a comprehensive, high-clarity PDF report using ReportLab.
    Returns a BytesIO object containing the PDF.
    """
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=60,
            leftMargin=60,
            topMargin=60,
            bottomMargin=60
        )

        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name='CenterTitle', parent=styles['Heading1'], alignment=TA_CENTER, fontSize=18, spaceAfter=12, textColor=colors.darkgreen))
        styles.add(ParagraphStyle(name='SectionHeader', parent=styles['Heading2'], spaceBefore=14, spaceAfter=6, textColor=colors.darkgreen, fontSize=13))
        styles.add(ParagraphStyle(name='SubHeader', parent=styles['Heading3'], spaceBefore=8, spaceAfter=4, textColor=colors.darkblue, fontSize=11))
        styles.add(ParagraphStyle(name='StatusGood', parent=styles['Normal'], textColor=colors.green, fontName='Helvetica-Bold', fontSize=12))
        styles.add(ParagraphStyle(name='StatusFair', parent=styles['Normal'], textColor=colors.orange, fontName='Helvetica-Bold', fontSize=12))
        styles.add(ParagraphStyle(name='StatusPoor', parent=styles['Normal'], textColor=colors.red, fontName='Helvetica-Bold', fontSize=12))
        styles.add(ParagraphStyle(name='StatusHarvested', parent=styles['Normal'], textColor=colors.grey, fontName='Helvetica-Bold', fontSize=12))
        styles.add(ParagraphStyle(name='Justified', parent=styles['Normal'], alignment=TA_JUSTIFY, fontSize=10))
        styles.add(ParagraphStyle(name='SmallText', parent=styles['Normal'], fontSize=9))

        story = []

        # Extract data
        field_info = report_data.get("database_field_info", {})
        growth_stage = report_data.get("growth_stage", {})
        exec_verdict = report_data.get("executive_verdict", {})
        indices = report_data.get("vegetation_indices", {})
        
        # --- PAGE 1: Executive Summary & Field Details ---
        story.append(Paragraph("Agricultural Intelligence Report", styles['CenterTitle']))
        story.append(Spacer(1, 10))
        story.append(Paragraph(f"Field: {field_info.get('field_name', 'N/A')}", styles['Heading2']))
        story.append(Spacer(1, 10))

        # Field Details Table
        data = [
            ["Crop", field_info.get("crop", "N/A"), "Area", f"{field_info.get('area', 'N/A')} ha"],
            ["Planting Date", field_info.get("planting_date", "N/A"), "Irrigation", field_info.get("irrigation", "N/A")],
            ["Location", "Zimbabwe", "Report Date", report_data.get("report_date", "N/A")]
        ]
        t = Table(data, colWidths=[1.5*inch, 2*inch, 1.5*inch, 2*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 1, colors.white),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(t)
        story.append(Spacer(1, 18))

        # Current Status
        story.append(Paragraph("Current Status", styles['SectionHeader']))
        is_harvested = growth_stage.get('is_harvested', False)
        stage_display = growth_stage.get('stage', 'Unknown').title()
        if is_harvested:
            stage_display = "Harvested"
        
        story.append(Paragraph(f"<b>Growth Stage:</b> {stage_display}", styles['Normal']))
        story.append(Paragraph(f"<b>Description:</b> {growth_stage.get('stage_description', 'N/A')}", styles['Normal']))
        story.append(Paragraph(f"<b>Days Since Planting:</b> {growth_stage.get('days_since_planting', 'N/A')}", styles['Normal']))
        
        if not is_harvested and growth_stage.get('days_to_harvest'):
            story.append(Paragraph(f"<b>Days to Harvest:</b> ~{growth_stage.get('days_to_harvest', 'N/A')}", styles['Normal']))
        
        story.append(Spacer(1, 14))

        # Executive Verdict (NEW)
        story.append(Paragraph("Executive Verdict", styles['SectionHeader']))
        
        crop_status = exec_verdict.get("crop_status", "Pending")
        status_style = styles['Normal']
        if crop_status == "Good":
            status_style = styles['StatusGood']
        elif crop_status == "Fair":
            status_style = styles['StatusFair']
        elif crop_status == "Poor":
            status_style = styles['StatusPoor']
        elif crop_status == "Harvested":
            status_style = styles['StatusHarvested']
        
        story.append(Paragraph(f"<b>Crop Status:</b> {crop_status}", status_style))
        story.append(Paragraph(f"<b>Field Condition:</b> {exec_verdict.get('field_condition', 'N/A')}", styles['Normal']))
        story.append(Paragraph(f"<b>Management Priority:</b> {exec_verdict.get('management_priority', 'N/A')}", styles['Normal']))
        story.append(Spacer(1, 6))
        story.append(Paragraph(exec_verdict.get("one_line_summary", "Analysis pending..."), styles['Justified']))
        
        story.append(PageBreak())

        # --- PAGE 2: Physiological Narrative & Index Analysis ---
        physiological = report_data.get("physiological_narrative", "")
        index_interp = report_data.get("index_interpretation", {})
        temporal = report_data.get("temporal_trend", {})
        confidence = report_data.get("confidence_assessment", {})
        
        story.append(Paragraph("Field Analysis", styles['CenterTitle']))
        story.append(Spacer(1, 10))

        # Physiological Narrative (NEW)
        if physiological and physiological != "Analysis pending...":
            story.append(Paragraph("Physiological Narrative", styles['SectionHeader']))
            story.append(Paragraph(physiological, styles['Justified']))
            story.append(Spacer(1, 12))

        # Index-by-Index Interpretation (NEW)
        story.append(Paragraph("Index-by-Index Interpretation", styles['SectionHeader']))
        
        for idx_name in ["ndvi", "evi", "savi", "ndmi", "ndwi"]:
            idx_data = index_interp.get(idx_name, {})
            if idx_data:
                idx_upper = idx_name.upper()
                story.append(Paragraph(f"<b>{idx_upper}:</b>", styles['SubHeader']))
                story.append(Paragraph(idx_data.get("interpretation", "N/A"), styles['Justified']))
                
                cause_effect = idx_data.get("cause_effect", "")
                if cause_effect:
                    story.append(Paragraph(f"<i>Why & What:</i> {cause_effect}", styles['SmallText']))
                story.append(Spacer(1, 6))
        
        story.append(Spacer(1, 8))

        # Temporal Trend (NEW)
        story.append(Paragraph("Temporal Trend", styles['SectionHeader']))
        story.append(Paragraph(f"<b>Direction:</b> {temporal.get('direction', 'Unknown')}", styles['Normal']))
        story.append(Paragraph(temporal.get("statement", "N/A"), styles['Justified']))
        story.append(Spacer(1, 10))

        # Confidence Assessment (NEW)
        story.append(Paragraph("Confidence Assessment", styles['SectionHeader']))
        conf_score = confidence.get("score", 0.0)
        story.append(Paragraph(f"<b>Confidence Score:</b> {conf_score:.2f}", styles['Normal']))
        story.append(Paragraph(confidence.get("explanation", "N/A"), styles['Justified']))
        
        story.append(PageBreak())

        # --- PAGE 3: Farmer Guidance & Professional Notes ---
        farmer = report_data.get("farmer_guidance", {})
        tech_notes = report_data.get("professional_technical_notes", {})
        
        story.append(Paragraph("Farmer Guidance & Technical Assessment", styles['CenterTitle']))
        story.append(Spacer(1, 10))

        # Farmer Guidance (NEW)
        story.append(Paragraph("Farmer Guidance", styles['SectionHeader']))
        
        story.append(Paragraph("<b>Immediate Actions (0-7 Days):</b>", styles['SubHeader']))
        for action in farmer.get("immediate_actions_0_7_days", []):
            story.append(Paragraph(f"• {action}", styles['Normal']))
        story.append(Spacer(1, 6))

        story.append(Paragraph("<b>Field Checks:</b>", styles['SubHeader']))
        for check in farmer.get("field_checks", []):
            story.append(Paragraph(f"• {check}", styles['Normal']))
        story.append(Spacer(1, 6))

        harvest_guidance = farmer.get("harvest_or_next_season", "")
        if harvest_guidance and harvest_guidance != "Pending...":
            story.append(Paragraph("<b>Harvest / Next Season:</b>", styles['SubHeader']))
            story.append(Paragraph(harvest_guidance, styles['Justified']))
        
        story.append(Spacer(1, 14))

        # Professional Technical Notes (NEW)
        story.append(Paragraph("Professional Technical Notes", styles['SectionHeader']))
        
        for key, label in [
            ("canopy_structure", "Canopy Structure"),
            ("biomass_distribution", "Biomass Distribution"),
            ("moisture_stress_interaction", "Moisture-Stress Interaction"),
            ("senescence_quality", "Senescence Quality"),
            ("spatial_heterogeneity", "Spatial Heterogeneity")
        ]:
            value = tech_notes.get(key, "")
            if value and value != "Pending...":
                story.append(Paragraph(f"<b>{label}:</b> {value}", styles['Justified']))
                story.append(Spacer(1, 4))
        
        story.append(PageBreak())

        # --- PAGE 4: Agronomist Notes & Historical Context ---
        agronomist = report_data.get("agronomist_notes", {})
        historical = report_data.get("historical_context", {})
        
        story.append(Paragraph("Agronomist Notes & Context", styles['CenterTitle']))
        story.append(Spacer(1, 10))

        # Agronomist Notes
        story.append(Paragraph("Agronomist Notes", styles['SectionHeader']))
        
        cause_effect = agronomist.get("cause_and_effect", "")
        if cause_effect and cause_effect != "Pending analysis...":
            story.append(Paragraph("<b>Cause & Effect Analysis:</b>", styles['SubHeader']))
            story.append(Paragraph(cause_effect, styles['Justified']))
            story.append(Spacer(1, 8))
        
        story.append(Paragraph(f"<b>Technical Summary:</b> {agronomist.get('technical_summary', 'N/A')}", styles['Justified']))
        story.append(Spacer(1, 6))
        story.append(Paragraph(f"<b>Yield Implications:</b> {agronomist.get('yield_implications', 'N/A')}", styles['Justified']))
        story.append(Spacer(1, 8))
        
        risk_factors = agronomist.get("risk_factors", [])
        if risk_factors:
            story.append(Paragraph("<b>Risk Factors:</b>", styles['SubHeader']))
            for factor in risk_factors:
                story.append(Paragraph(f"• {factor}", styles['Normal']))
        
        story.append(Spacer(1, 14))

        # Historical Context
        story.append(Paragraph("Historical Context", styles['SectionHeader']))
        story.append(Paragraph(f"<b>Comparison Period:</b> {historical.get('comparison_period', 'N/A')}", styles['Normal']))
        story.append(Paragraph(f"<b>Seasonal Trend:</b> {historical.get('seasonal_trend', 'N/A')}", styles['Normal']))
        story.append(Paragraph(f"<b>Trend Description:</b> {historical.get('trend_description', 'N/A')}", styles['Justified']))

        doc.build(story)
        buffer.seek(0)
        return buffer

    except Exception as e:
        logger.error(f"Error generating PDF: {e}")
        return None
