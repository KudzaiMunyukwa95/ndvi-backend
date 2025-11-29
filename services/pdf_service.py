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
    Generate a comprehensive PDF report using ReportLab.
    Returns a BytesIO object containing the PDF.
    """
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )

        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name='CenterTitle', parent=styles['Heading1'], alignment=TA_CENTER, fontSize=18, spaceAfter=12))
        styles.add(ParagraphStyle(name='SectionHeader', parent=styles['Heading2'], spaceBefore=12, spaceAfter=6, textColor=colors.darkgreen, fontSize=14))
        styles.add(ParagraphStyle(name='SubHeader', parent=styles['Heading3'], spaceBefore=8, spaceAfter=4, textColor=colors.darkblue, fontSize=12))
        styles.add(ParagraphStyle(name='NormalSmall', parent=styles['Normal'], fontSize=10))
        styles.add(ParagraphStyle(name='VerdictGood', parent=styles['Normal'], textColor=colors.green, fontName='Helvetica-Bold', fontSize=12))
        styles.add(ParagraphStyle(name='VerdictFair', parent=styles['Normal'], textColor=colors.orange, fontName='Helvetica-Bold', fontSize=12))
        styles.add(ParagraphStyle(name='VerdictPoor', parent=styles['Normal'], textColor=colors.red, fontName='Helvetica-Bold', fontSize=12))
        styles.add(ParagraphStyle(name='RiskHigh', parent=styles['Normal'], textColor=colors.red, fontName='Helvetica-Bold'))
        styles.add(ParagraphStyle(name='RiskMedium', parent=styles['Normal'], textColor=colors.orange, fontName='Helvetica-Bold'))
        styles.add(ParagraphStyle(name='RiskLow', parent=styles['Normal'], textColor=colors.green, fontName='Helvetica-Bold'))
        styles.add(ParagraphStyle(name='Justified', parent=styles['Normal'], alignment=TA_JUSTIFY))

        story = []

        # --- PAGE 1: Header & Executive Summary ---
        field_info = report_data.get("database_field_info", {})
        growth_stage = report_data.get("growth_stage", {})
        exec_verdict = report_data.get("executive_verdict", {})
        mgmt_priority = report_data.get("management_priority", [])
        insurance_risk = report_data.get("insurance_risk_summary", {})

        story.append(Paragraph("Agricultural Intelligence Report", styles['CenterTitle']))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"Field: {field_info.get('field_name', 'N/A')}", styles['Heading2']))
        story.append(Spacer(1, 12))

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
        ]))
        story.append(t)
        story.append(Spacer(1, 24))

        # Current Status
        story.append(Paragraph("Current Status", styles['SectionHeader']))
        story.append(Paragraph(f"<b>Growth Stage:</b> {growth_stage.get('stage', 'Unknown').title()}", styles['Normal']))
        story.append(Paragraph(f"<b>Description:</b> {growth_stage.get('stage_description', 'N/A')}", styles['Normal']))
        story.append(Paragraph(f"<b>Days Since Planting:</b> {growth_stage.get('days_since_planting', 'N/A')}", styles['Normal']))
        story.append(Spacer(1, 12))

        # Executive Verdict (NEW)
        story.append(Paragraph("Executive Summary", styles['SectionHeader']))
        verdict_text = exec_verdict.get("verdict", "Pending analysis...")
        verdict_style = styles['Normal']
        if "Good" in verdict_text:
            verdict_style = styles['VerdictGood']
        elif "Fair" in verdict_text:
            verdict_style = styles['VerdictFair']
        elif "Poor" in verdict_text:
            verdict_style = styles['VerdictPoor']
        
        story.append(Paragraph(verdict_text, verdict_style))
        story.append(Spacer(1, 6))
        story.append(Paragraph(exec_verdict.get("trajectory_statement", "N/A"), styles['Normal']))
        story.append(Spacer(1, 12))

        # Management Priority (NEW)
        if mgmt_priority:
            story.append(Paragraph("Management Priority", styles['SubHeader']))
            for priority in mgmt_priority:
                story.append(Paragraph(f"• {priority}", styles['Normal']))
            story.append(Spacer(1, 12))

        # Insurance Risk Summary (NEW)
        story.append(Paragraph("Insurance Risk Summary", styles['SubHeader']))
        risk_level = insurance_risk.get("risk_level", "Unknown")
        risk_style = styles['Normal']
        if risk_level == "High":
            risk_style = styles['RiskHigh']
        elif risk_level == "Medium":
            risk_style = styles['RiskMedium']
        elif risk_level == "Low":
            risk_style = styles['RiskLow']
        
        story.append(Paragraph(insurance_risk.get("summary", "N/A"), risk_style))
        
        story.append(PageBreak())

        # --- PAGE 2: Index Interpretation & Consistency ---
        indices = report_data.get("vegetation_indices", {})
        index_interp = report_data.get("index_interpretation", {})
        consistency = report_data.get("consistency_check", {})
        
        story.append(Paragraph("Vegetation Index Analysis", styles['CenterTitle']))
        story.append(Spacer(1, 12))

        # Index Interpretation Table (NEW)
        story.append(Paragraph("Index-by-Index Interpretation", styles['SectionHeader']))
        
        index_data = [
            ["Index", "Value", "Physical Meaning", "Status"]
        ]
        
        for idx_name in ["ndvi", "evi", "savi", "ndmi", "ndwi"]:
            idx_upper = idx_name.upper()
            value = indices.get(idx_name)
            value_str = f"{value:.2f}" if value is not None else "N/A"
            
            interp = index_interp.get(idx_name, {})
            meaning = interp.get("physical_meaning", "N/A")
            expectation = interp.get("expectation", "N/A")
            
            index_data.append([idx_upper, value_str, meaning, expectation])
        
        t = Table(index_data, colWidths=[0.8*inch, 0.8*inch, 2.5*inch, 2*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
        ]))
        story.append(t)
        story.append(Spacer(1, 18))

        # Consistency Check (NEW)
        story.append(Paragraph("Index Agreement Check", styles['SectionHeader']))
        consistency_status = consistency.get("status", "Unknown")
        consistency_statement = consistency.get("statement", "N/A")
        
        if consistency_status == "Consistent":
            story.append(Paragraph(f"<b>Status:</b> ✓ {consistency_status}", styles['Normal']))
        else:
            story.append(Paragraph(f"<b>Status:</b> ⚠ {consistency_status}", styles['Normal']))
        
        story.append(Paragraph(consistency_statement, styles['Justified']))
        story.append(Spacer(1, 12))

        # Farmland Physiology (NEW)
        physiology = report_data.get("farmland_physiology", "")
        if physiology and physiology != "Pending analysis...":
            story.append(Paragraph("Crop Stage Physiology", styles['SectionHeader']))
            story.append(Paragraph(physiology, styles['Justified']))
        
        story.append(PageBreak())

        # --- PAGE 3: Practical Guidance & Agronomist Notes ---
        practical = report_data.get("practical_guidance", {})
        agronomist = report_data.get("agronomist_notes", {})
        
        story.append(Paragraph("Technical Assessment", styles['CenterTitle']))
        story.append(Spacer(1, 12))

        # Practical Guidance (NEW)
        story.append(Paragraph("What This Means Practically", styles['SectionHeader']))
        story.append(Paragraph(f"<b>Crop Status:</b> {practical.get('crop_status', 'N/A')}", styles['Normal']))
        story.append(Paragraph(f"<b>Yield Risk:</b> {practical.get('yield_risk', 'N/A')}", styles['Normal']))
        story.append(Paragraph(f"<b>Action Timeline:</b> {practical.get('action_timeline', 'N/A')}", styles['Normal']))
        story.append(Spacer(1, 12))

        # Agronomist Notes (UPDATED)
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
        
        story.append(PageBreak())

        # --- PAGE 4: Farmer Narrative & Action Plan ---
        farmer = report_data.get("farmer_narrative", {})
        historical = report_data.get("historical_context", {})
        
        story.append(Paragraph("Farmer Guidance & Action Plan", styles['CenterTitle']))
        story.append(Spacer(1, 12))

        story.append(Paragraph("Summary for Farmer", styles['SectionHeader']))
        story.append(Paragraph(farmer.get("plain_language_summary", "N/A"), styles['Justified']))
        story.append(Spacer(1, 12))

        story.append(Paragraph("Action Plan", styles['SectionHeader']))
        
        story.append(Paragraph("<b>Immediate Actions (0-7 Days):</b>", styles['SubHeader']))
        for action in farmer.get("immediate_actions", []):
            story.append(Paragraph(f"• {action}", styles['Normal']))
        story.append(Spacer(1, 6))

        story.append(Paragraph("<b>Short Term (7-14 Days):</b>", styles['SubHeader']))
        for action in farmer.get("short_term_actions", []):
            story.append(Paragraph(f"• {action}", styles['Normal']))
        story.append(Spacer(1, 6))

        story.append(Paragraph("<b>Seasonal Outlook:</b>", styles['SubHeader']))
        for action in farmer.get("seasonal_actions", []):
            story.append(Paragraph(f"• {action}", styles['Normal']))
        story.append(Spacer(1, 12))

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
