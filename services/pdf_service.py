import os
import io
import logging
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT

logger = logging.getLogger(__name__)

def generate_pdf_report(report_data):
    """
    Generate a PDF report using ReportLab.
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
        styles.add(ParagraphStyle(name='CenterTitle', parent=styles['Heading1'], alignment=TA_CENTER))
        styles.add(ParagraphStyle(name='SectionHeader', parent=styles['Heading2'], spaceBefore=12, spaceAfter=6, textColor=colors.darkgreen))
        styles.add(ParagraphStyle(name='NormalSmall', parent=styles['Normal'], fontSize=10))
        styles.add(ParagraphStyle(name='RiskHigh', parent=styles['Normal'], textColor=colors.red, fontName='Helvetica-Bold'))
        styles.add(ParagraphStyle(name='RiskMedium', parent=styles['Normal'], textColor=colors.orange, fontName='Helvetica-Bold'))
        styles.add(ParagraphStyle(name='RiskLow', parent=styles['Normal'], textColor=colors.green, fontName='Helvetica-Bold'))

        story = []

        # --- PAGE 1: Executive Summary ---
        field_info = report_data.get("database_field_info", {})
        growth_stage = report_data.get("growth_stage", {})
        prof_summary = report_data.get("professional_summary", {})

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

        # Professional Summary
        story.append(Paragraph("Executive Summary", styles['SectionHeader']))
        story.append(Paragraph(prof_summary.get("current_position", "N/A"), styles['Normal']))
        story.append(Spacer(1, 6))
        story.append(Paragraph(f"<b>Trajectory:</b> {prof_summary.get('trajectory_analysis', 'N/A')}", styles['Normal']))
        story.append(Spacer(1, 6))
        story.append(Paragraph(f"<b>Guidance:</b> {prof_summary.get('stakeholder_guidance', 'N/A')}", styles['Normal']))
        
        story.append(PageBreak())

        # --- PAGE 2: Multi-Index & Deep Science ---
        indices = report_data.get("vegetation_indices", {})
        multi_index = report_data.get("multi_index_assessment", {})
        
        story.append(Paragraph("Multi-Index Analysis", styles['CenterTitle']))
        story.append(Spacer(1, 12))

        # Indices Table
        data = [
            ["Index", "Value", "Interpretation"],
            ["NDVI", f"{indices.get('ndvi', 0):.2f}" if indices.get('ndvi') is not None else "N/A", "Biomass & Vigor"],
            ["EVI", f"{indices.get('evi', 0):.2f}" if indices.get('evi') is not None else "N/A", "Enhanced Vigor (High Biomass)"],
            ["SAVI", f"{indices.get('savi', 0):.2f}" if indices.get('savi') is not None else "N/A", "Soil Adjusted Vigor"],
            ["NDMI", f"{indices.get('ndmi', 0):.2f}" if indices.get('ndmi') is not None else "N/A", "Canopy Moisture"],
            ["NDWI", f"{indices.get('ndwi', 0):.2f}" if indices.get('ndwi') is not None else "N/A", "Water Stress / Logging"]
        ]
        t = Table(data, colWidths=[1*inch, 1*inch, 4*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(t)
        story.append(Spacer(1, 24))

        # Deep Dive
        story.append(Paragraph("Scientific Assessment", styles['SectionHeader']))
        story.append(Paragraph(f"<b>Canopy Analysis:</b> {multi_index.get('canopy_analysis', 'N/A')}", styles['Normal']))
        story.append(Spacer(1, 6))
        story.append(Paragraph(f"<b>Combined Interpretation:</b> {multi_index.get('combined_interpretation', 'N/A')}", styles['Normal']))
        
        story.append(Spacer(1, 12))
        story.append(Paragraph("Stress Indicators", styles['SectionHeader']))
        stress_list = multi_index.get("stress_indicators", [])
        if stress_list:
            for item in stress_list:
                story.append(Paragraph(f"• {item}", styles['Normal']))
        else:
            story.append(Paragraph("No significant stress indicators detected.", styles['Normal']))

        story.append(PageBreak())

        # --- PAGE 3: Farmer & Season Plan ---
        farmer = report_data.get("farmer_narrative", {})
        
        story.append(Paragraph("Farmer & Season Plan", styles['CenterTitle']))
        story.append(Spacer(1, 12))

        story.append(Paragraph("Summary", styles['SectionHeader']))
        story.append(Paragraph(farmer.get("plain_language_summary", "N/A"), styles['Normal']))
        story.append(Spacer(1, 12))

        story.append(Paragraph("Action Plan", styles['SectionHeader']))
        
        story.append(Paragraph("<b>Immediate Actions (0-7 Days):</b>", styles['Normal']))
        for action in farmer.get("immediate_actions", []):
            story.append(Paragraph(f"• {action}", styles['Normal']))
        story.append(Spacer(1, 6))

        story.append(Paragraph("<b>Short Term (7-14 Days):</b>", styles['Normal']))
        for action in farmer.get("short_term_actions", []):
            story.append(Paragraph(f"• {action}", styles['Normal']))
        story.append(Spacer(1, 6))

        story.append(Paragraph("<b>Seasonal Outlook:</b>", styles['Normal']))
        for action in farmer.get("seasonal_actions", []):
            story.append(Paragraph(f"• {action}", styles['Normal']))

        story.append(PageBreak())

        # --- PAGE 4: Risk & Finance View ---
        risk = report_data.get("risk_finance_view", {})
        agronomist = report_data.get("agronomist_notes", {})

        story.append(Paragraph("Risk & Finance View", styles['CenterTitle']))
        story.append(Spacer(1, 12))

        # Risk Level
        risk_level = risk.get("risk_level", "Unknown")
        risk_style = styles['Normal']
        if "High" in risk_level:
            risk_style = styles['RiskHigh']
        elif "Medium" in risk_level:
            risk_style = styles['RiskMedium']
        elif "Low" in risk_level:
            risk_style = styles['RiskLow']
            
        story.append(Paragraph(f"Risk Level: {risk_level}", risk_style))
        story.append(Paragraph(f"Yield Outlook: {risk.get('yield_outlook', 'Unknown')}", styles['Normal']))
        story.append(Spacer(1, 12))

        story.append(Paragraph("Underwriting Signals", styles['SectionHeader']))
        for signal in risk.get("underwriting_signals", []):
            story.append(Paragraph(f"• {signal}", styles['Normal']))
        
        story.append(Spacer(1, 12))
        story.append(Paragraph("Credit Implications", styles['SectionHeader']))
        story.append(Paragraph(risk.get("credit_implications", "N/A"), styles['Normal']))

        story.append(Spacer(1, 24))
        story.append(Paragraph("Technical Agronomist Notes", styles['SectionHeader']))
        story.append(Paragraph(f"<b>Technical Summary:</b> {agronomist.get('technical_summary', 'N/A')}", styles['Normal']))
        story.append(Spacer(1, 6))
        story.append(Paragraph(f"<b>Yield Implications:</b> {agronomist.get('yield_implications', 'N/A')}", styles['Normal']))

        doc.build(story)
        buffer.seek(0)
        return buffer

    except Exception as e:
        logger.error(f"Error generating PDF: {e}")
        return None
