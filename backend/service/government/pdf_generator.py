import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

DOSSIER_DIR = Path("static/dossiers")
DOSSIER_DIR.mkdir(parents=True, exist_ok=True)

def generate_grievance_pdf(
    ticket_id: str,
    road_name: str,
    ward_name: str,
    city: str,
    lat: float,
    lng: float,
    severity: str,
    priority_score: float,
    report_count: int,
    image_path: Optional[str] = None,
    department: str = "Roads & Traffic Infrastructure Department",
    sla_hours: int = 72,
) -> str:
    """
    Generate an official Municipal Infrastructure Grievance PDF Dossier.
    Returns: Absolute filepath to generated PDF.
    """
    filename = f"Grievance_Dossier_{ticket_id}.pdf"
    pdf_path = DOSSIER_DIR / filename

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#0f172a'),
        alignment=1, # Center
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#475569'),
        alignment=1,
    )

    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#1e293b'),
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor('#334155'),
    )

    alert_style = ParagraphStyle(
        'AlertText',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#b91c1c'),
    )

    elements = []

    # 1. Header Banner
    elements.append(Paragraph("MUNICIPAL CORPORATION INFRASTRUCTURE GRIEVANCE DOSSIER", title_style))
    elements.append(Paragraph(f"Official Citizen Safety Redressal Packet • Generated via CityPulse Civic Platform", subtitle_style))
    elements.append(Spacer(1, 10))
    elements.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#2563eb'), spaceAfter=15))

    # 2. Executive Summary Table
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    google_maps_url = f"https://maps.google.com/?q={lat:.5f},{lng:.5f}"

    summary_data = [
        [
            Paragraph("<b>Ticket Reference ID:</b>", body_style),
            Paragraph(f"<font color='#2563eb'><b>#{ticket_id}</b></font>", body_style),
            Paragraph("<b>Date & Time:</b>", body_style),
            Paragraph(now_str, body_style),
        ],
        [
            Paragraph("<b>Target Jurisdiction:</b>", body_style),
            Paragraph(f"{ward_name} Ward, {city}", body_style),
            Paragraph("<b>Assigned Department:</b>", body_style),
            Paragraph(department, body_style),
        ],
        [
            Paragraph("<b>Hazard Classification:</b>", body_style),
            Paragraph(f"<font color='#dc2626'><b>{severity.replace('_', ' ').upper()}</b></font>", body_style),
            Paragraph("<b>Hazard Priority Score:</b>", body_style),
            Paragraph(f"<b>{priority_score:.1f} / 100</b>", body_style),
        ],
        [
            Paragraph("<b>Location Coordinates:</b>", body_style),
            Paragraph(f"{lat:.5f}, {lng:.5f}", body_style),
            Paragraph("<b>Citizen Confirmations:</b>", body_style),
            Paragraph(f"{report_count} Independent Citizens", body_style),
        ],
        [
            Paragraph("<b>Road / Landmark:</b>", body_style),
            Paragraph(f"<b>{road_name}</b>", body_style),
            Paragraph("<b>Charter Resolution SLA:</b>", body_style),
            Paragraph(f"<font color='#b91c1c'><b>{sla_hours} Hours</b></font>", body_style),
        ],
    ]

    summary_table = Table(summary_data, colWidths=[1.8 * inch, 2.0 * inch, 1.8 * inch, 1.8 * inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 15))

    # 3. Evidence Section (Image Proof)
    elements.append(Paragraph("<b>PHOTOGRAPHIC EVIDENCE & GEOTAGGED PROOF</b>", section_heading))
    elements.append(Spacer(1, 8))

    if image_path and os.path.exists(image_path):
        try:
            img = RLImage(image_path, width=4.5 * inch, height=3.0 * inch)
            img_table = Table([[img]], colWidths=[7.4 * inch])
            img_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#0f172a')),
                ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#94a3b8')),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))
            elements.append(img_table)
        except Exception as e:
            elements.append(Paragraph(f"<i>[Image preview rendering error: {e}]</i>", body_style))
    else:
        elements.append(Paragraph("<i>[Photographic evidence attached via secure cloud URL]</i>", body_style))

    elements.append(Spacer(1, 15))

    # 4. Mandatory SLA & Action Required Notice
    action_box_data = [
        [
            Paragraph(
                f"<b>ACTION REQUIRED BY MUNICIPAL WARD ENGINEER:</b><br/>"
                f"1. Dispatch inspection and cold-mix/asphalt patch repair crew to <b>{road_name}</b>.<br/>"
                f"2. Inspect location pin: <font color='#2563eb'><u>{google_maps_url}</u></font><br/>"
                f"3. Upload completion proof once repaired to trigger Citizen Verification clearance.<br/>"
                f"4. Failure to address high-priority hazards within <b>{sla_hours} hours</b> escalates to Superintending Engineer under Citizen Charter.",
                body_style
            )
        ]
    ]
    action_table = Table(action_box_data, colWidths=[7.4 * inch])
    action_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#fef2f2')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#f87171')),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
    ]))
    elements.append(action_table)

    # Build PDF
    doc.build(elements)
    return str(pdf_path.resolve())
