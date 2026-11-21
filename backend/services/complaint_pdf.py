"""One-page complaint PDF (English; ReportLab cannot shape Devanagari). Layout adapted from the original
grievance dossier, but it states plainly that it is a citizen complaint, not a government document."""

import io
from datetime import UTC, datetime
from xml.sax.saxutils import escape

from reportlab.graphics.barcode.qr import QrCodeWidget
from reportlab.graphics.shapes import Drawing
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import HRFlowable, Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from backend.models import Hazard, aware
from backend.services.complaint import TYPE_LABELS, city_of, map_link
from backend.services.present import days_open


def _qr(url: str, size: float = 1.3 * inch) -> Drawing:
    widget = QrCodeWidget(url)
    x1, y1, x2, y2 = widget.getBounds()
    d = Drawing(size, size, transform=[size / (x2 - x1), 0, 0, size / (y2 - y1), 0, 0])
    d.add(widget)
    return d


def build_pdf(h: Hazard, app_url: str, photo: bytes | None) -> bytes:
    city = city_of(h)
    share = f"{app_url.rstrip('/')}/h/{h.id}"
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=40, rightMargin=40, topMargin=36, bottomMargin=36,
                            title=f"Citizen Road Hazard Complaint CP-{h.id}", author="CityPulse",
                            pageCompression=0)
    base = getSampleStyleSheet()["Normal"]
    title = ParagraphStyle("t", parent=base, fontName="Helvetica-Bold", fontSize=17, leading=21,
                           textColor=colors.HexColor("#0f172a"))
    sub = ParagraphStyle("s", parent=base, fontSize=9, leading=12, textColor=colors.HexColor("#475569"))
    body = ParagraphStyle("b", parent=base, fontSize=10, leading=14, textColor=colors.HexColor("#1e293b"))
    label = ParagraphStyle("l", parent=body, fontName="Helvetica-Bold")

    def p(text, style=body):
        return Paragraph(escape(str(text)), style)

    type_label = TYPE_LABELS["en"].get(h.type, "road hazard").capitalize()
    rows = [
        ("Reference", f"CP-{h.id}"),
        ("Hazard", f"{type_label} - {h.severity} severity"),
        ("Location", h.address or f"{h.lat:.5f}, {h.lng:.5f}"),
        ("Road / ward", f"{h.road or '-'} / {h.ward or '-'}"),
        ("Coordinates", f"{h.lat:.6f}, {h.lng:.6f}  ({map_link(h)})"),
        ("Reported", f"{aware(h.first_reported_at):%d %b %Y} - open {days_open(h)} days, "
                     f"{h.report_count} citizen report(s)"),
        ("Status", h.status.value.replace("_", " ")),
    ]
    if h.official_ticket_id:
        rows.append(("Complaint no.", f"{h.official_ticket_id} ({h.official_portal or 'city portal'})"))
    table = Table([[p(k, label), p(v)] for k, v in rows], colWidths=[1.4 * inch, 5.6 * inch])
    table.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e2e8f0")),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f1f5f9")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))

    story = [
        p("Citizen Road Hazard Complaint", title),
        p(f"Prepared with CityPulse on {datetime.now(UTC):%d %b %Y}. "
          "This is a complaint from citizens, not a government document.", sub),
        Spacer(1, 8), HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563eb")), Spacer(1, 10),
        p(f"To: {city.authority}", label), Spacer(1, 6), table, Spacer(1, 12),
    ]
    if photo:
        try:
            img = Image(io.BytesIO(photo))
            scale = min(4.5 * inch / img.imageWidth, 3.2 * inch / img.imageHeight)
            img.drawWidth, img.drawHeight = img.imageWidth * scale, img.imageHeight * scale
            story += [img, Spacer(1, 10)]
        except Exception:  # a broken photo should never block the complaint
            pass
    request_text = ("Requested action: please inspect this location, repair the hazard, and record the work so "
                    "that the citizens following it can confirm the repair. Photos, the full history and live "
                    f"status are at {share}")
    qr_table = Table([[p(request_text), _qr(share)]], colWidths=[5.4 * inch, 1.6 * inch])
    qr_table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                  ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#eff6ff")),
                                  ("BOX", (0, 0), (0, 0), 0.8, colors.HexColor("#93c5fd")),
                                  ("LEFTPADDING", (0, 0), (0, 0), 10), ("RIGHTPADDING", (0, 0), (0, 0), 10),
                                  ("TOPPADDING", (0, 0), (0, 0), 8), ("BOTTOMPADDING", (0, 0), (0, 0), 8)]))
    story.append(qr_table)
    doc.build(story)
    return buf.getvalue()
