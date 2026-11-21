import os
import base64
from typing import Dict, Any
from pathlib import Path
from backend.service.government.base import GovernmentAdapter
from backend.service.government.pdf_generator import generate_grievance_pdf

try:
    import sendgrid
    from sendgrid.helpers.mail import Mail, Attachment, FileContent, FileName, FileType, Disposition
    _HAS_SENDGRID = True
except ImportError:
    _HAS_SENDGRID = False

class EmailPdfGovernmentAdapter(GovernmentAdapter):
    """
    Tier-2 Modular Adapter: Generates official PDF Grievance Dossier
    and dispatches via SendGrid / SMTP email to Ward Road Engineers.
    """

    def __init__(self, sender_email: str = "grievance@citypulse.ai"):
        self.sender_email = os.environ.get("SENDGRID_FROM_EMAIL", sender_email)
        self.sendgrid_api_key = os.environ.get("SENDGRID_API_KEY")

    async def submit_complaint(self, incident_data: Dict[str, Any]) -> Dict[str, Any]:
        ticket_id = incident_data.get("ticket_id") or f"CP-{incident_data.get('cluster_id', 101)}"
        road_name = incident_data.get("road_name", "Arterial Road")
        ward_name = incident_data.get("ward_name", "Central")
        city = incident_data.get("city", "Pune")
        lat = float(incident_data.get("lat", 18.5204))
        lng = float(incident_data.get("lng", 73.8567))
        severity = incident_data.get("severity", "moderate")
        priority_score = float(incident_data.get("priority_score", 65.0))
        report_count = int(incident_data.get("report_count", 1))
        image_path = incident_data.get("image_path")
        recipient_email = incident_data.get("recipient_email", os.environ.get("ALERT_EMAIL", "ward.engineer@pmc.gov.in"))

        # 1. Generate official PDF Dossier
        pdf_path = generate_grievance_pdf(
            ticket_id=ticket_id,
            road_name=road_name,
            ward_name=ward_name,
            city=city,
            lat=lat,
            lng=lng,
            severity=severity,
            priority_score=priority_score,
            report_count=report_count,
            image_path=image_path,
            sla_hours=72 if priority_score > 60 else 120,
        )

        channels_dispatched = ["PDF Generated"]
        delivery_status = "generated_local"

        # 2. Dispatch via SendGrid if configured
        if self.sendgrid_api_key and _HAS_SENDGRID:
            try:
                sg = sendgrid.SendGridAPIClient(self.sendgrid_api_key)
                subject = f"[URGENT GRIEVANCE #{ticket_id}] High Priority Road Hazard on {road_name} ({ward_name} Ward)"
                body_html = f"""
                <h3>Municipal Road Redressal Notice</h3>
                <p>An infrastructure hazard with Priority Score <strong>{priority_score:.1f}/100</strong> has been confirmed by {report_count} citizens on <strong>{road_name}</strong>.</p>
                <p>Location: <a href="https://maps.google.com/?q={lat},{lng}">{lat:.5f}, {lng:.5f} (Google Maps)</a></p>
                <p>Please find attached the official Municipal Grievance Dossier PDF for inspection and repair crew assignment.</p>
                <p><em>CityPulse Civic Technology Platform</em></p>
                """

                message = Mail(
                    from_email=self.sender_email,
                    to_emails=recipient_email,
                    subject=subject,
                    html_content=body_html,
                )

                # Attach PDF
                with open(pdf_path, "rb") as f:
                    data = f.read()
                    encoded = base64.b64encode(data).decode()
                    attachment = Attachment(
                        FileContent(encoded),
                        FileName(os.path.basename(pdf_path)),
                        FileType('application/pdf'),
                        Disposition('attachment'),
                    )
                    message.attachment = attachment

                response = sg.send(message)
                if response.status_code in (200, 202):
                    channels_dispatched.append(f"SendGrid Email to {recipient_email}")
                    delivery_status = "dispatched"
            except Exception as e:
                channels_dispatched.append(f"Email delivery error: {e}")
                delivery_status = "email_failed"

        return {
            "success": True,
            "ticket_id": ticket_id,
            "pdf_path": pdf_path,
            "status": delivery_status,
            "channels": channels_dispatched,
            "recipient": recipient_email,
        }

    async def check_status(self, ticket_id: str) -> Dict[str, Any]:
        return {
            "ticket_id": ticket_id,
            "status": "in_progress",
            "message": "Assigned to Ward Road Maintenance Crew",
        }
