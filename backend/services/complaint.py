"""The complaint pack: ready-to-send text (en/hi/mr) the citizen shares with their city themselves."""

from backend.cities import CityConfig, city_by_name, city_for
from backend.models import Hazard, aware
from backend.services.present import days_open

LANGS = ("en", "hi", "mr")

TYPE_LABELS = {
    "en": {"pothole": "pothole", "open_manhole": "open manhole", "waterlogging": "waterlogging",
           "debris": "road obstruction", "other": "road hazard"},
    "hi": {"pothole": "गड्ढा", "open_manhole": "खुला मैनहोल", "waterlogging": "जलभराव",
           "debris": "सड़क पर रुकावट", "other": "सड़क खतरा"},
    "mr": {"pothole": "खड्डा", "open_manhole": "उघडे मॅनहोल", "waterlogging": "पाणी साचणे",
           "debris": "रस्त्यावरील अडथळा", "other": "रस्त्यावरील धोका"},
}
SEVERITY_LABELS = {
    "en": {"low": "low", "medium": "medium", "high": "high"},
    "hi": {"low": "कम", "medium": "मध्यम", "high": "गंभीर"},
    "mr": {"low": "कमी", "medium": "मध्यम", "high": "गंभीर"},
}

TEMPLATES = {
    "en": {
        "subject": "Road hazard complaint: {type} on {place} (CityPulse #{id})",
        "body": ("To: {authority}\n\n"
                 "I would like to report a {type} that needs urgent repair.\n\n"
                 "Location: {address}\nMap: {map}\nWard: {ward}\nSeverity: {severity}\n"
                 "Reported by {count} citizen(s), first on {date} ({days} days ago).\n\n"
                 "Photos and live status: {share}\n\n"
                 "Please inspect and repair it, and share the complaint number so progress can be tracked.\n\n"
                 "Thank you."),
        "whatsapp": "{type_cap} on {place} ({ward}) has been open for {days} days. Map: {map} · Photos: {share}",
        "escalation": ("To: {escalation}\n\n"
                       "A {type} on {place} ({ward}, {city}) was reported to {authority}{ticket} but is still not "
                       "repaired after {days} days. It has been reported by {count} citizen(s).\n\n"
                       "Location: {map}\nPhotos and full history: {share}\n\n"
                       "I request you to direct the concerned department to repair it."),
        "ticket": " (complaint no. {ticket})",
    },
    "hi": {
        "subject": "सड़क खतरे की शिकायत: {place} पर {type} (CityPulse #{id})",
        "body": ("प्रति: {authority}\n\n"
                 "मैं एक {type} की शिकायत दर्ज करना चाहता/चाहती हूँ, जिसकी तुरंत मरम्मत ज़रूरी है।\n\n"
                 "स्थान: {address}\nनक्शा: {map}\nवार्ड: {ward}\nगंभीरता: {severity}\n"
                 "{count} नागरिकों ने इसकी सूचना दी है, पहली बार {date} को ({days} दिन पहले)।\n\n"
                 "फ़ोटो और ताज़ा स्थिति: {share}\n\n"
                 "कृपया जाँच कर इसकी मरम्मत कराएँ और शिकायत संख्या बताएँ ताकि प्रगति देखी जा सके।\n\n"
                 "धन्यवाद।"),
        "whatsapp": "{place} ({ward}) पर {type} {days} दिनों से ठीक नहीं हुआ है। नक्शा: {map} · फ़ोटो: {share}",
        "escalation": ("प्रति: {escalation}\n\n"
                       "{place} ({ward}, {city}) पर {type} की शिकायत {authority} को दी गई थी{ticket}, लेकिन {days} "
                       "दिनों बाद भी मरम्मत नहीं हुई है। {count} नागरिकों ने इसकी सूचना दी है।\n\n"
                       "स्थान: {map}\nफ़ोटो और पूरा इतिहास: {share}\n\n"
                       "कृपया संबंधित विभाग को इसकी मरम्मत का निर्देश दें।"),
        "ticket": " (शिकायत संख्या {ticket})",
    },
    "mr": {
        "subject": "रस्त्यावरील धोक्याची तक्रार: {place} येथे {type} (CityPulse #{id})",
        "body": ("प्रति: {authority}\n\n"
                 "मला एका {type} बद्दल तक्रार करायची आहे, ज्याची तातडीने दुरुस्ती आवश्यक आहे.\n\n"
                 "ठिकाण: {address}\nनकाशा: {map}\nप्रभाग: {ward}\nतीव्रता: {severity}\n"
                 "{count} नागरिकांनी याची नोंद केली आहे, पहिल्यांदा {date} रोजी ({days} दिवसांपूर्वी).\n\n"
                 "फोटो आणि सद्यस्थिती: {share}\n\n"
                 "कृपया पाहणी करून दुरुस्ती करावी आणि तक्रार क्रमांक कळवावा, जेणेकरून प्रगती पाहता येईल.\n\n"
                 "धन्यवाद."),
        "whatsapp": "{place} ({ward}) येथील {type} {days} दिवसांपासून दुरुस्त झालेला नाही. नकाशा: {map} · फोटो: {share}",
        "escalation": ("प्रति: {escalation}\n\n"
                       "{place} ({ward}, {city}) येथील {type} ची तक्रार {authority} कडे केली होती{ticket}, परंतु {days} "
                       "दिवसांनंतरही दुरुस्ती झालेली नाही. {count} नागरिकांनी याची नोंद केली आहे.\n\n"
                       "ठिकाण: {map}\nफोटो आणि संपूर्ण इतिहास: {share}\n\n"
                       "कृपया संबंधित विभागाला दुरुस्तीचे निर्देश द्यावेत."),
        "ticket": " (तक्रार क्र. {ticket})",
    },
}


def city_of(h: Hazard) -> CityConfig:
    by_name = city_by_name(h.city)
    return by_name if by_name.slug != "generic" else city_for(h.lat, h.lng)


def map_link(h: Hazard) -> str:
    return f"https://maps.google.com/?q={h.lat:.6f},{h.lng:.6f}"


def build_complaint(h: Hazard, lang: str, app_url: str) -> dict:
    city = city_of(h)
    t = TEMPLATES[lang]
    type_label = TYPE_LABELS[lang].get(h.type, TYPE_LABELS[lang]["other"])
    share = f"{app_url.rstrip('/')}/h/{h.id}"
    values = {
        "id": h.id, "type": type_label, "type_cap": type_label[:1].upper() + type_label[1:],
        "place": h.road or h.ward or (h.address or "").split(",")[0] or f"{h.lat:.5f}, {h.lng:.5f}",
        "address": h.address or f"{h.lat:.5f}, {h.lng:.5f}", "map": map_link(h), "ward": h.ward or "-",
        "city": h.city or city.name, "severity": SEVERITY_LABELS[lang].get(h.severity, h.severity),
        "count": h.report_count, "date": aware(h.first_reported_at).strftime("%d %b %Y"), "days": days_open(h),
        "share": share, "authority": city.authority, "escalation": city.escalation_name or "",
        "ticket": t["ticket"].format(ticket=h.official_ticket_id) if h.official_ticket_id else "",
    }
    escalation = None
    if city.escalation_name:
        escalation = {"name": city.escalation_name, "url": city.escalation_url,
                      "body": t["escalation"].format(**values)}
    return {
        "lang": lang,
        "subject": t["subject"].format(**values),
        "body": t["body"].format(**values),
        "whatsapp_text": t["whatsapp"].format(**values),
        "authority": city.authority,
        "email_to": city.grievance_email,
        "portal_name": city.portal_name,
        "portal_url": city.portal_url,
        "helpline": city.helpline,
        "whatsapp_number": city.whatsapp,
        "contacts_verified": city.verified,
        "pdf_url": f"/api/hazards/{h.id}/complaint.pdf",
        "share_url": share,
        "escalation": escalation,
    }
