"""Web Push (VAPID). Disabled automatically when VAPID keys are not configured."""

import json
import logging

from pywebpush import WebPushException, webpush

from backend.domain.status import HazardStatus as S
from backend.models import Follow, Hazard, PushSubscription, StatusEvent, User
from backend.services.complaint import TYPE_LABELS

log = logging.getLogger("citypulse.push")
TTL_S = 24 * 3600

# (title, body) per language and event kind. {type}, {place} and {ticket} are filled in.
MESSAGES = {
    "en": {
        "filed": ("Complaint filed", "The {type} on {place} was filed with the city{ticket}."),
        "acknowledged": ("The city has seen it", "The {type} on {place} was acknowledged by the city."),
        "in_progress": ("Repair in progress", "Work has started on the {type} on {place}."),
        "fixed_unverified": ("Is it fixed? Please confirm", "Someone says the {type} on {place} is fixed. Tap to confirm."),
        "fixed": ("Fixed!", "The {type} on {place} is confirmed fixed. Thanks for helping."),
        "reopened": ("Still not fixed", "People say the {type} on {place} is still there. It has been reopened."),
        "approved": ("Your report is live", "Your report of a {type} on {place} is now public."),
        "rejected": ("Report closed", "The report of a {type} on {place} was closed by a moderator."),
        "remind_file": ("Send it to the city", "The {type} on {place} hasn't been filed with the city yet. It takes a minute."),
        "remind_escalate": ("No action yet: escalate?", "The {type} on {place} was filed 3 weeks ago with no response. Escalate it."),
    },
    "hi": {
        "filed": ("शिकायत दर्ज हुई", "{place} पर {type} की शिकायत नगर निगम में दर्ज हुई{ticket}।"),
        "acknowledged": ("नगर निगम ने देख लिया", "{place} पर {type} की शिकायत स्वीकार कर ली गई है।"),
        "in_progress": ("मरम्मत जारी है", "{place} पर {type} की मरम्मत शुरू हो गई है।"),
        "fixed_unverified": ("क्या यह ठीक हो गया? पुष्टि करें", "किसी ने बताया कि {place} पर {type} ठीक हो गया है। पुष्टि करें।"),
        "fixed": ("ठीक हो गया!", "{place} पर {type} के ठीक होने की पुष्टि हो गई। मदद के लिए धन्यवाद।"),
        "reopened": ("अभी ठीक नहीं हुआ", "लोगों के अनुसार {place} पर {type} अभी भी है। इसे फिर से खोला गया है।"),
        "approved": ("आपकी रिपोर्ट प्रकाशित हुई", "{place} पर {type} की आपकी रिपोर्ट अब सार्वजनिक है।"),
        "rejected": ("रिपोर्ट बंद की गई", "{place} पर {type} की रिपोर्ट मॉडरेटर ने बंद कर दी।"),
        "remind_file": ("नगर निगम को भेजें", "{place} पर {type} की शिकायत अभी दर्ज नहीं हुई है। इसमें एक मिनट लगेगा।"),
        "remind_escalate": ("कोई कार्रवाई नहीं: आगे भेजें?", "{place} पर {type} की शिकायत 3 हफ़्ते पहले दर्ज हुई थी, जवाब नहीं आया।"),
    },
    "mr": {
        "filed": ("तक्रार नोंदवली", "{place} येथील {type} ची तक्रार महापालिकेत नोंदवली गेली{ticket}."),
        "acknowledged": ("महापालिकेने दखल घेतली", "{place} येथील {type} ची तक्रार स्वीकारली गेली आहे."),
        "in_progress": ("दुरुस्ती सुरू आहे", "{place} येथील {type} ची दुरुस्ती सुरू झाली आहे."),
        "fixed_unverified": ("दुरुस्त झाले का? खात्री करा", "{place} येथील {type} दुरुस्त झाल्याचे कोणी सांगितले आहे. खात्री करा."),
        "fixed": ("दुरुस्त झाले!", "{place} येथील {type} दुरुस्त झाल्याची खात्री झाली. मदतीबद्दल धन्यवाद."),
        "reopened": ("अजून दुरुस्त नाही", "लोकांच्या मते {place} येथील {type} अजूनही आहे. ते पुन्हा उघडले आहे."),
        "approved": ("तुमचा अहवाल प्रसिद्ध झाला", "{place} येथील {type} चा तुमचा अहवाल आता सार्वजनिक आहे."),
        "rejected": ("अहवाल बंद केला", "{place} येथील {type} चा अहवाल मॉडरेटरने बंद केला."),
        "remind_file": ("महापालिकेकडे पाठवा", "{place} येथील {type} ची तक्रार अजून नोंदवली नाही. यास एक मिनिट लागेल."),
        "remind_escalate": ("कारवाई नाही: पुढे पाठवायचे?", "{place} येथील {type} ची तक्रार 3 आठवड्यांपूर्वी केली, उत्तर नाही."),
    },
}


def event_kind(event: StatusEvent) -> str | None:
    to, frm = event.to_status, event.from_status
    if to == S.open:
        return "reopened" if frm == S.fixed_unverified else ("approved" if frm == S.pending_review else None)
    return {S.filed: "filed", S.acknowledged: "acknowledged", S.in_progress: "in_progress",
            S.fixed_unverified: "fixed_unverified", S.fixed: "fixed", S.rejected: "rejected"}.get(to)


def message(kind: str, lang: str, hazard: Hazard) -> dict:
    lang = lang if lang in MESSAGES else "en"
    title, body = MESSAGES[lang][kind]
    values = {"type": TYPE_LABELS[lang].get(hazard.type, TYPE_LABELS[lang]["other"]),
              "place": hazard.road or hazard.ward or hazard.city or "—",
              "ticket": f" ({hazard.official_ticket_id})" if hazard.official_ticket_id else ""}
    return {"title": title, "body": body.format(**values), "url": f"/h/{hazard.id}", "tag": f"hazard-{hazard.id}"}


class PushService:
    def __init__(self, settings, session_factory, sender=webpush):
        self.settings = settings
        self.SessionLocal = session_factory
        self.sender = sender

    @property
    def enabled(self) -> bool:
        return bool(self.settings.vapid_public_key and self.settings.vapid_private_key)

    def _send(self, db, sub: PushSubscription, payload: dict) -> None:
        try:
            self.sender(subscription_info={"endpoint": sub.endpoint, "keys": {"p256dh": sub.p256dh, "auth": sub.auth}},
                        data=json.dumps(payload), vapid_private_key=self.settings.vapid_private_key,
                        vapid_claims={"sub": self.settings.vapid_subject}, ttl=TTL_S)
        except WebPushException as e:
            status = getattr(e.response, "status_code", None)
            if status in (404, 410):  # the browser unsubscribed; forget it
                db.delete(sub)
            else:
                log.warning("push to %s failed (%s): %s", sub.id, status, e)
        except Exception:
            log.exception("push to %s failed", sub.id)

    def send_to_users(self, db, user_ids, kind: str, hazard: Hazard) -> int:
        if not self.enabled or not user_ids:
            return 0
        sent = 0
        rows = (db.query(PushSubscription, User.language).join(User, User.id == PushSubscription.user_id)
                .filter(PushSubscription.user_id.in_(list(user_ids)), User.is_active.is_(True)).all())
        for sub, lang in rows:
            self._send(db, sub, message(kind, lang, hazard))
            sent += 1
        db.commit()
        return sent

    def send_for_event(self, event_id: int, exclude_user_id: int | None = None) -> int:
        if not self.enabled:
            return 0
        with self.SessionLocal() as db:
            event = db.get(StatusEvent, event_id)
            kind = event_kind(event) if event else None
            if kind is None:
                return 0
            hazard = event.hazard
            followers = {uid for (uid,) in db.query(Follow.user_id).filter(Follow.hazard_id == hazard.id)}
            followers.discard(exclude_user_id)
            return self.send_to_users(db, followers, kind, hazard)
