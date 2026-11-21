# CityPulse v1 — Design

Approved by the project owner on 2026-11-21 (plan mode). The implementation plan is `docs/superpowers/plans/2026-11-21-citypulse-v1.md`.

## Context

The project stalled about 6 months ago at a demo stage: three separate frontends, a backend that crashes (`IncidentStatus.notified` was removed but `admin.py` still uses it), fake mobile login, alerts that report success when nothing was sent, demo code that sends **fake locations** to real authorities, a Docker image that's missing dependencies, and a **privilege escalation** (anyone can register as `admin`, because `UserCreate.role` is accepted from the client). The user wants it finished properly and then closed out: a reliable app that real people would use, not a feature list.

**Decisions the user made:**
1. **Potholes first.** Citizen reporting and city accountability is the product. Accident/emergency is a secondary mode.
2. **Web only (PWA).** Expo `mobile/` and vanilla `frontend/` are retired. The React `app/` becomes the single frontend (citizen + officer/admin).
3. **Free/cheap hosting:** Render (Docker), Neon Postgres, Cloudflare R2, all free tiers. Render Starter ($7/mo) recommended at launch to avoid cold starts.
4. **Citizens send their own complaints.** The app generates the complaint pack (PDF + text + portal link) and the citizen shares it via WhatsApp/email or files it on the city portal. The server never emails government addresses. Twilio and SendGrid are removed.

**Calls I made (with reasons, all can be overridden):**
- **No AI in emergency mode.** The accident classes are weak (overall mAP50 0.57), and wrong medical advice is a liability. Users pick their situation ("someone is injured / only vehicles damaged / not sure") and get vetted static first-aid steps plus one-tap 112/108. Gemini is removed (its SDK `google-generativeai` is deprecated, and hallucination is a risk here).
- **AI assists, it doesn't decide.** The model suggests "pothole + severity" and the user confirms. If the model finds nothing, the user can still submit; the hazard goes to `pending_review` instead of being rejected outright.
- **More hazard types, chosen manually:** open manhole, waterlogging, debris/obstruction, other. These kill people in Indian monsoons. They need no ML and always go to `pending_review`.
- **Photos are images only, no video.** Video upload is removed, since it was heavy on CPU and adds nothing for potholes.
- **Privacy by cropping:** the public sees a crop around the detected pothole (plus a margin) with EXIF stripped. The full photo is visible only to the reporter and officers. This avoids face/plate exposure without needing a blur model.
- **Leaderboard and karma currency are dropped.** They're replaced with a personal "Impact" summary (reports, fixes confirmed). Roads and wards get scorecards, because public accountability is the real motivation for using the app.
- **Marathi added** (Pune pilot) next to English and Hindi.
- **Server-side geocoding** with caching (Nominatim with a proper User-Agent; switches to LocationIQ if `LOCATIONIQ_KEY` is set). This gives a reliable road and ward and respects tile/geocode usage policies.
- **Map tiles** from CARTO basemaps (free, attribution required) instead of `tile.openstreetmap.org`, whose usage policy forbids production apps.
- **Password reset email** via generic SMTP (Brevo free tier works with a single verified sender; no domain needed).

---

## Product spec (what a user experiences)

**Citizen: report (target under 60 seconds)**
1. Home → big **Report a hazard** button → camera opens (`<input type=file accept=image/* capture=environment>`; the gallery is also allowed).
2. The photo is compressed on the device (≤1600px JPEG, ~0.5–1 MB, orientation fixed, HEIC converted). GPS is captured with its accuracy shown. A mini-map lets the user drag the pin. If accuracy is worse than 100 m or GPS is denied, placing the pin is **required**; the app never invents coordinates.
3. Upload → AI result: "Looks like a pothole (87%), severity: Medium", with editable type and severity and an optional note.
4. **Nearby check:** "3 people already reported a hazard 12 m away. Is this the same one?" [Yes, add my +1] [No, it's new].
5. Done screen → **Send to PMC** sheet: share the PDF + text via WhatsApp/email (Web Share API with the file; fallback is download + `mailto:` + copy text), and an **Open PMC grievance portal** link. Then "I filed it, here's my complaint number" (optional), which moves the hazard to `filed`.
6. If offline, the report is queued in IndexedDB and sent automatically when back online, without creating duplicates (idempotency key).

**Citizen: follow up**
- **My reports:** each hazard with its status, days open and a timeline (reported → filed #PMC123 → acknowledged → fixed → confirmed).
- Push notifications (Web Push) when a followed hazard changes status, when someone says it's fixed ("Can you confirm?"), and as reminders: day 7 "Not filed yet? File it with PMC"; day 21 after filing with no progress, "Escalate" (an escalation pack addressed to the next level from the city config).
- **Verify:** "Is it fixed?" [Yes, with an optional photo] [No, still there]. Two "fixed" votes from different users confirm it; two "not fixed" votes reopen it with a priority boost.

**Public**
- **Map:** clustered pins for active hazards, filter chips (Active / Fixed / All), tap to open a bottom sheet with a link to the hazard page.
- **Hazard page `/h/:id`:** a public, shareable link with Open Graph tags (the server injects them so WhatsApp previews show a photo and "Open 47 days on FC Road").
- **City scorecard `/city`:** per ward, the number of active hazards, the median days to fix (last 90 days) and the % fixed within 30 days; plus the 10 worst roads.

**Emergency `/emergency`** (works offline): big **Call 112** / **108 Ambulance** buttons, triage choice → step-by-step first aid, and **Share my location** (Web Share with a Google Maps link). Nothing is stored.

**Officer/admin (`/admin`, same app, lazy-loaded)**
- **Work queue:** filter by ward and status, sorted by priority. Hazard detail with the full photo and timeline. Status change with a note and an optional "fixed" proof photo.
- **Moderation:** `pending_review` items and abuse flags → approve / reject (with a reason) / merge a duplicate into another hazard.
- **Analytics:** the existing Recharts page, rewired to hazards. **Users:** list and role changes (admin only).
- First admin created with `python -m backend.cli create-admin --email ...`.

**Account:** email + password, forgot/reset password, language, notification toggle, **delete account** (reports are anonymized and stay public), Privacy and Terms pages.

### Hazard status machine (single source of truth: `backend/domain/status.py`)

| From → To | Who | Trigger |
|---|---|---|
| (new) → `pending_review` | system | manual type, or AI confidence < 0.25 |
| (new) → `open` | system | AI-detected pothole ≥ 0.25 |
| `pending_review` → `open` / `rejected` | officer/admin | moderation |
| `open` → `filed` | any follower | submits official ticket id |
| `open`/`filed` → `acknowledged` → `in_progress` | officer/admin | queue |
| any active → `fixed_unverified` | officer (proof photo), or a citizen "fixed" vote | |
| `fixed_unverified` → `fixed` | system | 2 distinct "fixed" votes, or officer proof + 1 vote, or 14 days with no dispute (daily job) |
| `fixed_unverified` → `open` | system | 2 distinct "not fixed" votes (priority ×1.5) |
| any → `rejected` | admin | reason required (`duplicate`, `not_a_hazard`, `abuse`) |

"Active" = `open, filed, acknowledged, in_progress`. `rejected`/`pending_review` are never public. Every transition writes a `status_events` row.

---

