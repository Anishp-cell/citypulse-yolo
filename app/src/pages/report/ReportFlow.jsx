import { useEffect, useRef, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useQueryClient } from '@tanstack/react-query';
import { useTranslation } from 'react-i18next';
import { useMapEvents } from 'react-leaflet';
import { Camera, CircleCheck, CloudOff, Image as ImageIcon, MapPin, ShieldCheck, TriangleAlert } from 'lucide-react';
import { analyzePhoto, submitReport } from '../../api/hooks';
import { useAuth } from '../../hooks/useAuth';
import ComplaintSheet from '../../components/ComplaintSheet';
import MapView from '../../components/MapView';
import PushPrompt from '../../components/PushPrompt';
import { toast } from '../../components/Toaster';
import { Notice, Spinner, TypeLabel } from '../../components/ui';
import { compressImage } from '../../lib/compressImage';
import { errorMessage } from '../../lib/format';
import { DEFAULT_CENTER, getPosition } from '../../lib/geo';
import { enqueue } from '../../lib/offlineQueue';
import { HAZARD_TYPES, SEVERITIES } from '../../lib/status';

const GOOD_ACCURACY_M = 100;
const isOffline = (err) => !navigator.onLine || err?.code === 'network_error';

function Steps({ step }) {
  const order = ['photo', 'location', 'review'];
  const idx = Math.max(0, order.indexOf(step));
  return <div className="steps" aria-hidden="true">{order.map((s, i) => <span key={s} className={i <= idx ? 'on' : ''} />)}</div>;
}

function PhotoStep({ onPicked }) {
  const { t } = useTranslation();
  const [busy, setBusy] = useState(false);
  const pick = async (e) => {
    const file = e.target.files?.[0];
    e.target.value = '';
    if (!file) return;
    setBusy(true);
    try {
      onPicked(await compressImage(file));
    } catch (err) {
      toast(errorMessage(t, err));
    } finally {
      setBusy(false);
    }
  };
  if (busy) return <div className="card center stack"><Spinner /><p>{t('report.processing')}</p></div>;
  return (
    <div className="photo-pick">
      <label className="btn btn-primary btn-lg btn-block">
        <Camera size={22} />{t('report.take_photo')}
        <input type="file" accept="image/*" capture="environment" hidden onChange={pick} data-testid="camera-input" />
      </label>
      <label className="btn btn-outline btn-block">
        <ImageIcon size={20} />{t('report.choose_photo')}
        <input type="file" accept="image/jpeg,image/png,image/webp,image/heic,image/heif" hidden onChange={pick} data-testid="gallery-input" />
      </label>
      <p className="tiny center">{t('report.photo_hint')}</p>
    </div>
  );
}

function PinTracker({ onPin, onMoved }) {
  const map = useMapEvents({
    dragstart: () => onMoved(),
    zoomstart: () => onMoved(),
    moveend: () => onPin(map.getCenter()),
  });
  return null;
}

function LocationStep({ gps, gpsError, onConfirm, busy }) {
  const { t } = useTranslation();
  const precise = gps && gps.accuracy <= GOOD_ACCURACY_M;
  const start = gps || DEFAULT_CENTER;
  const [pin, setPin] = useState(start);
  const [moved, setMoved] = useState(false);
  const pinRequired = !precise;
  const ready = !pinRequired || moved;
  const locating = !gps && !gpsError;

  return (
    <div className="stack">
      <h2>{t('report.location_title')}</h2>
      {locating && <p className="row small muted"><Spinner />{t('report.locating')}</p>}
      {gpsError && <Notice tone="warn" icon={TriangleAlert}>{t('report.gps_denied')}</Notice>}
      {gps && !precise && <Notice tone="warn" icon={TriangleAlert}>{t('report.gps_poor')}</Notice>}
      {gps && <p className="tiny">{t('report.accuracy', { m: Math.round(gps.accuracy) })}</p>}
      {!locating && (
        <div className="pin-map" data-testid="pin-map">
          <MapView center={start} zoom={gps ? 18 : 13} cluster={false}>
            <PinTracker onPin={(c) => setPin({ lat: c.lat, lng: c.lng })} onMoved={() => setMoved(true)} />
          </MapView>
          <div className="pin-center" aria-hidden="true"><MapPin size={40} fill="currentColor" stroke="#fff" /></div>
        </div>
      )}
      <p className="tiny center">{t('report.pin_hint')}</p>
      <button className="btn btn-primary btn-lg btn-block" disabled={!ready || busy || locating}
        onClick={() => onConfirm({ lat: pin.lat, lng: pin.lng, accuracy: moved ? null : gps?.accuracy,
          source: moved || !gps ? 'manual' : 'gps' })}>
        {busy ? <><Spinner />{t('report.analyzing')}</> : ready ? t('report.use_location') : t('report.pin_required')}
      </button>
    </div>
  );
}

function ReviewStep({ analysis, offline, onSubmit, busy }) {
  const { t } = useTranslation();
  const ai = analysis?.ai;
  const [type, setType] = useState('pothole');
  const [severity, setSeverity] = useState(ai?.is_pothole ? ai.severity_suggestion : 'medium');
  const [note, setNote] = useState('');
  const nearby = analysis?.nearby || [];
  const submit = (attachTo) => onSubmit({ type, severity, note, attachTo });

  return (
    <div className="stack">
      {analysis?.preview_url && <img className="preview" src={analysis.preview_url} alt="" />}
      {offline && <Notice tone="warn" icon={CloudOff}>{t('report.queued')}</Notice>}
      {ai && (ai.is_pothole
        ? <Notice tone="ok" icon={ShieldCheck}>{t('report.ai_found', { pct: Math.round(ai.confidence * 100) })}</Notice>
        : <Notice tone="info">{t('report.ai_not_found')}</Notice>)}
      {analysis?.place?.display && <p className="small muted row"><MapPin size={14} />{analysis.place.display}</p>}

      {nearby.length > 0 && (
        <section className="card stack" style={{ borderColor: 'var(--warn)' }}>
          <h3>{t('report.nearby_title')}</h3>
          {nearby.map((n) => (
            <div key={n.id} className="stack-sm">
              <div className="row">
                {n.thumb_url && <img className="hazard-thumb" src={n.thumb_url} alt="" />}
                <span className="small">{t('report.nearby_item', { type: t(`types.${n.type}`),
                  distance: t('common.meters', { count: Math.round(n.distance_m) }),
                  people: t('common.people', { count: n.report_count }) })}</span>
              </div>
              <button className="btn btn-primary" disabled={busy} onClick={() => submit(n.id)}>{t('report.same_one')}</button>
            </div>
          ))}
          <p className="tiny">{t('report.its_new')} ↓</p>
        </section>
      )}

      <fieldset className="stack-sm" style={{ border: 0, padding: 0, margin: 0 }}>
        <legend style={{ fontWeight: 600, marginBottom: 6 }}>{t('report.type_label')}</legend>
        <div className="choice-grid">
          {HAZARD_TYPES.map(({ value }) => (
            <button key={value} type="button" className="choice" aria-pressed={type === value} onClick={() => setType(value)}>
              <TypeLabel type={value} size={18} />
            </button>
          ))}
        </div>
      </fieldset>
      <fieldset className="stack-sm" style={{ border: 0, padding: 0, margin: 0 }}>
        <legend style={{ fontWeight: 600, marginBottom: 6 }}>{t('report.severity_label')}</legend>
        <div className="chips">
          {SEVERITIES.map((s) => (
            <button key={s} type="button" className="chip" aria-pressed={severity === s} onClick={() => setSeverity(s)}>
              {t(`severity.${s}`)}
            </button>
          ))}
        </div>
      </fieldset>
      <div className="field">
        <label htmlFor="note">{t('report.note_label')} <span className="tiny">({t('common.optional')})</span></label>
        <textarea id="note" className="textarea" maxLength={500} value={note} placeholder={t('report.note_placeholder')}
          onChange={(e) => setNote(e.target.value)} />
      </div>
      <p className="tiny">{t('report.privacy_note')}</p>
      <button className="btn btn-primary btn-lg btn-block" disabled={busy} onClick={() => submit(null)}>
        {busy ? <><Spinner />{t('report.submitting')}</> : t('report.submit')}
      </button>
    </div>
  );
}

function DoneStep({ result, onAnother }) {
  const { t } = useTranslation();
  const [complaintOpen, setComplaintOpen] = useState(false);
  const pending = result.status === 'pending_review';
  return (
    <div className="stack">
      <div className="card stack center">
        <CircleCheck size={48} color="var(--ok)" style={{ margin: '0 auto' }} aria-hidden="true" />
        <h1>{t(result.created_new ? 'report.done_new' : 'report.done_plus')}</h1>
        <p className="muted">{t(pending ? 'report.done_pending' : 'report.done_next')}</p>
      </div>
      {!pending && (
        <button className="btn btn-primary btn-lg btn-block" onClick={() => setComplaintOpen(true)}>{t('hazard.send_to_city')}</button>
      )}
      <Link className="btn btn-outline btn-block" to={`/h/${result.hazard_id}`}>{t('report.view_hazard')}</Link>
      <button className="btn btn-ghost" onClick={onAnother}>{t('report.report_another')}</button>
      <PushPrompt />
      <ComplaintSheet hazardId={result.hazard_id} open={complaintOpen} onClose={() => setComplaintOpen(false)} />
    </div>
  );
}

export default function ReportFlow() {
  const { t } = useTranslation();
  const { user, loading } = useAuth();
  const navigate = useNavigate();
  const qc = useQueryClient();
  const [step, setStep] = useState('photo');
  const [file, setFile] = useState(null);
  const [gps, setGps] = useState(null);
  const [gpsError, setGpsError] = useState(null);
  const [location, setLocation] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [offline, setOffline] = useState(false);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const idemKey = useRef(crypto.randomUUID());

  useEffect(() => {
    if (step !== 'location' || gps || gpsError) return;
    getPosition().then(setGps).catch((e) => setGpsError(e.code || 'unavailable'));
  }, [step, gps, gpsError]);

  if (loading) return null;
  if (!user) {
    return (
      <div className="card stack center">
        <h1>{t('report.title')}</h1>
        <p className="muted">{t('report.signin_needed')}</p>
        <button className="btn btn-primary btn-lg" onClick={() => navigate('/login?next=/report')}>{t('nav.login')}</button>
      </div>
    );
  }

  const reset = () => {
    setStep('photo'); setFile(null); setLocation(null); setAnalysis(null); setOffline(false); setResult(null);
    idemKey.current = crypto.randomUUID();
  };

  const confirmLocation = async (loc) => {
    setLocation(loc);
    if (!navigator.onLine) { setOffline(true); setStep('review'); return; }
    setBusy(true);
    try {
      setAnalysis(await analyzePhoto({ file, ...loc }));
      setStep('review');
    } catch (err) {
      if (isOffline(err)) { setOffline(true); setStep('review'); } else toast(errorMessage(t, err));
    } finally {
      setBusy(false);
    }
  };

  const queueIt = async (details) => {
    await enqueue({ idempotencyKey: idemKey.current, bytes: await file.arrayBuffer(), mime: file.type,
      lat: location.lat, lng: location.lng, accuracy: location.accuracy, source: location.source,
      type: details.type, severity: details.severity, note: details.note });
    setStep('queued');
  };

  const submit = async (details) => {
    if (offline || !analysis) { await queueIt(details); return; }
    setBusy(true);
    try {
      const r = await submitReport({ uploadId: analysis.upload_id, ...details, idempotencyKey: idemKey.current });
      setResult(r);
      setStep('done');
      qc.invalidateQueries({ queryKey: ['me'] });
      qc.invalidateQueries({ queryKey: ['hazards'] });
    } catch (err) {
      if (isOffline(err)) await queueIt(details);
      else toast(errorMessage(t, err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="stack">
      {step !== 'done' && step !== 'queued' && <><h1>{t('report.title')}</h1><Steps step={step} /></>}
      {step === 'photo' && <PhotoStep onPicked={(f) => { setFile(f); setStep('location'); }} />}
      {step === 'location' && <LocationStep gps={gps} gpsError={gpsError} busy={busy} onConfirm={confirmLocation} />}
      {step === 'review' && <ReviewStep analysis={analysis} offline={offline} busy={busy} onSubmit={submit} />}
      {step === 'done' && result && <DoneStep result={result} onAnother={reset} />}
      {step === 'queued' && (
        <div className="stack">
          <div className="card stack center">
            <CloudOff size={44} color="var(--warn)" style={{ margin: '0 auto' }} aria-hidden="true" />
            <h1>{t('report.queued_title')}</h1>
            <p className="muted">{t('report.queued')}</p>
          </div>
          <button className="btn btn-outline" onClick={reset}>{t('report.report_another')}</button>
        </div>
      )}
    </div>
  );
}
