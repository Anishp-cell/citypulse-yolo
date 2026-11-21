import { useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { Bell, BellOff, Camera, Check, ExternalLink, Flag, MapPin, Share2, ShieldCheck, Users, X } from 'lucide-react';
import { useFlag, useFollow, useHazard, useVerify } from '../api/hooks';
import { useAuth } from '../hooks/useAuth';
import ComplaintSheet from '../components/ComplaintSheet';
import { toast } from '../components/Toaster';
import { ErrorState, Loading, Notice, Sheet, Spinner, StatusBadge, TypeLabel } from '../components/ui';
import { compressImage } from '../lib/compressImage';
import { errorMessage, formatDate, formatDateTime, placeName } from '../lib/format';
import { googleMapsUrl } from '../lib/geo';
import { sharePage } from '../lib/share';

const FLAG_REASONS = ['not_a_hazard', 'duplicate', 'wrong_location', 'inappropriate_photo', 'other'];

function VerifyPanel({ hazard, onNeedLogin }) {
  const { t } = useTranslation();
  const verify = useVerify(hazard.id);
  const [photo, setPhoto] = useState(null);

  if (hazard.my_vote) {
    return <Notice tone="ok" icon={Check}>{t(hazard.my_vote === 'fixed' ? 'hazard.voted_fixed' : 'hazard.voted_not_fixed')}</Notice>;
  }
  const vote = async (isFixed) => {
    if (onNeedLogin()) return;
    try {
      await verify.mutateAsync({ isFixed, photo });
      toast(t('hazard.verify_thanks'));
    } catch (err) {
      toast(errorMessage(t, err));
    }
  };
  const pick = async (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    try { setPhoto(await compressImage(f)); } catch (err) { toast(errorMessage(t, err)); }
  };
  return (
    <section className="card stack" style={{ borderColor: 'var(--brand)' }} aria-labelledby="verify-title">
      <h2 id="verify-title">{t('hazard.verify_title')}</h2>
      <p className="small muted">{t('hazard.verify_desc')}</p>
      <label className="btn btn-ghost btn-sm" style={{ alignSelf: 'flex-start' }}>
        <Camera size={16} />{photo ? photo.name : t('hazard.verify_photo')}
        <input type="file" accept="image/*" capture="environment" hidden onChange={pick} />
      </label>
      <div className="choice-grid">
        <button className="btn btn-ok" disabled={verify.isPending} onClick={() => vote(true)}><Check size={18} />{t('hazard.verify_yes')}</button>
        <button className="btn btn-danger" disabled={verify.isPending} onClick={() => vote(false)}><X size={18} />{t('hazard.verify_no')}</button>
      </div>
    </section>
  );
}

export default function Hazard() {
  const { id } = useParams();
  const { t, i18n } = useTranslation();
  const navigate = useNavigate();
  const { user } = useAuth();
  const { data: h, isPending, error, refetch } = useHazard(id);
  const follow = useFollow(id);
  const flag = useFlag(id);
  const markFixed = useVerify(id);
  const [complaintOpen, setComplaintOpen] = useState(false);
  const [flagOpen, setFlagOpen] = useState(false);

  if (isPending) return <Loading />;
  if (error) return <ErrorState error={error} onRetry={error.status === 404 ? undefined : refetch} />;

  const needLogin = () => {
    if (user) return false;
    navigate(`/login?next=/h/${id}`);
    return true;
  };
  const days = t('common.days', { count: h.days_open });
  const place = placeName(h);
  const share = () => sharePage({
    title: t(`types.${h.type}`),
    text: t('hazard.share_text', { type: t(`types.${h.type}`), place, days }),
    url: h.share_url,
  }).then((r) => r === 'copied' && toast(t('common.copied')));
  const toggleFollow = async () => {
    if (needLogin()) return;
    try { await follow.mutateAsync(!h.is_following); } catch (err) { toast(errorMessage(t, err)); }
  };
  const sendFlag = async (reason) => {
    try { await flag.mutateAsync(reason); toast(t('hazard.flag_sent')); } catch (err) { toast(errorMessage(t, err)); }
    setFlagOpen(false);
  };
  const reportFixed = async () => {
    if (needLogin()) return;
    try { await markFixed.mutateAsync({ isFixed: true }); toast(t('hazard.verify_thanks')); } catch (err) { toast(errorMessage(t, err)); }
  };

  return (
    <article className="stack">
      {h.photo_url ? <img className="hazard-hero" src={h.photo_url} alt={t(`types.${h.type}`)} />
        : <div className="hazard-hero empty">{t('hazard.no_photo')}</div>}

      <header className="stack-sm">
        <div className="spread">
          <strong><TypeLabel type={h.type} /></strong>
          <StatusBadge status={h.status} />
        </div>
        <h1>{place}</h1>
        {h.address && <p className="small muted">{h.address}</p>}
        <div className="row-wrap small muted">
          <span>{t(h.status === 'fixed' ? 'hazard.fixed_after' : 'hazard.open_for', { days })}</span>
          <span aria-hidden="true">·</span>
          <span className="row" style={{ gap: 4 }}><Users size={14} aria-hidden="true" />
            {t('hazard.reports', { people: t('common.people', { count: h.report_count }) })}</span>
          <span className={`badge tone-${h.severity === 'high' ? 'danger' : h.severity === 'medium' ? 'warn' : 'neutral'}`}>
            {t(`severity.${h.severity}`)}</span>
          {h.ai_verified && <span className="badge tone-brand"><ShieldCheck size={12} />{t('hazard.ai_verified')}</span>}
        </div>
        {h.official_ticket_id && <p className="small"><strong>{t('hazard.ticket', { ticket: h.official_ticket_id, portal: h.official_portal })}</strong></p>}
      </header>

      {h.status === 'pending_review' && <Notice tone="warn">{t('hazard.pending_note')}</Notice>}
      {h.status === 'rejected' && <Notice tone="neutral">{t('hazard.rejected_note')}</Notice>}
      {h.status === 'fixed_unverified' && <VerifyPanel hazard={h} onNeedLogin={needLogin} />}

      {(h.is_active || h.status === 'pending_review') && (
        <div className="stack-sm">
          {h.is_active && (
            <button className="btn btn-primary btn-lg btn-block" onClick={() => setComplaintOpen(true)}>
              {t('hazard.send_to_city')}
            </button>
          )}
          <div className="choice-grid">
            <button className="btn btn-outline" onClick={toggleFollow} aria-pressed={h.is_following} disabled={follow.isPending}>
              {h.is_following ? <><BellOff size={18} />{t('hazard.following')}</> : <><Bell size={18} />{t('hazard.follow')}</>}
            </button>
            <button className="btn btn-outline" onClick={share}><Share2 size={18} />{t('hazard.share_page')}</button>
          </div>
          {h.is_active && (
            <button className="btn btn-ghost btn-sm" onClick={reportFixed} disabled={markFixed.isPending}>
              <Check size={16} />{t('hazard.report_fixed')}
            </button>
          )}
        </div>
      )}
      {!h.is_active && h.status !== 'pending_review' && (
        <button className="btn btn-outline" onClick={share}><Share2 size={18} />{t('hazard.share_page')}</button>
      )}

      <section className="card stack" aria-labelledby="hist">
        <h2 id="hist">{t('hazard.timeline')}</h2>
        <ol className="timeline">
          {h.timeline.map((e, i) => (
            <li key={i}>
              <span className="dot" aria-hidden="true" />
              <div>
                <strong>{t(`status.${e.status}`)}</strong>
                <div className="tiny">{formatDateTime(e.at, i18n.language)}</div>
                {e.note && e.note !== 'reported' && <div className="small muted">{e.note}</div>}
              </div>
            </li>
          ))}
        </ol>
        <p className="tiny">{t('hazard.first_reported', { date: formatDate(h.first_reported_at, i18n.language) })}</p>
      </section>

      <section className="card stack-sm" aria-labelledby="loc">
        <h2 id="loc" className="row"><MapPin size={18} />{t('hazard.location')}</h2>
        <p className="small muted">{h.lat.toFixed(5)}, {h.lng.toFixed(5)}</p>
        <a className="btn btn-outline btn-sm" href={googleMapsUrl(h.lat, h.lng)} target="_blank" rel="noreferrer">
          <ExternalLink size={16} />{t('hazard.open_in_maps')}
        </a>
      </section>

      {h.full_photos?.length > 0 && (
        <section className="card stack-sm">
          <h3>{t('hazard.your_photos')}</h3>
          <div className="row-wrap">
            {h.full_photos.map((u) => <a key={u} href={u} target="_blank" rel="noreferrer"><img className="hazard-thumb" src={u} alt="" /></a>)}
          </div>
        </section>
      )}

      {user && h.status !== 'rejected' && (
        <button className="btn btn-ghost btn-sm" onClick={() => setFlagOpen(true)}><Flag size={16} />{t('hazard.flag')}</button>
      )}
      <Link to="/map" className="tiny center">{t('map.title')}</Link>

      <ComplaintSheet hazardId={h.id} open={complaintOpen} onClose={() => setComplaintOpen(false)} canFile={Boolean(user)} />
      <Sheet open={flagOpen} onClose={() => setFlagOpen(false)} title={t('hazard.flag_title')}>
        <div className="stack-sm">
          {FLAG_REASONS.map((r) => (
            <button key={r} className="btn btn-outline btn-block" disabled={flag.isPending} onClick={() => sendFlag(r)}>
              {flag.isPending ? <Spinner /> : t(`hazard.flag_reasons.${r}`)}
            </button>
          ))}
        </div>
      </Sheet>
    </article>
  );
}
