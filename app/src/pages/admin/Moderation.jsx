import { createElement, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { Check, ChevronDown, CircleCheck, Flag, GitMerge, MapPin, MessageSquare, Users as UsersIcon } from 'lucide-react';
import { useAdminAction, useModeration } from '../../api/hooks';
import { useAuth } from '../../hooks/useAuth';
import { toast } from '../../components/Toaster';
import { ErrorState, StatusBadge } from '../../components/ui';
import { errorMessage, placeName } from '../../lib/format';
import { typeIcon } from '../../lib/status';
import { PageHeader, SkeletonPanel } from './parts';

const REASONS = ['duplicate', 'not_a_hazard', 'abuse'];

function RejectMenu({ disabled, onReject }) {
  const { t } = useTranslation();
  const ref = useRef(null);
  const choose = (r) => { ref.current.open = false; onReject(r); };
  return (
    <details className="menu" ref={ref}>
      <summary className="btn btn-sm btn-outline" aria-disabled={disabled} style={disabled ? { pointerEvents: 'none', opacity: 0.55 } : undefined}>
        {t('admin.reject')}<ChevronDown size={14} aria-hidden="true" />
      </summary>
      <div className="menu-pop" role="menu">
        {REASONS.map((r) => (
          <button key={r} type="button" role="menuitem" onClick={() => choose(r)}>{t(`admin.reject_reasons.${r}`)}</button>
        ))}
      </div>
    </details>
  );
}

function Item({ hazard, isAdmin }) {
  const { t } = useTranslation();
  const action = useAdminAction();
  const [mergeInto, setMergeInto] = useState('');
  const run = async (path, body) => {
    try { await action.mutateAsync({ path, body }); } catch (err) { toast(errorMessage(t, err)); }
  };
  const canReject = isAdmin || hazard.status === 'pending_review';
  const photo = hazard.photo_url || hazard.thumb_url;
  return (
    <article className="panel mod-card" data-hazard-id={hazard.id}>
      {photo ? <img src={photo} alt={t('admin.photo_of', { type: t(`types.${hazard.type}`) })} loading="lazy" />
        : <span className="ph" />}
      <div style={{ minWidth: 0 }}>
        <div className="mod-head">
          <Link className="mod-title" to={`/h/${hazard.id}`}>
            {createElement(typeIcon(hazard.type), { size: 16, 'aria-hidden': true, style: { verticalAlign: '-2px', marginRight: 6 } })}
            {t(`types.${hazard.type}`)} · {placeName(hazard)}
          </Link>
          <StatusBadge status={hazard.status} />
        </div>
        <div className="mod-meta">
          <span className="mono">#{hazard.id}</span>
          {hazard.ward && <span><MapPin size={13} aria-hidden="true" />{hazard.ward}</span>}
          <span><UsersIcon size={13} aria-hidden="true" />{t('common.people', { count: hazard.report_count })}</span>
        </div>
        {hazard.flags?.length > 0 && (
          <div className="mod-note"><Flag size={13} aria-hidden="true" style={{ verticalAlign: '-2px', marginRight: 6 }} />
            <strong>{t('admin.flags')}:</strong> {hazard.flags.join(', ')}</div>
        )}
        {hazard.notes?.length > 0 && (
          <div className="mod-note"><MessageSquare size={13} aria-hidden="true" style={{ verticalAlign: '-2px', marginRight: 6 }} />
            {hazard.notes.join(' · ')}</div>
        )}
        <div className="mod-actions">
          <button className="btn btn-sm btn-ok" disabled={action.isPending}
            onClick={() => run(`/api/admin/hazards/${hazard.id}/approve`)}>
            <Check size={15} aria-hidden="true" />{t('admin.approve')}
          </button>
          {canReject && <RejectMenu disabled={action.isPending}
            onReject={(r) => run(`/api/admin/hazards/${hazard.id}/reject`, { reason: r })} />}
          {isAdmin && (
            <form className="merge-form" onSubmit={(e) => { e.preventDefault(); run(`/api/admin/hazards/${hazard.id}/merge`, { into_id: Number(mergeInto) }); }}>
              <input className="input" inputMode="numeric" placeholder={t('admin.merge_into')}
                aria-label={t('admin.merge_into')} value={mergeInto} onChange={(e) => setMergeInto(e.target.value.replace(/\D/g, ''))} />
              <button className="btn btn-sm btn-outline" disabled={!mergeInto || action.isPending}>
                <GitMerge size={14} aria-hidden="true" />{t('admin.merge')}
              </button>
            </form>
          )}
        </div>
      </div>
    </article>
  );
}

export default function Moderation() {
  const { t } = useTranslation();
  const { user } = useAuth();
  const { data, isPending, error, refetch } = useModeration();
  const items = data ? [...data.pending, ...data.flagged] : [];
  return (
    <>
      <PageHeader title={t('admin.moderation')} description={t('admin.moderation_desc')} />
      {isPending && <><SkeletonPanel height={148} /><SkeletonPanel height={148} /></>}
      {error && <ErrorState error={error} onRetry={refetch} />}
      {data && items.length === 0 && (
        <div className="panel empty-panel"><CircleCheck size={28} aria-hidden="true" /><strong>{t('admin.empty_moderation')}</strong></div>
      )}
      <div className="mod-list">
        {items.map((h) => <Item key={h.id} hazard={h} isAdmin={user.role === 'admin'} />)}
      </div>
    </>
  );
}
