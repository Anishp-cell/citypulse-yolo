import { useState } from 'react';
import { Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { useAdminAction, useModeration } from '../../api/hooks';
import { useAuth } from '../../hooks/useAuth';
import { toast } from '../../components/Toaster';
import { Empty, ErrorState, Loading, StatusBadge, TypeLabel } from '../../components/ui';
import { errorMessage, placeName } from '../../lib/format';

const REASONS = ['duplicate', 'not_a_hazard', 'abuse'];

function Item({ hazard, isAdmin }) {
  const { t } = useTranslation();
  const action = useAdminAction();
  const [mergeInto, setMergeInto] = useState('');
  const run = async (path, body) => {
    try { await action.mutateAsync({ path, body }); } catch (err) { toast(errorMessage(t, err)); }
  };
  const canReject = isAdmin || hazard.status === 'pending_review';
  return (
    <div className="card stack">
      <div className="row" style={{ alignItems: 'flex-start' }}>
        {(hazard.photo_url || hazard.thumb_url) && <img className="hazard-thumb" style={{ width: 120, height: 120 }}
          src={hazard.photo_url || hazard.thumb_url} alt="" />}
        <div className="stack-sm grow">
          <div className="spread"><TypeLabel type={hazard.type} /><StatusBadge status={hazard.status} /></div>
          <Link to={`/h/${hazard.id}`}>#{hazard.id} · {placeName(hazard)}</Link>
          <span className="tiny">{hazard.ward} · {t('common.people', { count: hazard.report_count })}</span>
          {hazard.flags?.length > 0 && <span className="small"><strong>{t('admin.flags')}:</strong> {hazard.flags.join(', ')}</span>}
          {hazard.notes?.length > 0 && <span className="small"><strong>{t('admin.notes')}:</strong> {hazard.notes.join(' · ')}</span>}
        </div>
      </div>
      <div className="row-wrap">
        <button className="btn btn-sm btn-ok" disabled={action.isPending}
          onClick={() => run(`/api/admin/hazards/${hazard.id}/approve`)}>{t('admin.approve')}</button>
        {canReject && REASONS.map((r) => (
          <button key={r} className="btn btn-sm btn-outline" disabled={action.isPending}
            onClick={() => run(`/api/admin/hazards/${hazard.id}/reject`, { reason: r })}>
            {t('admin.reject')}: {t(`admin.reject_reasons.${r}`)}
          </button>
        ))}
        {isAdmin && (
          <form className="row" onSubmit={(e) => { e.preventDefault(); run(`/api/admin/hazards/${hazard.id}/merge`, { into_id: Number(mergeInto) }); }}>
            <input className="input" style={{ width: 120 }} inputMode="numeric" placeholder={t('admin.merge_into')}
              aria-label={t('admin.merge_into')} value={mergeInto} onChange={(e) => setMergeInto(e.target.value.replace(/\D/g, ''))} />
            <button className="btn btn-sm btn-outline" disabled={!mergeInto || action.isPending}>{t('admin.merge')}</button>
          </form>
        )}
      </div>
    </div>
  );
}

export default function Moderation() {
  const { t } = useTranslation();
  const { user } = useAuth();
  const { data, isPending, error, refetch } = useModeration();
  if (isPending) return <Loading />;
  if (error) return <ErrorState error={error} onRetry={refetch} />;
  const items = [...data.pending, ...data.flagged];
  return (
    <div className="stack">
      <h1>{t('admin.moderation')}</h1>
      {items.length === 0 && <Empty>{t('admin.empty_moderation')}</Empty>}
      {items.map((h) => <Item key={h.id} hazard={h} isAdmin={user.role === 'admin'} />)}
    </div>
  );
}
