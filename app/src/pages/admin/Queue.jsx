import { createElement, useState } from 'react';
import { Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { CircleCheck, Search } from 'lucide-react';
import { useAdminAction, useAdminQueue } from '../../api/hooks';
import { toast } from '../../components/Toaster';
import { ErrorState, Sheet, Spinner, StatusBadge } from '../../components/ui';
import { errorMessage, placeName } from '../../lib/format';
import { typeIcon } from '../../lib/status';
import { PageHeader, PriorityMeter, SkeletonPanel } from './parts';

const PAGE = 50;
// What an officer can move a hazard to from each status (the backend enforces the real rules).
const NEXT = {
  open: ['acknowledged', 'in_progress', 'fixed_unverified'],
  filed: ['acknowledged', 'in_progress', 'fixed_unverified'],
  acknowledged: ['in_progress', 'fixed_unverified'],
  in_progress: ['fixed_unverified'],
  fixed_unverified: [],
};

function StatusSheet({ hazard, onClose }) {
  const { t } = useTranslation();
  const action = useAdminAction();
  const [to, setTo] = useState(NEXT[hazard.status]?.[0] || '');
  const [note, setNote] = useState('');
  const [photo, setPhoto] = useState(null);
  const apply = async (e) => {
    e.preventDefault();
    const form = new FormData();
    form.append('to', to);
    if (note.trim()) form.append('note', note.trim());
    if (photo) form.append('photo', photo);
    try {
      await action.mutateAsync({ path: `/api/admin/hazards/${hazard.id}/status`, form });
      onClose();
    } catch (err) {
      toast(errorMessage(t, err));
    }
  };
  return (
    <form className="stack" onSubmit={apply}>
      <div className="chips">
        {(NEXT[hazard.status] || []).map((s) => (
          <button key={s} type="button" className="chip" aria-pressed={to === s} onClick={() => setTo(s)}>{t(`status.${s}`)}</button>
        ))}
      </div>
      <div className="field">
        <label htmlFor="note">{t('admin.note')}</label>
        <textarea id="note" className="textarea" maxLength={500} value={note} onChange={(e) => setNote(e.target.value)} />
      </div>
      {to === 'fixed_unverified' && (
        <div className="field">
          <label htmlFor="proof">{t('admin.proof_photo')}</label>
          <input id="proof" type="file" accept="image/*" capture="environment" required onChange={(e) => setPhoto(e.target.files?.[0] || null)} />
        </div>
      )}
      <button className="btn btn-primary" disabled={!to || action.isPending}>{action.isPending ? <Spinner /> : t('admin.apply')}</button>
    </form>
  );
}

export default function Queue() {
  const { t } = useTranslation();
  const [ward, setWard] = useState('');
  const [status, setStatus] = useState('');
  const [sort, setSort] = useState('priority');
  const [offset, setOffset] = useState(0);
  const [editing, setEditing] = useState(null);
  const { data, isPending, error, refetch } = useAdminQueue({ ward, status, sort, offset: String(offset), limit: String(PAGE) });

  return (
    <>
      <PageHeader title={t('admin.queue')} description={t('admin.queue_desc')}>
        <label className="input-icon">
          <Search size={15} aria-hidden="true" />
          <input className="input" style={{ width: 180 }} placeholder={t('admin.filter_ward')} value={ward}
            onChange={(e) => { setWard(e.target.value); setOffset(0); }} aria-label={t('admin.filter_ward')} />
        </label>
        <select className="select" style={{ width: 'auto' }} value={status} aria-label={t('admin.filter_status')}
          onChange={(e) => { setStatus(e.target.value); setOffset(0); }}>
          <option value="">{t('admin.all_active')}</option>
          {['open', 'filed', 'acknowledged', 'in_progress', 'fixed_unverified', 'fixed'].map((s) => (
            <option key={s} value={s}>{t(`status.${s}`)}</option>))}
        </select>
        <div className="segmented" role="group" aria-label={t('admin.sort')}>
          {['priority', 'age'].map((s) => (
            <button key={s} type="button" aria-pressed={sort === s} onClick={() => { setSort(s); setOffset(0); }}>
              {t(`admin.sort_${s}`)}
            </button>
          ))}
        </div>
      </PageHeader>
      {isPending && <SkeletonPanel lines={6} />}
      {error && <ErrorState error={error} onRetry={refetch} />}
      {data && data.hazards.length === 0 && (
        <div className="panel empty-panel"><CircleCheck size={28} aria-hidden="true" /><strong>{t('admin.empty_queue')}</strong></div>
      )}
      {data && data.hazards.length > 0 && (
        <div className="panel">
          <div className="table-wrap">
            <table className="data-table">
              <thead><tr>
                <th>{t('admin.hazard')}</th><th>{t('admin.filter_status')}</th><th>{t('admin.priority')}</th>
                <th className="num">{t('admin.age')}</th><th className="num">{t('admin.reports')}</th><th><span className="sr-only">{t('admin.change_status')}</span></th>
              </tr></thead>
              <tbody>
                {data.hazards.map((h) => (
                  <tr key={h.id} data-hazard-id={h.id}>
                    <td>
                      <div className="cell-place">
                        {h.thumb_url ? <img src={h.thumb_url} alt="" loading="lazy" />
                          : <span className="ph">{createElement(typeIcon(h.type), { size: 16, 'aria-hidden': true })}</span>}
                        <div style={{ minWidth: 0 }}>
                          <Link to={`/h/${h.id}`}>{t(`types.${h.type}`)} · {placeName(h)}</Link>
                          <div className="cell-sub"><span className="mono">#{h.id}</span>{h.ward ? ` · ${h.ward}` : ''}</div>
                        </div>
                      </div>
                    </td>
                    <td><StatusBadge status={h.status} /></td>
                    <td><PriorityMeter score={h.priority_score} /></td>
                    <td className="num tabular">{t('common.days', { count: h.days_open })}</td>
                    <td className="num tabular">{h.report_count}</td>
                    <td className="num">{NEXT[h.status]?.length > 0 && (
                      <button className="btn btn-sm btn-outline" onClick={() => setEditing(h)}>{t('admin.change_status')}</button>)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="pager">
            <span className="tabular">{t('admin.showing', { from: offset + 1, to: Math.min(offset + PAGE, data.total), total: data.total })}</span>
            {data.total > PAGE && (
              <div className="row">
                <button className="btn btn-sm btn-outline" disabled={offset === 0} onClick={() => setOffset(offset - PAGE)}>{t('admin.page_prev')}</button>
                <button className="btn btn-sm btn-outline" disabled={offset + PAGE >= data.total} onClick={() => setOffset(offset + PAGE)}>{t('admin.page_next')}</button>
              </div>
            )}
          </div>
        </div>
      )}
      <Sheet open={Boolean(editing)} onClose={() => setEditing(null)} title={t('admin.change_status')}>
        {editing && <StatusSheet hazard={editing} onClose={() => setEditing(null)} />}
      </Sheet>
    </>
  );
}
