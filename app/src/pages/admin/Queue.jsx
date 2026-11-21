import { useState } from 'react';
import { Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { useAdminAction, useAdminQueue } from '../../api/hooks';
import { toast } from '../../components/Toaster';
import { Empty, ErrorState, Loading, Sheet, Spinner, StatusBadge, TypeLabel } from '../../components/ui';
import { errorMessage, placeName } from '../../lib/format';

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
    <div className="stack">
      <h1>{t('admin.queue')}</h1>
      <div className="row-wrap">
        <input className="input" style={{ maxWidth: 200 }} placeholder={t('admin.filter_ward')} value={ward}
          onChange={(e) => { setWard(e.target.value); setOffset(0); }} aria-label={t('admin.filter_ward')} />
        <select className="select" style={{ maxWidth: 200 }} value={status} aria-label={t('admin.filter_status')}
          onChange={(e) => { setStatus(e.target.value); setOffset(0); }}>
          <option value="">{t('admin.all')}</option>
          {['open', 'filed', 'acknowledged', 'in_progress', 'fixed_unverified', 'fixed'].map((s) => (
            <option key={s} value={s}>{t(`status.${s}`)}</option>))}
        </select>
        <select className="select" style={{ maxWidth: 200 }} value={sort} onChange={(e) => setSort(e.target.value)}>
          <option value="priority">{t('admin.sort_priority')}</option>
          <option value="age">{t('admin.sort_age')}</option>
        </select>
      </div>
      {isPending && <Loading />}
      {error && <ErrorState error={error} onRetry={refetch} />}
      {data && data.hazards.length === 0 && <Empty>{t('admin.empty_queue')}</Empty>}
      {data && data.hazards.length > 0 && (
        <div className="card table-wrap" style={{ padding: 0 }}>
          <table className="table">
            <thead><tr><th>#</th><th /><th>{t('hazard.location')}</th><th>{t('admin.filter_status')}</th>
              <th>{t('admin.priority')}</th><th>{t('admin.age')}</th><th>{t('admin.reports')}</th><th /></tr></thead>
            <tbody>
              {data.hazards.map((h) => (
                <tr key={h.id}>
                  <td><Link to={`/h/${h.id}`}>{h.id}</Link></td>
                  <td><TypeLabel type={h.type} size={14} /></td>
                  <td>{placeName(h)}<div className="tiny">{h.ward}</div></td>
                  <td><StatusBadge status={h.status} /></td>
                  <td><strong>{Math.round(h.priority_score)}</strong></td>
                  <td>{t('common.days', { count: h.days_open })}</td>
                  <td>{h.report_count}</td>
                  <td>{NEXT[h.status]?.length > 0 && (
                    <button className="btn btn-sm btn-outline" onClick={() => setEditing(h)}>{t('admin.change_status')}</button>)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {data && data.total > PAGE && (
        <div className="row">
          <button className="btn btn-sm btn-outline" disabled={offset === 0} onClick={() => setOffset(offset - PAGE)}>{t('admin.page_prev')}</button>
          <span className="tiny">{offset + 1}–{Math.min(offset + PAGE, data.total)} / {data.total}</span>
          <button className="btn btn-sm btn-outline" disabled={offset + PAGE >= data.total} onClick={() => setOffset(offset + PAGE)}>{t('admin.page_next')}</button>
        </div>
      )}
      <Sheet open={Boolean(editing)} onClose={() => setEditing(null)} title={t('admin.change_status')}>
        {editing && <StatusSheet hazard={editing} onClose={() => setEditing(null)} />}
      </Sheet>
    </div>
  );
}
