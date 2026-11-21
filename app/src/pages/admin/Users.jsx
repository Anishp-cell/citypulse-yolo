import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useSetRole, useUsers } from '../../api/hooks';
import { useAuth } from '../../hooks/useAuth';
import { toast } from '../../components/Toaster';
import { ErrorState, Loading } from '../../components/ui';
import { errorMessage, formatDate } from '../../lib/format';

const ROLES = ['citizen', 'officer', 'admin'];
const PAGE = 50;

export default function Users() {
  const { t, i18n } = useTranslation();
  const { user: me } = useAuth();
  const [q, setQ] = useState('');
  const [offset, setOffset] = useState(0);
  const { data, isPending, error, refetch } = useUsers(q, offset);
  const setRole = useSetRole();
  const change = async (id, role) => {
    try { await setRole.mutateAsync({ id, role }); } catch (err) { toast(errorMessage(t, err)); }
  };
  return (
    <div className="stack">
      <h1>{t('admin.users')}</h1>
      <input className="input" style={{ maxWidth: 360 }} placeholder={t('admin.search_users')} aria-label={t('admin.search_users')}
        value={q} onChange={(e) => { setQ(e.target.value); setOffset(0); }} />
      {isPending && <Loading />}
      {error && <ErrorState error={error} onRetry={refetch} />}
      {data && (
        <div className="card table-wrap" style={{ padding: 0 }}>
          <table className="table">
            <thead><tr><th>{t('auth.email')}</th><th>{t('auth.name')}</th><th>{t('admin.joined')}</th><th>{t('admin.role')}</th></tr></thead>
            <tbody>
              {data.users.map((u) => (
                <tr key={u.id}>
                  <td>{u.email}</td>
                  <td>{u.full_name || '—'}</td>
                  <td>{formatDate(u.created_at, i18n.language)}</td>
                  <td>
                    <select className="select" value={u.role} disabled={u.id === me.id || setRole.isPending}
                      aria-label={`${t('admin.role')} ${u.email}`} onChange={(e) => change(u.id, e.target.value)}>
                      {ROLES.map((r) => <option key={r} value={r}>{t(`admin.roles.${r}`)}</option>)}
                    </select>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {data && data.total > PAGE && (
        <div className="row">
          <button className="btn btn-sm btn-outline" disabled={offset === 0} onClick={() => setOffset(offset - PAGE)}>{t('admin.page_prev')}</button>
          <button className="btn btn-sm btn-outline" disabled={offset + PAGE >= data.total} onClick={() => setOffset(offset + PAGE)}>{t('admin.page_next')}</button>
        </div>
      )}
    </div>
  );
}
