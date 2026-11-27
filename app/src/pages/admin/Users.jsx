import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Search } from 'lucide-react';
import { useSetRole, useUsers } from '../../api/hooks';
import { useAuth } from '../../hooks/useAuth';
import { toast } from '../../components/Toaster';
import { ErrorState } from '../../components/ui';
import { errorMessage, formatDate } from '../../lib/format';
import { PageHeader, SkeletonPanel } from './parts';
import { initials } from './consoleUtils';

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
    <>
      <PageHeader title={t('admin.users')} description={t('admin.users_desc')}>
        <label className="input-icon">
          <Search size={15} aria-hidden="true" />
          <input className="input" style={{ width: 280 }} placeholder={t('admin.search_users')} aria-label={t('admin.search_users')}
            value={q} onChange={(e) => { setQ(e.target.value); setOffset(0); }} />
        </label>
      </PageHeader>
      {isPending && <SkeletonPanel lines={6} />}
      {error && <ErrorState error={error} onRetry={refetch} />}
      {data && (
        <div className="panel">
          <div className="table-wrap">
            <table className="data-table">
              <thead><tr><th>{t('auth.name')}</th><th>{t('admin.joined')}</th><th>{t('admin.role')}</th></tr></thead>
              <tbody>
                {data.users.map((u) => (
                  <tr key={u.id}>
                    <td>
                      <div className="cell-place">
                        <span className="avatar" aria-hidden="true">{initials(u)}</span>
                        <div style={{ minWidth: 0 }}>
                          <div style={{ fontWeight: 600 }}>{u.full_name || '—'}</div>
                          <div className="cell-sub">{u.email}</div>
                        </div>
                      </div>
                    </td>
                    <td className="tabular">{formatDate(u.created_at, i18n.language)}</td>
                    <td>
                      <select className="select" style={{ width: 'auto' }} value={u.role} disabled={u.id === me.id || setRole.isPending}
                        aria-label={`${t('admin.role')} ${u.email}`} onChange={(e) => change(u.id, e.target.value)}>
                        {ROLES.map((r) => <option key={r} value={r}>{t(`admin.roles.${r}`)}</option>)}
                      </select>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="pager">
            <span className="tabular">{t('admin.showing', { from: data.total ? offset + 1 : 0, to: Math.min(offset + PAGE, data.total), total: data.total })}</span>
            {data.total > PAGE && (
              <div className="row">
                <button className="btn btn-sm btn-outline" disabled={offset === 0} onClick={() => setOffset(offset - PAGE)}>{t('admin.page_prev')}</button>
                <button className="btn btn-sm btn-outline" disabled={offset + PAGE >= data.total} onClick={() => setOffset(offset + PAGE)}>{t('admin.page_next')}</button>
              </div>
            )}
          </div>
        </div>
      )}
    </>
  );
}
