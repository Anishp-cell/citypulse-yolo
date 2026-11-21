import { useTranslation } from 'react-i18next';
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { useAdminStats } from '../../api/hooks';
import { ErrorState, Loading } from '../../components/ui';
import { STATUS } from '../../lib/status';

export default function Dashboard() {
  const { t } = useTranslation();
  const { data, isPending, error, refetch } = useAdminStats();
  if (isPending) return <Loading />;
  if (error) return <ErrorState error={error} onRetry={refetch} />;
  const statusRows = Object.entries(data.by_status).filter(([s]) => s !== 'rejected')
    .map(([s, n]) => ({ name: t(`status.${s}`), n, fill: STATUS[s]?.color }));
  const typeRows = Object.entries(data.by_type).map(([k, n]) => ({ name: t(`types.${k}`), n }));
  return (
    <div className="stack">
      <h1>{t('admin.dashboard')}</h1>
      <div className="kpis">
        <div className="card kpi"><span className="tiny">{t('admin.active')}</span><strong>{data.totals.active}</strong></div>
        <div className="card kpi"><span className="tiny">{t('admin.pending')}</span><strong>{data.by_status.pending_review}</strong></div>
        <div className="card kpi"><span className="tiny">{t('admin.fixed')}</span><strong>{data.totals.fixed}</strong></div>
        <div className="card kpi"><span className="tiny">{t('admin.avg_fix')}</span><strong>{data.totals.avg_days_to_fix ?? '—'}</strong></div>
      </div>
      <section className="card stack">
        <h2>{t('admin.reports_30d')}</h2>
        <div style={{ height: 220 }}>
          <ResponsiveContainer>
            <BarChart data={data.daily_reports}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border)" />
              <XAxis dataKey="date" tickFormatter={(d) => d.slice(5)} fontSize={11} />
              <YAxis allowDecimals={false} fontSize={11} width={28} />
              <Tooltip />
              <Bar dataKey="count" fill="var(--brand)" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>
      <div className="admin-grid">
        <section className="card stack">
          <h2>{t('admin.by_status')}</h2>
          <div style={{ height: 240 }}>
            <ResponsiveContainer>
              <BarChart data={statusRows} layout="vertical" margin={{ left: 40 }}>
                <XAxis type="number" allowDecimals={false} fontSize={11} />
                <YAxis type="category" dataKey="name" fontSize={11} width={110} />
                <Tooltip />
                <Bar dataKey="n" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>
        <section className="card stack">
          <h2>{t('admin.by_type')}</h2>
          <table className="table"><tbody>
            {typeRows.map((r) => <tr key={r.name}><td>{r.name}</td><td><strong>{r.n}</strong></td></tr>)}
          </tbody></table>
        </section>
      </div>
    </div>
  );
}
