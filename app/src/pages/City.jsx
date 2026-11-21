import { useTranslation } from 'react-i18next';
import { useScorecard } from '../api/hooks';
import { useAuth } from '../hooks/useAuth';
import { Empty, ErrorState, Loading } from '../components/ui';

const healthColor = (h) => (h >= 80 ? 'var(--ok)' : h >= 50 ? 'var(--warn)' : 'var(--danger)');

export default function City() {
  const { t } = useTranslation();
  const { user } = useAuth();
  const city = user?.city || 'Pune';
  const { data, isPending, error, refetch } = useScorecard(city);
  if (isPending) return <Loading />;
  if (error) return <ErrorState error={error} onRetry={refetch} />;
  return (
    <div className="stack">
      <h1>{t('city.title', { city })}</h1>
      <p className="small muted">{t('city.explain')}</p>
      <section className="card stack">
        <h2>{t('city.wards')}</h2>
        {data.wards.length === 0 ? <Empty>{t('city.no_data')}</Empty> : (
          <div className="table-wrap">
            <table className="table">
              <thead><tr><th>{t('city.ward')}</th><th>{t('city.active')}</th><th>{t('city.median_fix')}</th><th>{t('city.pct_fixed')}</th></tr></thead>
              <tbody>
                {data.wards.map((w) => (
                  <tr key={w.ward}>
                    <td>{w.ward}</td>
                    <td><strong>{w.active}</strong></td>
                    <td>{w.median_days_to_fix_90d ?? '—'}</td>
                    <td>{w.pct_fixed_30d == null ? '—' : `${w.pct_fixed_30d}%`}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
      <section className="card stack">
        <h2>{t('city.worst_roads')}</h2>
        {data.worst_roads.length === 0 ? <Empty>{t('city.no_data')}</Empty> : (
          <div className="table-wrap">
            <table className="table">
              <thead><tr><th>{t('city.road')}</th><th>{t('city.active')}</th><th>{t('city.health')}</th></tr></thead>
              <tbody>
                {data.worst_roads.map((r) => (
                  <tr key={`${r.road}-${r.ward}`}>
                    <td>{r.road}<div className="tiny">{r.ward}</div></td>
                    <td>{r.active}</td>
                    <td>
                      <div className="row"><div className="health-bar grow"><span style={{ width: `${r.health}%`, background: healthColor(r.health) }} /></div>
                        <span className="tiny">{r.health}</span></div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
