import { useState } from 'react';
import { Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { useFollowing, useImpact, useMyReports } from '../api/hooks';
import { useAuth } from '../hooks/useAuth';
import HazardCard from '../components/HazardCard';
import { Empty, ErrorState, Loading } from '../components/ui';

export default function MyReports() {
  const { t } = useTranslation();
  const { user, loading } = useAuth();
  const [tab, setTab] = useState('reported');
  const reported = useMyReports(Boolean(user));
  const following = useFollowing(Boolean(user) && tab === 'following');
  const impact = useImpact(Boolean(user));
  if (loading) return <Loading />;
  if (!user) {
    return (
      <div className="card stack center">
        <h1>{t('mine.title')}</h1>
        <Link className="btn btn-primary" to="/login?next=/me">{t('nav.login')}</Link>
      </div>
    );
  }
  const q = tab === 'reported' ? reported : following;
  const list = q.data?.hazards || [];
  return (
    <div className="stack">
      <h1>{t('mine.title')}</h1>
      {impact.data && (
        <section aria-label={t('mine.impact_title')} className="stat-grid">
          <div className="stat"><strong>{impact.data.hazards_reported}</strong><span className="tiny">{t('mine.impact_reports')}</span></div>
          <div className="stat"><strong>{impact.data.hazards_fixed}</strong><span className="tiny">{t('mine.impact_fixed')}</span></div>
          <div className="stat"><strong>{impact.data.verifications}</strong><span className="tiny">{t('mine.impact_verifications')}</span></div>
        </section>
      )}
      <div className="chips" role="tablist">
        {['reported', 'following'].map((k) => (
          <button key={k} role="tab" className="chip" aria-selected={tab === k} aria-pressed={tab === k} onClick={() => setTab(k)}>
            {t(`mine.${k}`)}
          </button>
        ))}
      </div>
      {q.isPending && <Loading />}
      {q.isError && <ErrorState error={q.error} onRetry={q.refetch} />}
      {q.isSuccess && list.length === 0 && <Empty>{t(tab === 'reported' ? 'mine.empty_reported' : 'mine.empty_following')}</Empty>}
      <div className="stack-sm">{list.map((h) => <HazardCard key={h.id} hazard={h} />)}</div>
      {q.isSuccess && list.length === 0 && tab === 'reported' && (
        <Link className="btn btn-primary" to="/report">{t('home.report_cta')}</Link>
      )}
    </div>
  );
}
