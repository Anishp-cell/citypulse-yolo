import { useState, useEffect } from 'react';
import { api } from '../../utils/api';
import { getSeverityMeta, formatDate } from '../../utils/severity';
import { useTranslation } from 'react-i18next';
import { MapPin } from 'lucide-react';
import './PublicPages.css';

export default function MyReports() {
  const { t } = useTranslation();
  const [incidents, setIncidents] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.get('/api/incidents?limit=50')
      .then(res => setIncidents(res.incidents || []))
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="page-loading"><span className="spinner" style={{width:24,height:24}}/><p>{t('common.loading')}</p></div>;

  return (
    <div className="public-page fade-in">
      <h2 className="pub-page-title">{t('my_reports.title')}</h2>
      {incidents.length === 0 ? (
        <div className="empty-state"><p>{t('my_reports.empty')}</p></div>
      ) : (
        <div className="reports-list">
          {incidents.map(inc => {
            const meta = getSeverityMeta(inc.severity);
            return (
              <div key={inc.id} className="report-card card">
                {inc.image_url && <img src={inc.image_url} alt="" className="report-thumb" />}
                <div className="report-info">
                  <div className="report-sev">
                    <span className={`badge ${meta.badge}`}>{meta.icon} {meta.label}</span>
                    <span className={`badge status-${inc.status || 'detected'}`}>{(inc.status || 'detected').replace(/_/g, ' ')}</span>
                  </div>
                  <p className="report-loc"><MapPin size={12} /> {inc.address_text || t('my_reports.location_not_set')}</p>
                  <p className="report-date">{formatDate(inc.created_at)}</p>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
