import { useState, useEffect } from 'react';
import { api } from '../../utils/api';
import { formatSeverity, getSeverityMeta } from '../../utils/severity';
import { useTranslation } from 'react-i18next';
import { BarChart, Bar, Cell, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';
import './AdminPages.css';

export default function Analytics() {
  const { t } = useTranslation();
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.get('/api/admin/stats').then(setStats).catch(console.error).finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="page-loading"><span className="spinner" style={{width:32,height:32}}/><p>{t('common.loading')}</p></div>;

  const trendData = stats?.daily_trend || [];
  const severityData = stats?.by_severity
    ? Object.entries(stats.by_severity).map(([k,v]) => ({ name: formatSeverity(k), value: v, fill: getSeverityMeta(k).color }))
    : [];

  const chartTooltipStyle = { background:'var(--bg-card)', border:'1px solid var(--border)', borderRadius:'var(--radius-md)', color:'var(--text-primary)', fontSize:'0.8125rem' };

  return (
    <div className="admin-page fade-in">
      <div className="page-header"><h1>{t('admin.analytics')}</h1></div>

      <div className="analytics-grid">
        <div className="card analytics-chart-card">
          <h3>{t('admin.incident_trends')}</h3>
          <p className="chart-subtitle">Daily incident count over the last 30 days</p>
          {trendData.length > 0 ? (
            <ResponsiveContainer width="100%" height={280}>
              <AreaChart data={trendData}>
                <defs>
                  <linearGradient id="colorCount" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#6366f1" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="date" tick={{fill:'var(--text-muted)',fontSize:11}} axisLine={{stroke:'var(--border)'}} />
                <YAxis tick={{fill:'var(--text-muted)',fontSize:11}} axisLine={{stroke:'var(--border)'}} />
                <Tooltip contentStyle={chartTooltipStyle} />
                <Area type="monotone" dataKey="count" stroke="#6366f1" strokeWidth={2} fillOpacity={1} fill="url(#colorCount)" />
              </AreaChart>
            </ResponsiveContainer>
          ) : <p className="no-data">{t('common.no_data')}</p>}
        </div>

        <div className="card analytics-chart-card">
          <h3>{t('admin.severity_breakdown')}</h3>
          <p className="chart-subtitle">Incidents by severity class</p>
          {severityData.length > 0 ? (
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={severityData} barSize={32}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="name" tick={{fill:'var(--text-muted)',fontSize:11}} axisLine={{stroke:'var(--border)'}} />
                <YAxis tick={{fill:'var(--text-muted)',fontSize:11}} axisLine={{stroke:'var(--border)'}} />
                <Tooltip contentStyle={chartTooltipStyle} />
                <Bar dataKey="value" radius={[4,4,0,0]}>
                  {severityData.map((entry,i) => (
                    <Cell key={i} fill={entry.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : <p className="no-data">{t('common.no_data')}</p>}
        </div>

        <div className="card analytics-chart-card full-width">
          <h3>{t('admin.hotspots')}</h3>
          <p className="chart-subtitle">Top locations with the most incidents</p>
          {stats?.hotspots?.length > 0 ? (
            <div className="hotspots-list">
              {stats.hotspots.map((h,i) => (
                <div key={i} className="hotspot-item">
                  <span className="hotspot-rank">#{i+1}</span>
                  <span className="hotspot-location">{h.location}</span>
                  <span className="hotspot-count">{h.count} incidents</span>
                  <div className="hotspot-bar"><div style={{width:`${(h.count / stats.hotspots[0].count) * 100}%`,background:'var(--gradient-brand)'}} /></div>
                </div>
              ))}
            </div>
          ) : <p className="no-data">{t('common.no_data')}</p>}
        </div>
      </div>
    </div>
  );
}
