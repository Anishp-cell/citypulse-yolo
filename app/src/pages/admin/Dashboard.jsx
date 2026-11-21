import { useState, useEffect } from 'react';
import { api } from '../../utils/api';
import { getSeverityMeta, formatDate, formatSeverity } from '../../utils/severity';
import { useTranslation } from 'react-i18next';
import { AlertTriangle, CheckCircle, Clock, Activity, TrendingUp, MapPin } from 'lucide-react';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';
import './AdminPages.css';

export default function Dashboard() {
  const { t } = useTranslation();
  const [stats, setStats] = useState(null);
  const [recent, setRecent] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 30000); // auto-refresh 30s
    return () => clearInterval(interval);
  }, []);

  async function loadData() {
    try {
      const [statsRes, incidentsRes] = await Promise.all([
        api.get('/api/admin/stats'),
        api.get('/api/incidents?limit=8'),
      ]);
      setStats(statsRes);
      setRecent(incidentsRes.incidents || []);
    } catch (err) {
      console.error('Dashboard load error:', err);
    } finally {
      setLoading(false);
    }
  }

  if (loading) {
    return (
      <div className="page-loading">
        <span className="spinner" style={{ width: 32, height: 32 }} />
        <p>{t('common.loading')}</p>
      </div>
    );
  }

  const statCards = [
    { label: t('admin.total_incidents'), value: stats?.total || 0, icon: AlertTriangle, color: 'var(--accent)' },
    { label: t('admin.active_alerts'), value: stats?.active || 0, icon: Activity, color: 'var(--red)' },
    { label: t('admin.resolved_today'), value: stats?.resolved_today || 0, icon: CheckCircle, color: 'var(--green)' },
    { label: t('admin.response_time'), value: stats?.avg_response || '—', icon: Clock, color: 'var(--yellow)' },
  ];

  const severityData = stats?.by_severity
    ? Object.entries(stats.by_severity).map(([key, val]) => ({
        name: formatSeverity(key),
        value: val,
        color: getSeverityMeta(key).color,
      }))
    : [];

  return (
    <div className="admin-page fade-in">
      <div className="page-header">
        <h1>{t('admin.dashboard')}</h1>
        <p className="page-desc">Real-time overview of road safety incidents across your city</p>
      </div>

      {/* Stats Cards */}
      <div className="stats-grid">
        {statCards.map((card, i) => (
          <div key={i} className="stat-card card" style={{ '--accent-c': card.color, animationDelay: `${i * 0.08}s` }}>
            <div className="stat-card-icon" style={{ background: `${card.color}18`, color: card.color }}>
              <card.icon size={22} />
            </div>
            <div className="stat-card-info">
              <span className="stat-card-value">{card.value}</span>
              <span className="stat-card-label">{card.label}</span>
            </div>
          </div>
        ))}
      </div>

      {/* Two-column: Severity Chart + Recent Activity */}
      <div className="dash-grid">
        {/* Severity Breakdown */}
        <div className="card dash-chart-card">
          <h3>{t('admin.severity_breakdown')}</h3>
          {severityData.length > 0 ? (
            <div className="pie-container">
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie
                    data={severityData}
                    dataKey="value"
                    nameKey="name"
                    cx="50%" cy="50%"
                    innerRadius={55} outerRadius={85}
                    paddingAngle={3}
                    stroke="none"
                  >
                    {severityData.map((entry, i) => (
                      <Cell key={i} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{
                      background: 'var(--bg-card)',
                      border: '1px solid var(--border)',
                      borderRadius: 'var(--radius-md)',
                      color: 'var(--text-primary)',
                      fontSize: '0.8125rem',
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>
              <div className="pie-legend">
                {severityData.map((d, i) => (
                  <div key={i} className="pie-legend-item">
                    <span className="pie-dot" style={{ background: d.color }} />
                    <span>{d.name}</span>
                    <span className="pie-count">{d.value}</span>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <p className="no-data">{t('common.no_data')}</p>
          )}
        </div>

        {/* Recent Activity */}
        <div className="card dash-recent-card">
          <h3>{t('admin.recent_activity')}</h3>
          <div className="recent-list">
            {recent.length > 0 ? recent.map(inc => {
              const meta = getSeverityMeta(inc.severity);
              return (
                <div key={inc.id} className="recent-item">
                  <div className="recent-dot" style={{ background: meta.color }} />
                  <div className="recent-info">
                    <span className="recent-severity">{meta.icon} {meta.label}</span>
                    <span className="recent-location">
                      <MapPin size={12} />
                      {inc.address_text || 'Location unknown'}
                    </span>
                  </div>
                  <div className="recent-meta">
                    <span className={`badge status-${inc.status || 'detected'}`}>
                      {(inc.status || 'detected').replace(/_/g, ' ')}
                    </span>
                    <span className="recent-time">{formatDate(inc.created_at)}</span>
                  </div>
                </div>
              );
            }) : (
              <p className="no-data">{t('common.no_data')}</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
