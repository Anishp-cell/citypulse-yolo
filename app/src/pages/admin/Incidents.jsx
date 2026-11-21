import { useState, useEffect } from 'react';
import { api } from '../../utils/api';
import { getSeverityMeta, formatDate, STATUS_OPTIONS } from '../../utils/severity';
import { useTranslation } from 'react-i18next';
import { MapPin, Eye, RefreshCw } from 'lucide-react';
import './AdminPages.css';

export default function Incidents() {
  const { t } = useTranslation();
  const [incidents, setIncidents] = useState([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(0);
  const [filters, setFilters] = useState({ severity: '', status: '' });
  const [selected, setSelected] = useState(null);
  const limit = 15;

  useEffect(() => { loadIncidents(); }, [page, filters]);

  async function loadIncidents() {
    setLoading(true);
    try {
      let url = `/api/incidents?skip=${page * limit}&limit=${limit}`;
      if (filters.severity) url += `&severity=${filters.severity}`;
      if (filters.status) url += `&status=${filters.status}`;
      const res = await api.get(url);
      setIncidents(res.incidents || []);
      setTotal(res.total || 0);
    } catch (err) { console.error(err); }
    finally { setLoading(false); }
  }

  async function viewDetail(id) {
    try {
      const detail = await api.get(`/api/incidents/${id}`);
      setSelected(detail);
    } catch (err) { console.error(err); }
  }

  async function updateStatus(id, newStatus) {
    try {
      await api.patch(`/api/incidents/${id}/status`, { status: newStatus });
      loadIncidents();
      if (selected?.id === id) setSelected(prev => ({ ...prev, status: newStatus }));
    } catch (err) { console.error(err); }
  }

  const totalPages = Math.ceil(total / limit);

  return (
    <div className="admin-page fade-in">
      <div className="page-header">
        <h1>{t('admin.incidents')}</h1>
        <button className="btn btn-ghost btn-sm" onClick={loadIncidents}><RefreshCw size={16} /> Refresh</button>
      </div>

      <div className="table-filters">
        <select className="input filter-select" value={filters.severity}
          onChange={e => { setFilters(f => ({ ...f, severity: e.target.value })); setPage(0); }}>
          <option value="">All Severities</option>
          <option value="minor_accident">Minor</option>
          <option value="moderate_accident">Moderate</option>
          <option value="severe_accident">Severe</option>
          <option value="totaled_vehicle">Critical</option>
          <option value="pothole">Pothole</option>
        </select>
        <select className="input filter-select" value={filters.status}
          onChange={e => { setFilters(f => ({ ...f, status: e.target.value })); setPage(0); }}>
          <option value="">All Statuses</option>
          {STATUS_OPTIONS.map(s => <option key={s.value} value={s.value}>{s.label}</option>)}
        </select>
        <span className="filter-count">{total} incidents</span>
      </div>

      <div className="incidents-layout">
        <div className="card incidents-table-card">
          {loading ? (
            <div className="table-loading"><span className="spinner" /> {t('common.loading')}</div>
          ) : incidents.length === 0 ? (
            <p className="no-data">{t('common.no_data')}</p>
          ) : (
            <>
              <table className="incidents-table">
                <thead><tr><th>ID</th><th>Severity</th><th>Status</th><th>Location</th><th>Date</th><th></th></tr></thead>
                <tbody>
                  {incidents.map(inc => {
                    const meta = getSeverityMeta(inc.severity);
                    return (
                      <tr key={inc.id} className={selected?.id === inc.id ? 'row-selected' : ''}>
                        <td>#{inc.id}</td>
                        <td><span className={`badge ${meta.badge}`}>{meta.icon} {meta.label}</span></td>
                        <td>
                          <select className="status-select" value={inc.status || 'detected'}
                            onChange={e => updateStatus(inc.id, e.target.value)}>
                            {STATUS_OPTIONS.map(s => <option key={s.value} value={s.value}>{s.label}</option>)}
                          </select>
                        </td>
                        <td className="td-location"><MapPin size={13} />{inc.address_text || '—'}</td>
                        <td className="td-date">{formatDate(inc.created_at)}</td>
                        <td><button className="btn btn-ghost btn-sm" onClick={() => viewDetail(inc.id)}><Eye size={14} /></button></td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              {totalPages > 1 && (
                <div className="pagination">
                  <button className="btn btn-ghost btn-sm" disabled={page === 0} onClick={() => setPage(p => p - 1)}>← Prev</button>
                  <span>Page {page + 1} / {totalPages}</span>
                  <button className="btn btn-ghost btn-sm" disabled={page >= totalPages - 1} onClick={() => setPage(p => p + 1)}>Next →</button>
                </div>
              )}
            </>
          )}
        </div>

        {selected && (
          <div className="card incident-detail-panel fade-in">
            <h3>Incident #{selected.id}</h3>
            {selected.image_url && <img src={selected.image_url} alt="" className="detail-image" />}
            <div className="detail-fields">
              <div className="detail-field"><span className="detail-label">Severity</span><span className={`badge ${getSeverityMeta(selected.severity).badge}`}>{getSeverityMeta(selected.severity).label}</span></div>
              <div className="detail-field"><span className="detail-label">Status</span><span className={`badge status-${selected.status||'detected'}`}>{(selected.status||'detected').replace(/_/g,' ')}</span></div>
              <div className="detail-field"><span className="detail-label">Location</span><span>{selected.address_text||'—'}</span></div>
              <div className="detail-field"><span className="detail-label">Reported</span><span>{formatDate(selected.created_at)}</span></div>
            </div>
            {selected.guidance?.llm_enhanced && <div className="detail-guidance"><h4>🧠 AI Guidance</h4><p>{selected.guidance.llm_enhanced}</p></div>}
            <button className="btn btn-ghost btn-sm" onClick={() => setSelected(null)} style={{marginTop:'1rem'}}>Close</button>
          </div>
        )}
      </div>
    </div>
  );
}
