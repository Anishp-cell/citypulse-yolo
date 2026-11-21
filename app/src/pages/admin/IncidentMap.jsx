import { useState, useEffect, useRef } from 'react';
import { MapContainer, TileLayer, CircleMarker, Popup, useMap } from 'react-leaflet';
import { api } from '../../utils/api';
import { getSeverityMeta, formatDate, formatSeverity } from '../../utils/severity';
import { useTranslation } from 'react-i18next';
import { Layers, Filter } from 'lucide-react';
import 'leaflet/dist/leaflet.css';
import './AdminPages.css';

// Heatmap layer component using leaflet.heat
function HeatmapLayer({ points }) {
  const map = useMap();

  useEffect(() => {
    if (!points || points.length === 0) return;

    import('leaflet.heat').then(() => {
      const heat = window.L.heatLayer(
        points.map(p => [p.lat, p.lng, p.intensity || 1]),
        {
          radius: 25,
          blur: 20,
          maxZoom: 17,
          max: 1.0,
          gradient: {
            0.2: '#22c55e',
            0.4: '#f59e0b',
            0.6: '#f97316',
            0.8: '#ef4444',
            1.0: '#dc2626',
          },
        }
      ).addTo(map);

      return () => map.removeLayer(heat);
    });
  }, [map, points]);

  return null;
}

export default function IncidentMap() {
  const { t } = useTranslation();
  const [incidents, setIncidents] = useState([]);
  const [heatPoints, setHeatPoints] = useState([]);
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState('all');

  useEffect(() => {
    loadData();
  }, []);

  async function loadData() {
    try {
      const [incRes, heatRes] = await Promise.all([
        api.get('/api/incidents?limit=100'),
        api.get('/api/admin/heatmap'),
      ]);
      setIncidents(incRes.incidents || []);
      setHeatPoints(heatRes.points || []);
    } catch (err) {
      console.error('Map data error:', err);
    } finally {
      setLoading(false);
    }
  }

  const filtered = filter === 'all'
    ? incidents
    : incidents.filter(i => i.severity === filter);

  // Default center: India
  const center = [20.5937, 78.9629];
  const zoom = 5;

  return (
    <div className="admin-page fade-in">
      <div className="page-header">
        <h1>{t('admin.map')}</h1>
        <div className="page-actions">
          <button
            className={`btn btn-sm ${showHeatmap ? 'btn-primary' : 'btn-ghost'}`}
            onClick={() => setShowHeatmap(!showHeatmap)}
          >
            <Layers size={16} />
            {t('admin.heatmap')}
          </button>
          <select className="input filter-select" value={filter} onChange={e => setFilter(e.target.value)}>
            <option value="all">All Severities</option>
            <option value="minor_accident">Minor</option>
            <option value="moderate_accident">Moderate</option>
            <option value="severe_accident">Severe</option>
            <option value="totaled_vehicle">Critical</option>
            <option value="pothole">Pothole</option>
          </select>
        </div>
      </div>

      <div className="map-wrapper card">
        {loading ? (
          <div className="map-loading">
            <span className="spinner" style={{ width: 32, height: 32 }} />
            <p>{t('common.loading')}</p>
          </div>
        ) : (
          <MapContainer center={center} zoom={zoom} className="incident-map" scrollWheelZoom>
            <TileLayer
              url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/">CARTO</a>'
            />

            {showHeatmap && <HeatmapLayer points={heatPoints} />}

            {filtered.map(inc => {
              if (!inc.lat || !inc.lng) return null;
              const meta = getSeverityMeta(inc.severity);
              return (
                <CircleMarker
                  key={inc.id}
                  center={[inc.lat, inc.lng]}
                  radius={8}
                  fillColor={meta.color}
                  fillOpacity={0.8}
                  stroke
                  color={meta.color}
                  weight={2}
                  opacity={0.4}
                >
                  <Popup>
                    <div className="map-popup">
                      <div className="popup-severity">
                        <span>{meta.icon}</span>
                        <strong>{meta.label}</strong>
                      </div>
                      {inc.image_url && (
                        <img src={inc.image_url} alt="" className="popup-img" />
                      )}
                      <p className="popup-location">{inc.address_text || 'Unknown location'}</p>
                      <p className="popup-date">{formatDate(inc.created_at)}</p>
                      <span className={`badge status-${inc.status || 'detected'}`}>
                        {(inc.status || 'detected').replace(/_/g, ' ')}
                      </span>
                    </div>
                  </Popup>
                </CircleMarker>
              );
            })}
          </MapContainer>
        )}
      </div>

      {/* Stats bar */}
      <div className="map-stats">
        <div className="map-stat">
          <span className="map-stat-val">{incidents.length}</span>
          <span className="map-stat-label">Total Incidents</span>
        </div>
        <div className="map-stat">
          <span className="map-stat-val">{incidents.filter(i => i.lat && i.lng).length}</span>
          <span className="map-stat-label">Mapped</span>
        </div>
        <div className="map-stat">
          <span className="map-stat-val">{heatPoints.length}</span>
          <span className="map-stat-label">Heatmap Points</span>
        </div>
      </div>
    </div>
  );
}
