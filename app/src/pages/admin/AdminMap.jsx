import { useCallback, useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import L from 'leaflet';
import 'leaflet.heat';
import { useMap } from 'react-leaflet';
import { useAdminStats, useHazards } from '../../api/hooks';
import MapView from '../../components/MapView';
import { boundsParam, DEFAULT_CENTER } from '../../lib/geo';

function Heat({ points }) {
  const map = useMap();
  useEffect(() => {
    const layer = L.heatLayer(points.map((p) => [p.lat, p.lng, Math.max(0.2, p.weight)]), { radius: 25, blur: 18 });
    layer.addTo(map);
    return () => { map.removeLayer(layer); };
  }, [map, points]);
  return null;
}

export default function AdminMap() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const [mode, setMode] = useState('pins');
  const [bbox, setBbox] = useState(null);
  const onMove = useCallback((map) => setBbox(boundsParam(map.getBounds())), []);
  const hazards = useHazards(bbox, 'all');
  const stats = useAdminStats();
  return (
    <div className="stack">
      <div className="spread">
        <h1>{t('admin.map')}</h1>
        <div className="chips">
          {['pins', 'heatmap'].map((m) => (
            <button key={m} className="chip" aria-pressed={mode === m} onClick={() => setMode(m)}>{t(`admin.${m}`)}</button>
          ))}
        </div>
      </div>
      <div className="admin-map">
        <MapView center={DEFAULT_CENTER} zoom={12} onMove={onMove}
          hazards={mode === 'pins' ? hazards.data?.hazards || [] : []} onSelect={(h) => navigate(`/h/${h.id}`)}>
          {mode === 'heatmap' && stats.data && <Heat points={stats.data.heat} />}
        </MapView>
      </div>
    </div>
  );
}
