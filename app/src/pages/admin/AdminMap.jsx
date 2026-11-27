import { useCallback, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { useAdminStats, useHazards } from '../../api/hooks';
import MapView from '../../components/MapView';
import { boundsParam, DEFAULT_CENTER } from '../../lib/geo';
import { statusMeta } from '../../lib/status';
import { Heat, PageHeader } from './parts';
import { PIPELINE } from './consoleUtils';

export default function AdminMap() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const [mode, setMode] = useState('pins');
  const [bbox, setBbox] = useState(null);
  const onMove = useCallback((map) => setBbox(boundsParam(map.getBounds())), []);
  const hazards = useHazards(bbox, 'all');
  const stats = useAdminStats();
  return (
    <>
      <PageHeader title={t('admin.map')} description={t('admin.map_desc')}>
        <div className="segmented" role="group" aria-label={t('admin.map')}>
          {['pins', 'heatmap'].map((m) => (
            <button key={m} type="button" aria-pressed={mode === m} onClick={() => setMode(m)}>{t(`admin.${m}`)}</button>
          ))}
        </div>
      </PageHeader>
      <div className="panel admin-map">
        <MapView center={DEFAULT_CENTER} zoom={12} onMove={onMove}
          hazards={mode === 'pins' ? hazards.data?.hazards || [] : []} onSelect={(h) => navigate(`/h/${h.id}`)}>
          {mode === 'heatmap' && stats.data && <Heat points={stats.data.heat} />}
        </MapView>
        {mode === 'pins' && (
          <div className="map-legend" aria-label={t('admin.legend')}>
            {PIPELINE.map((s) => <span key={s}><i style={{ background: statusMeta(s).color }} />{t(`status.${s}`)}</span>)}
          </div>
        )}
      </div>
    </>
  );
}
