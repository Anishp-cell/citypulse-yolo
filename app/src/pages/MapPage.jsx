import { useCallback, useState } from 'react';
import { Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { LocateFixed } from 'lucide-react';
import { useHazards } from '../api/hooks';
import HazardCard from '../components/HazardCard';
import MapView from '../components/MapView';
import { toast } from '../components/Toaster';
import { Sheet } from '../components/ui';
import { boundsParam, DEFAULT_CENTER, getPosition } from '../lib/geo';

const FILTERS = ['active', 'fixed', 'all'];

export default function MapPage() {
  const { t } = useTranslation();
  const [filter, setFilter] = useState('active');
  const [bbox, setBbox] = useState(null);
  const [center, setCenter] = useState(DEFAULT_CENTER);
  const [selected, setSelected] = useState(null);
  const { data } = useHazards(bbox, filter);
  const onMove = useCallback((map) => setBbox(boundsParam(map.getBounds())), []);

  const locate = async () => {
    try {
      const p = await getPosition({ timeout: 10000 });
      setCenter({ lat: p.lat, lng: p.lng });
    } catch {
      toast(t('errors.geolocation_unsupported'));
    }
  };

  return (
    <div className="map-page">
      <MapView center={center} zoom={14} hazards={data?.hazards || []} onSelect={setSelected} onMove={onMove} />
      <div className="map-toolbar">
        <div className="chips" role="group" aria-label={t('map.title')}>
          {FILTERS.map((f) => (
            <button key={f} className="chip" aria-pressed={filter === f} onClick={() => setFilter(f)}>{t(`map.filter_${f}`)}</button>
          ))}
        </div>
        <button className="btn icon-btn" style={{ background: 'var(--surface)', boxShadow: 'var(--shadow)' }}
          onClick={locate} aria-label={t('map.locate')}><LocateFixed size={20} /></button>
      </div>
      {data?.truncated && (
        <div className="toasts" style={{ bottom: 16 }}><div className="toast">{t('map.too_many')}</div></div>
      )}
      <Sheet open={Boolean(selected)} onClose={() => setSelected(null)} title={selected ? t(`types.${selected.type}`) : ''}>
        {selected && (
          <div className="stack">
            <HazardCard hazard={selected} />
            <Link className="btn btn-primary btn-block" to={`/h/${selected.id}`}>{t('map.open_details')}</Link>
          </div>
        )}
      </Sheet>
    </div>
  );
}
