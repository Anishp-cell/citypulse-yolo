import { useEffect, useMemo } from 'react';
import L from 'leaflet';
import { MapContainer, Marker, TileLayer, useMap, useMapEvents } from 'react-leaflet';
import MarkerClusterGroup from 'react-leaflet-cluster';
import 'leaflet/dist/leaflet.css';
import 'react-leaflet-cluster/dist/assets/MarkerCluster.css';
import 'react-leaflet-cluster/dist/assets/MarkerCluster.Default.css';
import { statusMeta } from '../lib/status';

// OpenStreetMap standard tiles: fine for light use with attribution. Switch to a keyed provider
// (MapTiler, Stadia) before heavy traffic. CARTO's free basemaps now require an API key.
const TILE_URL = 'https://tile.openstreetmap.org/{z}/{x}/{y}.png';
const ATTRIBUTION = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors';

const iconCache = new Map();
function pinIcon(status) {
  if (!iconCache.has(status)) {
    iconCache.set(status, L.divIcon({
      className: '',
      html: `<div class="marker-pin" style="background:${statusMeta(status).color}"></div>`,
      iconSize: [22, 22],
      iconAnchor: [11, 11],
    }));
  }
  return iconCache.get(status);
}

function Events({ onMove }) {
  const map = useMapEvents({ moveend: () => onMove?.(map) });
  useEffect(() => { onMove?.(map); }, []); // eslint-disable-line react-hooks/exhaustive-deps
  return null;
}

function Recenter({ center, zoom }) {
  const map = useMap();
  useEffect(() => {
    if (center) map.setView([center.lat, center.lng], zoom ?? map.getZoom());
  }, [center?.lat, center?.lng]); // eslint-disable-line react-hooks/exhaustive-deps
  return null;
}

const prefersDark = () => window.matchMedia?.('(prefers-color-scheme: dark)').matches;

export default function MapView({ center, zoom = 15, hazards = [], onSelect, onMove, children, cluster = true }) {
  const dark = useMemo(() => prefersDark(), []);
  const markers = hazards.map((h) => (
    <Marker key={h.id} position={[h.lat, h.lng]} icon={pinIcon(h.status)}
      eventHandlers={{ click: () => onSelect?.(h) }} title={h.road || h.type} />
  ));
  return (
    <MapContainer center={[center.lat, center.lng]} zoom={zoom} zoomControl={false} attributionControl
      style={{ height: '100%', width: '100%' }}>
      <TileLayer url={TILE_URL} attribution={ATTRIBUTION} maxZoom={19} className={dark ? 'tiles-dark' : undefined} />
      <Events onMove={onMove} />
      <Recenter center={center} zoom={zoom} />
      {cluster ? <MarkerClusterGroup chunkedLoading maxClusterRadius={45}>{markers}</MarkerClusterGroup> : markers}
      {children}
    </MapContainer>
  );
}
