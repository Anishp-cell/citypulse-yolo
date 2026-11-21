import { useEffect, useMemo } from 'react';
import L from 'leaflet';
import { MapContainer, Marker, TileLayer, useMap, useMapEvents } from 'react-leaflet';
import MarkerClusterGroup from 'react-leaflet-cluster';
import 'leaflet/dist/leaflet.css';
import 'react-leaflet-cluster/dist/assets/MarkerCluster.css';
import 'react-leaflet-cluster/dist/assets/MarkerCluster.Default.css';
import { statusMeta } from '../lib/status';

// CARTO basemaps: free with attribution (tile.openstreetmap.org forbids heavy app usage).
const TILES = {
  light: 'https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png',
  dark: 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
};
const ATTRIBUTION = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>';

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
  const tiles = useMemo(() => (prefersDark() ? TILES.dark : TILES.light), []);
  const markers = hazards.map((h) => (
    <Marker key={h.id} position={[h.lat, h.lng]} icon={pinIcon(h.status)}
      eventHandlers={{ click: () => onSelect?.(h) }} title={h.road || h.type} />
  ));
  return (
    <MapContainer center={[center.lat, center.lng]} zoom={zoom} zoomControl={false} attributionControl
      style={{ height: '100%', width: '100%' }}>
      <TileLayer url={tiles} attribution={ATTRIBUTION} subdomains="abcd" maxZoom={20} detectRetina />
      <Events onMove={onMove} />
      <Recenter center={center} zoom={zoom} />
      {cluster ? <MarkerClusterGroup chunkedLoading maxClusterRadius={45}>{markers}</MarkerClusterGroup> : markers}
      {children}
    </MapContainer>
  );
}
