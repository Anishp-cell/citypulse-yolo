// Severity mapping utility — shared across admin & public UIs

export const SEVERITY_MAP = {
  no_accident:       { icon: '✅', label: 'No Incident',   color: '#22c55e', badge: 'badge-green',   level: 0 },
  minor_accident:    { icon: '⚠️', label: 'Minor',         color: '#f59e0b', badge: 'badge-yellow',  level: 1 },
  moderate_accident: { icon: '🔶', label: 'Moderate',      color: '#f97316', badge: 'badge-orange',  level: 2 },
  severe_accident:   { icon: '🔴', label: 'Severe',        color: '#ef4444', badge: 'badge-red',     level: 3 },
  totaled_vehicle:   { icon: '🚨', label: 'Critical',      color: '#dc2626', badge: 'badge-crimson', level: 4 },
  pothole:           { icon: '🕳️', label: 'Pothole',       color: '#06b6d4', badge: 'badge-cyan',    level: 0 },
};

export const STATUS_OPTIONS = [
  { value: 'detected',     label: 'Detected',     color: '#f59e0b' },
  { value: 'notified',     label: 'Notified',     color: '#3b82f6' },
  { value: 'acknowledged', label: 'Acknowledged', color: '#6366f1' },
  { value: 'resolved',     label: 'Resolved',     color: '#22c55e' },
  { value: 'closed',       label: 'Closed',       color: '#64748b' },
];

export function getSeverityMeta(severity) {
  return SEVERITY_MAP[severity] || SEVERITY_MAP.no_accident;
}

export function formatSeverity(severity) {
  return (severity || '').replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

export function formatDate(iso) {
  if (!iso) return '—';
  return new Date(iso).toLocaleString('en-IN', {
    day: '2-digit', month: 'short', year: 'numeric',
    hour: '2-digit', minute: '2-digit',
  });
}
