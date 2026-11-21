import { CircleAlert, Construction, Droplets, MapPinned, TriangleAlert } from 'lucide-react';

// Tone = colour family used by badges and map pins. Keep in sync with backend/domain/status.py
// (enforced by status.contract.test.js).
export const STATUS = {
  pending_review: { tone: 'warn', color: '#b45309' },
  open: { tone: 'danger', color: '#dc2626' },
  filed: { tone: 'info', color: '#0369a1' },
  acknowledged: { tone: 'violet', color: '#6d28d9' },
  in_progress: { tone: 'violet', color: '#7c3aed' },
  fixed_unverified: { tone: 'brand', color: '#1d4ed8' },
  fixed: { tone: 'ok', color: '#15803d' },
  rejected: { tone: 'neutral', color: '#64748b' },
};

export const HAZARD_TYPES = [
  { value: 'pothole', icon: Construction },
  { value: 'open_manhole', icon: CircleAlert },
  { value: 'waterlogging', icon: Droplets },
  { value: 'debris', icon: TriangleAlert },
  { value: 'other', icon: MapPinned },
];

export const SEVERITIES = ['low', 'medium', 'high'];

export const typeIcon = (type) => (HAZARD_TYPES.find((t) => t.value === type) || HAZARD_TYPES[4]).icon;
export const statusMeta = (status) => STATUS[status] || STATUS.open;
