const R = 6371000;
const rad = (d) => (d * Math.PI) / 180;

export function haversineM(lat1, lng1, lat2, lng2) {
  const dphi = rad(lat2 - lat1);
  const dl = rad(lng2 - lng1);
  const a = Math.sin(dphi / 2) ** 2 + Math.cos(rad(lat1)) * Math.cos(rad(lat2)) * Math.sin(dl / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(a));
}

/** "minLng,minLat,maxLng,maxLat" covering a circle of `radiusM` around the point. */
export function bboxParam(lat, lng, radiusM) {
  const dLat = (radiusM / R) * (180 / Math.PI);
  const dLng = dLat / Math.max(Math.cos(rad(lat)), 1e-6);
  return [lng - dLng, lat - dLat, lng + dLng, lat + dLat].map((v) => v.toFixed(6)).join(',');
}

export function boundsParam(bounds) {
  const sw = bounds.getSouthWest();
  const ne = bounds.getNorthEast();
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  return [clamp(sw.lng, -180, 180), clamp(sw.lat, -90, 90), clamp(ne.lng, -180, 180), clamp(ne.lat, -90, 90)]
    .map((v) => v.toFixed(6)).join(',');
}

/**
 * Resolves {lat, lng, accuracy}; rejects with an Error whose `code` is 'denied' | 'unavailable' | 'timeout' |
 * 'unsupported'.
 */
export function getPosition({ timeout = 15000, maximumAge = 30000 } = {}) {
  return new Promise((resolve, reject) => {
    if (!('geolocation' in navigator)) {
      reject(Object.assign(new Error('unsupported'), { code: 'unsupported' }));
      return;
    }
    navigator.geolocation.getCurrentPosition(
      (p) => resolve({ lat: p.coords.latitude, lng: p.coords.longitude, accuracy: p.coords.accuracy }),
      (e) => reject(Object.assign(new Error(e.message), {
        code: e.code === 1 ? 'denied' : e.code === 3 ? 'timeout' : 'unavailable' })),
      { enableHighAccuracy: true, timeout, maximumAge },
    );
  });
}

/** True only if the user already granted location, so we never prompt on page load. */
export async function locationAlreadyGranted() {
  try {
    const status = await navigator.permissions?.query({ name: 'geolocation' });
    return status?.state === 'granted';
  } catch {
    return false;
  }
}

export const googleMapsUrl = (lat, lng) => `https://maps.google.com/?q=${lat.toFixed(6)},${lng.toFixed(6)}`;

export const DEFAULT_CENTER = { lat: 18.5204, lng: 73.8567 }; // Pune
