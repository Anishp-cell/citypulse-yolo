// Web Push opt-in. iOS only supports it for apps added to the home screen (iOS 16.4+).
import { api } from '../api/client';

export const pushSupported = () =>
  'serviceWorker' in navigator && 'PushManager' in window && 'Notification' in window;

function urlBase64ToUint8Array(base64) {
  const padding = '='.repeat((4 - (base64.length % 4)) % 4);
  const raw = atob((base64 + padding).replace(/-/g, '+').replace(/_/g, '/'));
  return Uint8Array.from(raw, (c) => c.charCodeAt(0));
}

/** 'unsupported' | 'unavailable' | 'denied' | 'on' | 'off' */
export async function pushState() {
  if (!pushSupported()) return 'unsupported';
  if (Notification.permission === 'denied') return 'denied';
  const { enabled } = await api.get('/api/push/vapid-public-key').catch(() => ({ enabled: false }));
  if (!enabled) return 'unavailable';
  const reg = await navigator.serviceWorker.getRegistration();
  const sub = await reg?.pushManager.getSubscription();
  return sub ? 'on' : 'off';
}

export async function enablePush() {
  if (!pushSupported()) return 'unsupported';
  const { enabled, key } = await api.get('/api/push/vapid-public-key');
  if (!enabled) return 'unavailable';
  const permission = await Notification.requestPermission();
  if (permission !== 'granted') return 'denied';
  const reg = await navigator.serviceWorker.ready;
  const sub = (await reg.pushManager.getSubscription())
    || await reg.pushManager.subscribe({ userVisibleOnly: true, applicationServerKey: urlBase64ToUint8Array(key) });
  const json = sub.toJSON();
  await api.post('/api/push/subscribe', { endpoint: json.endpoint, keys: json.keys });
  return 'on';
}

export async function disablePush() {
  const reg = await navigator.serviceWorker?.getRegistration();
  const sub = await reg?.pushManager.getSubscription();
  if (sub) {
    await api.del('/api/push/subscribe', { endpoint: sub.endpoint }).catch(() => {});
    await sub.unsubscribe();
  }
  return 'off';
}
