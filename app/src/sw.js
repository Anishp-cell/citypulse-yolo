// Service worker: offline app shell, cached map tiles / photos / recent API reads, and Web Push.
import { cleanupOutdatedCaches, createHandlerBoundToURL, precacheAndRoute } from 'workbox-precaching';
import { NavigationRoute, registerRoute } from 'workbox-routing';
import { CacheFirst, NetworkFirst } from 'workbox-strategies';
import { ExpirationPlugin } from 'workbox-expiration';

self.skipWaiting();
self.addEventListener('activate', (event) => event.waitUntil(self.clients.claim()));

precacheAndRoute(self.__WB_MANIFEST);
cleanupOutdatedCaches();

const NOT_APP = [/^\/api\//, /^\/media\//, /^\/health/, /^\/docs/];
registerRoute(new NavigationRoute(createHandlerBoundToURL('/index.html'), { denylist: NOT_APP }));

const isCachedApi = (url) => /^\/api\/(hazards|me|scorecard)/.test(url.pathname) && !url.pathname.endsWith('.pdf');
registerRoute(
  ({ url, request }) => request.method === 'GET' && url.origin === self.location.origin && isCachedApi(url),
  new NetworkFirst({
    cacheName: 'api',
    networkTimeoutSeconds: 6,
    plugins: [new ExpirationPlugin({ maxEntries: 80, maxAgeSeconds: 24 * 3600 })],
  }),
);

registerRoute(
  ({ url }) => url.hostname.endsWith('basemaps.cartocdn.com'),
  new CacheFirst({ cacheName: 'tiles', plugins: [new ExpirationPlugin({ maxEntries: 600, maxAgeSeconds: 14 * 86400 })] }),
);

registerRoute(
  ({ url }) => url.pathname.startsWith('/media/public/') || url.pathname.startsWith('/public/'),
  new CacheFirst({ cacheName: 'photos', plugins: [new ExpirationPlugin({ maxEntries: 200, maxAgeSeconds: 30 * 86400 })] }),
);

self.addEventListener('push', (event) => {
  let data = {};
  try {
    data = event.data ? event.data.json() : {};
  } catch {
    data = { title: 'CityPulse', body: event.data ? event.data.text() : '' };
  }
  event.waitUntil(self.registration.showNotification(data.title || 'CityPulse', {
    body: data.body || '',
    icon: '/icons/icon-192.png',
    badge: '/icons/badge-72.png',
    tag: data.tag,
    renotify: Boolean(data.tag),
    data: { url: data.url || '/' },
  }));
});

self.addEventListener('notificationclick', (event) => {
  event.notification.close();
  const url = new URL((event.notification.data && event.notification.data.url) || '/', self.location.origin).href;
  event.waitUntil((async () => {
    const all = await clients.matchAll({ type: 'window', includeUncontrolled: true });
    const existing = all.find((c) => c.url.startsWith(self.location.origin));
    if (existing) {
      await existing.focus();
      return existing.navigate(url);
    }
    return clients.openWindow(url);
  })());
});
