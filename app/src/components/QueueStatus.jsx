import { useCallback, useEffect, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { useTranslation } from 'react-i18next';
import { CloudOff } from 'lucide-react';
import { analyzePhoto, submitReport } from '../api/hooks';
import { useAuth } from '../hooks/useAuth';
import { errorMessage } from '../lib/format';
import { flush, list } from '../lib/offlineQueue';
import { toast } from './Toaster';

export const RETRY_EVERY_MS = 60_000;

const toFile = (item) => new File([item.bytes], 'photo.jpg', { type: item.mime || 'image/jpeg' });

/** Shows how many offline reports are waiting and sends them when the connection comes back. */
export default function QueueStatus() {
  const { t } = useTranslation();
  const { user } = useAuth();
  const qc = useQueryClient();
  const [count, setCount] = useState(0);

  const refresh = useCallback(async () => setCount((await list()).length), []);

  const sync = useCallback(async () => {
    if (!user || !navigator.onLine) return;
    const res = await flush({
      analyze: (item) => analyzePhoto({ file: toFile(item), lat: item.lat, lng: item.lng,
        accuracy: item.accuracy, source: item.source }),
      submit: (item, uploadId) => submitReport({ uploadId, type: item.type, severity: item.severity,
        note: item.note, idempotencyKey: item.idempotencyKey }),
      onDrop: (_item, code) => toast(t('report.queued_failed', { reason: errorMessage(t, { code }) }), 8000),
    });
    if (res.sent) qc.invalidateQueries({ queryKey: ['me'] });
    refresh();
  }, [user, qc, t, refresh]);

  useEffect(() => {
    const first = setTimeout(() => { refresh(); sync(); }, 0);
    const onChange = () => refresh();
    window.addEventListener('online', sync);
    window.addEventListener('cp:queue-changed', onChange);
    return () => {
      clearTimeout(first);
      window.removeEventListener('online', sync);
      window.removeEventListener('cp:queue-changed', onChange);
    };
  }, [sync, refresh]);

  useEffect(() => {
    if (!count) return undefined;
    // The server may have been cold or the network flaky when we last tried; keep trying while
    // anything is queued, and immediately when the app comes back to the foreground.
    const timer = setInterval(sync, RETRY_EVERY_MS);
    const onVisible = () => document.visibilityState === 'visible' && sync();
    document.addEventListener('visibilitychange', onVisible);
    return () => {
      clearInterval(timer);
      document.removeEventListener('visibilitychange', onVisible);
    };
  }, [count, sync]);

  if (!count) return null;
  return (
    <div className="banner tone-warn row" style={{ justifyContent: 'center' }} role="status">
      <CloudOff size={16} aria-hidden="true" />{t('report.queued_count', { count })}
    </div>
  );
}
