import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';

// eslint-disable-next-line react-refresh/only-export-components
export function toast(message, ms = 4000) {
  window.dispatchEvent(new CustomEvent('cp:toast', { detail: { message, ms } }));
}

export function Toaster() {
  const [items, setItems] = useState([]);
  useEffect(() => {
    const onToast = (e) => {
      const id = Math.random();
      setItems((xs) => [...xs, { id, message: e.detail.message }]);
      setTimeout(() => setItems((xs) => xs.filter((x) => x.id !== id)), e.detail.ms);
    };
    window.addEventListener('cp:toast', onToast);
    return () => window.removeEventListener('cp:toast', onToast);
  }, []);
  return (
    <div className="toasts" aria-live="polite">
      {items.map((x) => <div key={x.id} className="toast">{x.message}</div>)}
    </div>
  );
}

/** Offline + "server waking up" banners. */
export function ConnectionBanner() {
  const { t } = useTranslation();
  const [online, setOnline] = useState(navigator.onLine);
  const [waking, setWaking] = useState(false);
  useEffect(() => {
    const on = () => setOnline(true);
    const off = () => setOnline(false);
    const wake = () => setWaking(true);
    const awake = () => setWaking(false);
    window.addEventListener('online', on);
    window.addEventListener('offline', off);
    window.addEventListener('cp:waking', wake);
    window.addEventListener('cp:awake', awake);
    return () => {
      window.removeEventListener('online', on);
      window.removeEventListener('offline', off);
      window.removeEventListener('cp:waking', wake);
      window.removeEventListener('cp:awake', awake);
    };
  }, []);
  if (!online) return <div className="banner tone-warn" role="status">{t('common.offline')}</div>;
  if (waking) return <div className="banner tone-info" role="status">{t('common.waking')}</div>;
  return null;
}
