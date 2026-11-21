import { createElement, useEffect, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { CircleAlert, Info, X } from 'lucide-react';
import { statusMeta, typeIcon } from '../lib/status';
import { errorMessage } from '../lib/format';

export function Spinner({ label }) {
  return <span className="spinner" role="status" aria-label={label || 'loading'} />;
}

export function Loading() {
  const { t } = useTranslation();
  return <div className="loading-page"><Spinner label={t('common.loading')} /></div>;
}

export function ErrorState({ error, onRetry }) {
  const { t } = useTranslation();
  return (
    <div className="card empty stack" role="alert">
      <CircleAlert size={28} style={{ margin: '0 auto', color: 'var(--danger)' }} />
      <p>{errorMessage(t, error)}</p>
      {onRetry && <button className="btn btn-outline" onClick={onRetry}>{t('common.retry')}</button>}
    </div>
  );
}

export function Empty({ children, icon: Icon = Info }) {
  return (
    <div className="empty stack">
      <Icon size={28} style={{ margin: '0 auto', color: 'var(--text-3)' }} />
      <p>{children}</p>
    </div>
  );
}

export function Notice({ tone = 'info', icon: Icon = Info, children }) {
  return <div className={`notice tone-${tone}`} role="note"><Icon size={18} /><div>{children}</div></div>;
}

export function StatusBadge({ status }) {
  const { t } = useTranslation();
  return <span className={`badge tone-${statusMeta(status).tone}`}>{t(`status.${status}`)}</span>;
}

export function TypeLabel({ type, size = 16 }) {
  const { t } = useTranslation();
  return (
    <span className="row" style={{ gap: 6 }}>
      {createElement(typeIcon(type), { size, 'aria-hidden': true })}{t(`types.${type}`)}
    </span>
  );
}

export function Sheet({ open, onClose, title, children }) {
  const { t } = useTranslation();
  const ref = useRef(null);
  useEffect(() => {
    if (!open) return undefined;
    const onKey = (e) => e.key === 'Escape' && onClose();
    document.addEventListener('keydown', onKey);
    ref.current?.focus();
    return () => document.removeEventListener('keydown', onKey);
  }, [open, onClose]);
  if (!open) return null;
  return (
    <div className="sheet-backdrop" onClick={onClose}>
      <div className="sheet" role="dialog" aria-modal="true" aria-label={title} tabIndex={-1} ref={ref}
        onClick={(e) => e.stopPropagation()}>
        <div className="spread">
          <h2>{title}</h2>
          <button className="btn btn-ghost icon-btn" onClick={onClose} aria-label={t('common.close')}><X size={20} /></button>
        </div>
        {children}
      </div>
    </div>
  );
}
