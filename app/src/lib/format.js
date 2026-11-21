const LOCALES = { en: 'en-IN', hi: 'hi-IN', mr: 'mr-IN' };

export function formatDate(iso, lang = 'en', opts = { day: 'numeric', month: 'short', year: 'numeric' }) {
  if (!iso) return '';
  try {
    return new Intl.DateTimeFormat(LOCALES[lang] || 'en-IN', opts).format(new Date(iso));
  } catch {
    return new Date(iso).toDateString();
  }
}

export const formatDateTime = (iso, lang) =>
  formatDate(iso, lang, { day: 'numeric', month: 'short', hour: 'numeric', minute: '2-digit' });

export function errorMessage(t, err) {
  const code = err?.code;
  const key = `errors.${code}`;
  const msg = code ? t(key) : null;
  return msg && msg !== key ? msg : t('errors.server_error');
}

export const placeName = (h) => h?.road || h?.ward || h?.city || (h ? `${h.lat.toFixed(4)}, ${h.lng.toFixed(4)}` : '');
