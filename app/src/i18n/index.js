import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import en from './en.json';
import hi from './hi.json';
import mr from './mr.json';

export const LANGUAGES = [
  { code: 'en', label: 'English' },
  { code: 'hi', label: 'हिन्दी' },
  { code: 'mr', label: 'मराठी' },
];

function initialLanguage() {
  try {
    const saved = localStorage.getItem('cp_lang');
    if (saved && LANGUAGES.some((l) => l.code === saved)) return saved;
  } catch { /* storage unavailable */ }
  const browser = (navigator.language || 'en').slice(0, 2);
  return LANGUAGES.some((l) => l.code === browser) ? browser : 'en';
}

i18n.use(initReactI18next).init({
  resources: { en: { translation: en }, hi: { translation: hi }, mr: { translation: mr } },
  lng: initialLanguage(),
  fallbackLng: 'en',
  interpolation: { escapeValue: false },
  returnNull: false,
});

export function setLanguage(code) {
  i18n.changeLanguage(code);
  document.documentElement.lang = code;
  try { localStorage.setItem('cp_lang', code); } catch { /* ignore */ }
}

document.documentElement.lang = i18n.language;
export default i18n;
