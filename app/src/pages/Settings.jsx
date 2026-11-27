import { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { Bell, BellOff, Download, LogOut, Trash2 } from 'lucide-react';
import { useAuth } from '../hooks/useAuth';
import { LANGUAGES, setLanguage } from '../i18n';
import { toast } from '../components/Toaster';
import { Notice, Spinner } from '../components/ui';
import { disablePush, enablePush, pushState } from '../lib/push';
import { errorMessage } from '../lib/format';

function useInstallPrompt() {
  const [evt, setEvt] = useState(null);
  useEffect(() => {
    const onPrompt = (e) => { e.preventDefault(); setEvt(e); };
    window.addEventListener('beforeinstallprompt', onPrompt);
    return () => window.removeEventListener('beforeinstallprompt', onPrompt);
  }, []);
  return evt;
}

const isStandalone = () => window.matchMedia?.('(display-mode: standalone)').matches || navigator.standalone;
const isIos = () => /iphone|ipad|ipod/i.test(navigator.userAgent);

function Notifications() {
  const { t } = useTranslation();
  const [state, setState] = useState(null);
  const [busy, setBusy] = useState(false);
  useEffect(() => { pushState().then(setState); }, []);
  const toggle = async () => {
    setBusy(true);
    try { setState(state === 'on' ? await disablePush() : await enablePush()); } catch (e) { toast(errorMessage(t, e)); }
    setBusy(false);
  };
  return (
    <section className="card stack">
      <h2>{t('settings.notifications')}</h2>
      <p className="small muted">{t('settings.notif_desc')}</p>
      {state === null && <Spinner />}
      {state === 'unsupported' && <Notice>{t('settings.notif_unsupported')}</Notice>}
      {state === 'unavailable' && <Notice>{t('settings.notif_unavailable')}</Notice>}
      {state === 'denied' && <Notice tone="warn">{t('settings.notif_denied')}</Notice>}
      {state === 'on' && <Notice tone="ok">{t('settings.notif_on')}</Notice>}
      {(state === 'on' || state === 'off') && (
        <button className="btn btn-outline" onClick={toggle} disabled={busy}>
          {state === 'on' ? <><BellOff size={18} />{t('settings.notif_disable')}</> : <><Bell size={18} />{t('settings.notif_enable')}</>}
        </button>
      )}
    </section>
  );
}

export default function Settings() {
  const { t, i18n } = useTranslation();
  const { user, logout, logoutAll, updateProfile, deleteAccount } = useAuth();
  const navigate = useNavigate();
  const [name, setName] = useState(user?.full_name || '');
  const install = useInstallPrompt();

  const changeLanguage = async (code) => {
    setLanguage(code);
    if (user) await updateProfile({ language: code }).catch(() => {});
  };
  const saveName = async (e) => {
    e.preventDefault();
    try { await updateProfile({ full_name: name.trim() || null }); toast(t('settings.saved')); } catch (err) { toast(errorMessage(t, err)); }
  };
  const remove = async () => {
    if (!window.confirm(t('settings.delete_confirm'))) return;
    try { await deleteAccount(); navigate('/'); } catch (err) { toast(errorMessage(t, err)); }
  };

  return (
    <div className="stack">
      <h1>{t('settings.title')}</h1>
      <section className="card stack">
        <label htmlFor="lang"><h2>{t('settings.language')}</h2></label>
        <select id="lang" className="select" value={i18n.language} onChange={(e) => changeLanguage(e.target.value)}>
          {LANGUAGES.map((l) => <option key={l.code} value={l.code}>{l.label}</option>)}
        </select>
      </section>

      {!isStandalone() && (install || isIos()) && (
        <section className="card stack">
          <h2>{t('settings.install')}</h2>
          <p className="small muted">{install ? t('settings.install_desc') : t('settings.install_ios')}</p>
          {install && <button className="btn btn-primary" onClick={() => install.prompt()}><Download size={18} />{t('settings.install')}</button>}
        </section>
      )}

      {user && <Notifications />}

      <section className="card stack">
        <h2>{t('settings.account')}</h2>
        {user ? (
          <>
            <p className="small muted">{t('settings.signed_in_as', { email: user.email })}</p>
            <form className="row" onSubmit={saveName}>
              <label htmlFor="name" className="sr-only">{t('settings.name')}</label>
              <input id="name" className="input grow" value={name} maxLength={120} placeholder={t('settings.name')}
                onChange={(e) => setName(e.target.value)} />
              <button className="btn btn-outline">{t('common.save')}</button>
            </form>
            <button className="btn btn-outline" onClick={async () => { await logout(); navigate('/'); }}>
              <LogOut size={18} />{t('nav.logout')}
            </button>
            <button className="btn btn-ghost" onClick={async () => {
              try { await logoutAll(); navigate('/'); } catch (err) { toast(errorMessage(t, err)); }
            }}>
              <LogOut size={18} />{t('settings.logout_all')}
            </button>
            <button className="btn btn-ghost" style={{ color: 'var(--danger)' }} onClick={remove}>
              <Trash2 size={18} />{t('settings.delete_account')}
            </button>
          </>
        ) : <Link className="btn btn-primary" to="/login">{t('nav.login')}</Link>}
      </section>
      <p className="tiny center"><Link to="/privacy">{t('settings.privacy')}</Link> · <Link to="/terms">{t('settings.terms')}</Link></p>
    </div>
  );
}
