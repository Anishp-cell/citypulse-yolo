import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Bell } from 'lucide-react';
import { enablePush, pushState } from '../lib/push';
import { toast } from './Toaster';

const DISMISS_KEY = 'cp_push_prompt_dismissed';

/** Asked only after a successful report, never on first load. */
export default function PushPrompt() {
  const { t } = useTranslation();
  const [visible, setVisible] = useState(false);
  useEffect(() => {
    let dismissed = false;
    try { dismissed = localStorage.getItem(DISMISS_KEY) === '1'; } catch { /* ignore */ }
    if (dismissed) return;
    pushState().then((s) => setVisible(s === 'off'));
  }, []);
  if (!visible) return null;
  const dismiss = () => {
    try { localStorage.setItem(DISMISS_KEY, '1'); } catch { /* ignore */ }
    setVisible(false);
  };
  const enable = async () => {
    const result = await enablePush().catch(() => 'unavailable');
    if (result === 'denied') toast(t('settings.notif_denied'));
    if (result === 'on') toast(t('settings.notif_on'));
    dismiss();
  };
  return (
    <div className="card stack" role="region" aria-label={t('push.prompt_title')}>
      <div className="row"><Bell size={20} color="var(--brand)" aria-hidden="true" /><h3>{t('push.prompt_title')}</h3></div>
      <p className="small muted">{t('push.prompt_desc')}</p>
      <div className="row">
        <button className="btn btn-primary grow" onClick={enable}>{t('push.enable')}</button>
        <button className="btn btn-ghost" onClick={dismiss}>{t('push.later')}</button>
      </div>
    </div>
  );
}
