import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Ambulance, MapPin, Phone, Shield } from 'lucide-react';
import { toast } from '../components/Toaster';
import { Notice, Spinner } from '../components/ui';
import { getPosition, googleMapsUrl } from '../lib/geo';
import { sharePage } from '../lib/share';

const SITUATIONS = ['injured', 'vehicles', 'unsure'];

// Everything here is bundled and precached by the service worker, so it works with no connection.
export default function Emergency() {
  const { t } = useTranslation();
  const [situation, setSituation] = useState(null);
  const [locating, setLocating] = useState(false);

  const shareLocation = async () => {
    setLocating(true);
    try {
      const p = await getPosition({ timeout: 10000, maximumAge: 0 });
      const url = googleMapsUrl(p.lat, p.lng);
      const r = await sharePage({ title: t('emergency.title'), text: t('emergency.share_location_text'), url });
      if (r === 'copied') toast(t('common.copied'));
    } catch {
      toast(t('errors.geolocation_unsupported'));
    } finally {
      setLocating(false);
    }
  };

  const steps = situation ? t(`emergency.steps.${situation}`, { returnObjects: true }) : [];
  return (
    <div className="stack">
      <h1>{t('emergency.title')}</h1>
      <div className="sos">
        <a className="btn btn-danger btn-lg" href="tel:112"><Phone size={22} />{t('emergency.call_112')}</a>
        <div className="choice-grid">
          <a className="btn btn-outline btn-lg" href="tel:108"><Ambulance size={20} />{t('emergency.call_108')}</a>
          <a className="btn btn-outline btn-lg" href="tel:100"><Shield size={20} />{t('emergency.call_100')}</a>
        </div>
        <button className="btn btn-outline" onClick={shareLocation} disabled={locating}>
          {locating ? <><Spinner />{t('emergency.locating')}</> : <><MapPin size={18} />{t('emergency.share_location')}</>}
        </button>
      </div>

      <section className="card stack" aria-labelledby="triage">
        <h2 id="triage">{t('emergency.triage_title')}</h2>
        <div className="stack-sm">
          {SITUATIONS.map((s) => (
            <button key={s} className="choice" aria-pressed={situation === s} onClick={() => setSituation(s)}>
              {t(`emergency.${s}`)}
            </button>
          ))}
        </div>
        {situation && (
          <div className="stack-sm">
            <h3>{t('emergency.steps_title')}</h3>
            <ol className="steps-list">{steps.map((s) => <li key={s}>{s}</li>)}</ol>
          </div>
        )}
      </section>
      <Notice tone="neutral">{t('emergency.disclaimer')}</Notice>
    </div>
  );
}
