import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { Building2, Camera, ChevronRight, MapPin, Siren } from 'lucide-react';
import { useHazards, useMyReports } from '../api/hooks';
import { useAuth } from '../hooks/useAuth';
import HazardCard from '../components/HazardCard';
import { bboxParam, getPosition, haversineM, locationAlreadyGranted } from '../lib/geo';

function NearbyCount() {
  const { t } = useTranslation();
  const [pos, setPos] = useState(null);
  useEffect(() => {
    // Only use location if the user already allowed it; never prompt on the home page.
    locationAlreadyGranted().then((ok) => ok && getPosition({ timeout: 8000 }).then(setPos).catch(() => {}));
  }, []);
  const { data } = useHazards(pos ? bboxParam(pos.lat, pos.lng, 1000) : null, 'active');
  if (!pos || !data) return null;
  const count = data.hazards.filter((h) => haversineM(pos.lat, pos.lng, h.lat, h.lng) <= 1000).length;
  return (
    <Link to="/map" className="card card-flat spread" style={{ textDecoration: 'none', color: 'inherit' }}>
      <span className="row"><MapPin size={18} color="var(--danger)" />
        {count ? t('home.nearby', { count }) : t('home.nearby_none')}</span>
      <ChevronRight size={18} />
    </Link>
  );
}

export default function Home() {
  const { t } = useTranslation();
  const { user } = useAuth();
  const mine = useMyReports(Boolean(user));
  const active = (mine.data?.hazards || []).filter((h) => h.status !== 'fixed').slice(0, 3);
  return (
    <div className="stack">
      <section className="stack-sm" style={{ padding: '8px 0' }}>
        <h1>{t('home.title')}</h1>
        <p className="muted">{t('home.subtitle')}</p>
      </section>
      <Link to="/report" className="btn btn-primary btn-lg btn-block"><Camera size={22} />{t('home.report_cta')}</Link>
      <Link to="/emergency" className="btn btn-outline btn-block" style={{ color: 'var(--danger)' }}>
        <Siren size={18} />{t('home.emergency_cta')}
      </Link>
      <NearbyCount />
      {active.length > 0 && (
        <section className="stack-sm">
          <div className="spread"><h2>{t('home.my_active')}</h2><Link to="/me" className="small">{t('nav.mine')}</Link></div>
          {active.map((h) => <HazardCard key={h.id} hazard={h} />)}
        </section>
      )}
      <section className="card stack">
        <h2>{t('home.how_title')}</h2>
        <ol className="steps-list">
          <li>{t('home.how_1')}</li>
          <li>{t('home.how_2')}</li>
          <li>{t('home.how_3')}</li>
        </ol>
      </section>
      <Link to="/city" className="card card-flat spread" style={{ textDecoration: 'none', color: 'inherit' }}>
        <span className="row"><Building2 size={18} />{t('home.city_link')}</span><ChevronRight size={18} />
      </Link>
      <p className="tiny center"><Link to="/privacy">{t('settings.privacy')}</Link> · <Link to="/terms">{t('settings.terms')}</Link></p>
    </div>
  );
}
