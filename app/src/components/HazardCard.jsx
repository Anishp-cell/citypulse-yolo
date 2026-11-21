import { Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { Users } from 'lucide-react';
import { placeName } from '../lib/format';
import { StatusBadge, TypeLabel } from './ui';

export default function HazardCard({ hazard }) {
  const { t } = useTranslation();
  return (
    <Link to={`/h/${hazard.id}`} className="card card-flat hazard-card">
      {hazard.thumb_url
        ? <img className="hazard-thumb" src={hazard.thumb_url} alt="" loading="lazy" />
        : <div className="hazard-thumb" aria-hidden="true" />}
      <div className="grow stack-sm" style={{ gap: 2 }}>
        <div className="spread">
          <strong className="ellipsis"><TypeLabel type={hazard.type} size={15} /></strong>
          <StatusBadge status={hazard.status} />
        </div>
        <span className="small muted ellipsis">{placeName(hazard)}{hazard.ward ? ` · ${hazard.ward}` : ''}</span>
        <span className="tiny row" style={{ gap: 10 }}>
          <span>{t(hazard.status === 'fixed' ? 'hazard.fixed_after' : 'hazard.open_for',
            { days: t('common.days', { count: hazard.days_open }) })}</span>
          <span className="row" style={{ gap: 3 }}><Users size={12} aria-hidden="true" />{hazard.report_count}</span>
        </span>
      </div>
    </Link>
  );
}
