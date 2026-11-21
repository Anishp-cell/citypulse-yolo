import { Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';

export default function NotFound() {
  const { t } = useTranslation();
  return (
    <div className="card stack center">
      <h1>{t('common.not_found')}</h1>
      <Link className="btn btn-primary" to="/">{t('common.go_home')}</Link>
    </div>
  );
}
