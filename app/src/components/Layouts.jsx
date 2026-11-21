import { Link, NavLink, Outlet, useLocation } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { ClipboardList, House, LayoutDashboard, LogIn, Map, Plus, Settings, Siren } from 'lucide-react';
import { isStaff, useAuth } from '../hooks/useAuth';
import { ConnectionBanner } from './Toaster';
import QueueStatus from './QueueStatus';

export function Brand() {
  const { t } = useTranslation();
  return (
    <Link to="/" className="brand" aria-label={t('app.name')}>
      <span className="brand-mark" aria-hidden="true"><Siren size={16} /></span>
      <span>{t('app.name')}</span>
    </Link>
  );
}

export function PublicLayout() {
  const { t } = useTranslation();
  const { user } = useAuth();
  const { pathname } = useLocation();
  const fullBleed = pathname === '/map';
  return (
    <div className="app-shell">
      <header className="app-header">
        <Brand />
        <div className="header-actions">
          {isStaff(user) && (
            <Link to="/admin" className="btn btn-sm btn-outline"><LayoutDashboard size={16} />{t('nav.admin')}</Link>
          )}
          {user
            ? <Link to="/settings" className="btn btn-ghost icon-btn" aria-label={t('nav.settings')}><Settings size={20} /></Link>
            : <Link to="/login" className="btn btn-sm btn-primary"><LogIn size={16} />{t('nav.login')}</Link>}
        </div>
      </header>
      <ConnectionBanner />
      <QueueStatus />
      <main className={fullBleed ? 'page-full' : 'page'}>
        <Outlet />
      </main>
      <nav className="bottom-nav" aria-label="Main">
        <NavLink to="/" end><House size={22} />{t('nav.home')}</NavLink>
        <NavLink to="/map"><Map size={22} />{t('nav.map')}</NavLink>
        <NavLink to="/report" className="nav-report">
          <span className="nav-report-dot"><Plus size={24} /></span>{t('nav.report')}
        </NavLink>
        <NavLink to="/me"><ClipboardList size={22} />{t('nav.mine')}</NavLink>
        <NavLink to="/emergency" className="nav-emergency"><Siren size={22} />{t('nav.emergency')}</NavLink>
      </nav>
    </div>
  );
}
