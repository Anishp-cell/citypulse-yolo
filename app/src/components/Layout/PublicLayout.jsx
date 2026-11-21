import { Outlet, NavLink, useNavigate } from 'react-router-dom';
import { useAuth } from '../../hooks/useAuth';
import { useTranslation } from 'react-i18next';
import { Home, Camera, ClipboardList, ShieldCheck, LogOut, Globe } from 'lucide-react';
import './PublicLayout.css';

export default function PublicLayout() {
  const { user, logout } = useAuth();
  const { t, i18n } = useTranslation();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const toggleLang = () => {
    const next = i18n.language === 'en' ? 'hi' : 'en';
    i18n.changeLanguage(next);
    localStorage.setItem('cp_lang', next);
  };

  return (
    <div className="public-layout">
      {/* Top Header */}
      <header className="public-header">
        <div className="public-header-inner">
          <div className="public-logo" onClick={() => navigate('/')}>
            <span>🚨</span>
            <span className="logo-text">CityPulse</span>
          </div>
          <div className="public-header-actions">
            <button className="lang-btn" onClick={toggleLang} title="Switch language">
              <Globe size={18} />
              <span>{i18n.language === 'en' ? 'हिन्दी' : 'English'}</span>
            </button>
            {user ? (
              <button className="btn btn-ghost btn-sm" onClick={handleLogout}>
                <LogOut size={16} />
                {t('nav.logout')}
              </button>
            ) : (
              <button className="btn btn-primary btn-sm" onClick={() => navigate('/login')}>
                {t('nav.login')}
              </button>
            )}
          </div>
        </div>
      </header>

      {/* Page Content */}
      <main className="public-main">
        <Outlet />
      </main>

      {/* Bottom Navigation (mobile-first) */}
      <nav className="bottom-nav">
        <NavLink to="/" end className={({ isActive }) => `bottom-nav-item ${isActive ? 'active' : ''}`}>
          <Home size={22} />
          <span>{t('nav.home')}</span>
        </NavLink>
        <NavLink to="/report" className={({ isActive }) => `bottom-nav-item ${isActive ? 'active' : ''}`}>
          <Camera size={22} />
          <span>{t('nav.report')}</span>
        </NavLink>
        <NavLink to="/my-reports" className={({ isActive }) => `bottom-nav-item ${isActive ? 'active' : ''}`}>
          <ClipboardList size={22} />
          <span>{t('nav.my_reports')}</span>
        </NavLink>
        <NavLink to="/safety" className={({ isActive }) => `bottom-nav-item ${isActive ? 'active' : ''}`}>
          <ShieldCheck size={22} />
          <span>{t('nav.safety')}</span>
        </NavLink>
      </nav>
    </div>
  );
}
