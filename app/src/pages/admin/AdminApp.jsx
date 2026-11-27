import { createElement } from 'react';
import { Link, Navigate, NavLink, Outlet, Route, Routes, useLocation } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { useIsFetching, useQueryClient } from '@tanstack/react-query';
import { ChartColumn, ExternalLink, ListChecks, Map, RefreshCw, ShieldAlert, Users as UsersIcon } from 'lucide-react';
import { isStaff, useAuth } from '../../hooks/useAuth';
import { useAdminStats, useModeration } from '../../api/hooks';
import { Brand } from '../../components/Layouts';
import { Loading } from '../../components/ui';
import { ConnectionBanner } from '../../components/Toaster';
import { formatDate } from '../../lib/format';
import { initials } from './consoleUtils';
import Dashboard from './Dashboard';
import Queue from './Queue';
import AdminMap from './AdminMap';
import Moderation from './Moderation';
import Users from './Users';

const NAV = [
  { group: 'operate', to: '/admin', end: true, icon: ChartColumn, label: 'dashboard' },
  { group: 'operate', to: '/admin/queue', icon: ListChecks, label: 'queue', count: 'active' },
  { group: 'operate', to: '/admin/moderation', icon: ShieldAlert, label: 'moderation', count: 'review' },
  { group: 'operate', to: '/admin/map', icon: Map, label: 'map' },
  { group: 'manage', to: '/admin/users', icon: UsersIcon, label: 'users', adminOnly: true },
];

function AdminLayout() {
  const { t, i18n } = useTranslation();
  const { user } = useAuth();
  const location = useLocation();
  const qc = useQueryClient();
  const stats = useAdminStats();
  const moderation = useModeration();
  const fetching = useIsFetching({ queryKey: ['admin'] }) > 0;

  const counts = {
    active: stats.data?.totals.active,
    review: moderation.data ? moderation.data.pending.length + moderation.data.flagged.length : undefined,
  };
  const items = NAV.filter((n) => !n.adminOnly || user.role === 'admin');
  const current = [...items].reverse().find((n) => (n.end ? location.pathname === n.to : location.pathname.startsWith(n.to)));

  return (
    <div className="admin-shell">
      <nav className="admin-nav" aria-label={t('admin.title')}>
        <Brand />
        {['operate', 'manage'].map((group) => {
          const groupItems = items.filter((n) => n.group === group);
          if (!groupItems.length) return null;
          return (
            <div key={group} style={{ display: 'contents' }}>
              <div className="nav-group">{t(`admin.${group}`)}</div>
              {groupItems.map((n) => {
                const count = n.count ? counts[n.count] : undefined;
                return (
                  <NavLink key={n.to} to={n.to} end={n.end} className="nav-item">
                    {createElement(n.icon, { size: 17, 'aria-hidden': true })}
                    {t(`admin.${n.label}`)}
                    {count > 0 && (
                      <span className={`nav-count${n.count === 'review' ? ' attn' : ''}`}>
                        {count}<span className="sr-only"> {t(n.count === 'review' ? 'admin.pending' : 'admin.active')}</span>
                      </span>
                    )}
                  </NavLink>
                );
              })}
            </div>
          );
        })}
      </nav>

      <div className="admin-body">
        <header className="admin-topbar">
          <span className="crumb-muted">{t('admin.title')}</span>
          <span className="crumb-sep" aria-hidden="true">/</span>
          <span className="crumb">{current ? t(`admin.${current.label}`) : ''}</span>
          <div className="topbar-actions">
            {stats.dataUpdatedAt > 0 && (
              <span className="updated">
                <span className="live-dot" aria-hidden="true" />
                {t('admin.updated', { time: formatDate(new Date(stats.dataUpdatedAt).toISOString(), i18n.language, { hour: 'numeric', minute: '2-digit' }) })}
              </span>
            )}
            <button type="button" className="btn btn-outline btn-sm" onClick={() => qc.invalidateQueries({ queryKey: ['admin'] })}
              disabled={fetching} aria-label={t('admin.refresh')} title={t('admin.refresh')}>
              <RefreshCw size={14} aria-hidden="true" style={fetching ? { animation: 'spin 0.8s linear infinite' } : undefined} />
            </button>
            <Link className="btn btn-outline btn-sm" to="/"><ExternalLink size={14} aria-hidden="true" />{t('admin.back_to_app')}</Link>
            <div className="user-chip">
              <span className="avatar" aria-hidden="true">{initials(user)}</span>
              <span className="who">
                <span>{user.full_name || user.email}</span>
                <span>{t(`admin.roles.${user.role}`)}</span>
              </span>
            </div>
          </div>
        </header>
        <main className="admin-main">
          <ConnectionBanner />
          <div key={location.pathname} className="page-enter">
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  );
}

export default function AdminApp() {
  const { user, loading } = useAuth();
  if (loading) return <Loading />;
  if (!user) return <Navigate to="/login?next=/admin" replace />;
  if (!isStaff(user)) return <Navigate to="/" replace />;
  return (
    <Routes>
      <Route element={<AdminLayout />}>
        <Route index element={<Dashboard />} />
        <Route path="queue" element={<Queue />} />
        <Route path="moderation" element={<Moderation />} />
        <Route path="map" element={<AdminMap />} />
        <Route path="users" element={user.role === 'admin' ? <Users /> : <Navigate to="/admin" replace />} />
        <Route path="*" element={<Navigate to="/admin" replace />} />
      </Route>
    </Routes>
  );
}
