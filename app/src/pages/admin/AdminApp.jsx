import { Navigate, NavLink, Outlet, Route, Routes } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { ArrowLeft, ChartColumn, ListChecks, Map, ShieldAlert, Users as UsersIcon } from 'lucide-react';
import { isStaff, useAuth } from '../../hooks/useAuth';
import { Brand } from '../../components/Layouts';
import { Loading } from '../../components/ui';
import { ConnectionBanner } from '../../components/Toaster';
import Dashboard from './Dashboard';
import Queue from './Queue';
import AdminMap from './AdminMap';
import Moderation from './Moderation';
import Users from './Users';

function AdminLayout() {
  const { t } = useTranslation();
  const { user } = useAuth();
  return (
    <div className="admin-shell">
      <nav className="admin-nav" aria-label={t('admin.title')}>
        <div style={{ padding: '4px 8px 12px' }}><Brand /></div>
        <NavLink to="/admin" end><ChartColumn size={18} />{t('admin.dashboard')}</NavLink>
        <NavLink to="/admin/queue"><ListChecks size={18} />{t('admin.queue')}</NavLink>
        <NavLink to="/admin/moderation"><ShieldAlert size={18} />{t('admin.moderation')}</NavLink>
        <NavLink to="/admin/map"><Map size={18} />{t('admin.map')}</NavLink>
        {user.role === 'admin' && <NavLink to="/admin/users"><UsersIcon size={18} />{t('admin.users')}</NavLink>}
        <NavLink to="/" style={{ marginTop: 'auto' }}><ArrowLeft size={18} />{t('admin.back_to_app')}</NavLink>
      </nav>
      <main className="admin-main">
        <ConnectionBanner />
        <Outlet />
      </main>
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
