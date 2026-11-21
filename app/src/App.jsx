import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './hooks/useAuth';
import AdminLayout from './components/Layout/AdminLayout';
import PublicLayout from './components/Layout/PublicLayout';
import Login from './pages/auth/Login';
import Register from './pages/auth/Register';
import Dashboard from './pages/admin/Dashboard';
import IncidentMap from './pages/admin/IncidentMap';
import Incidents from './pages/admin/Incidents';
import Analytics from './pages/admin/Analytics';
import Home from './pages/public/Home';
import Report from './pages/public/Report';
import MyReports from './pages/public/MyReports';
import Safety from './pages/public/Safety';
import './i18n';

function RequireAuth({ children, roles }) {
  const { user, loading } = useAuth();
  if (loading) return <div className="page-loading"><span className="spinner" style={{width:32,height:32}} /></div>;
  if (!user) return <Navigate to="/login" replace />;
  if (roles && !roles.includes(user.role)) return <Navigate to="/" replace />;
  return children;
}

function AppRoutes() {
  const { user, loading } = useAuth();
  if (loading) return <div className="page-loading" style={{minHeight:'100vh',display:'flex',alignItems:'center',justifyContent:'center'}}><span className="spinner" style={{width:32,height:32}} /></div>;

  return (
    <Routes>
      {/* Auth */}
      <Route path="/login" element={user ? <Navigate to={user.role === 'citizen' ? '/' : '/admin'} replace /> : <Login />} />
      <Route path="/register" element={user ? <Navigate to="/" replace /> : <Register />} />

      {/* Admin Routes */}
      <Route path="/admin" element={<RequireAuth roles={['admin','officer']}><AdminLayout /></RequireAuth>}>
        <Route index element={<Dashboard />} />
        <Route path="map" element={<IncidentMap />} />
        <Route path="incidents" element={<Incidents />} />
        <Route path="analytics" element={<Analytics />} />
      </Route>

      {/* Public Routes */}
      <Route element={<PublicLayout />}>
        <Route path="/" element={<Home />} />
        <Route path="/report" element={<RequireAuth><Report /></RequireAuth>} />
        <Route path="/my-reports" element={<RequireAuth><MyReports /></RequireAuth>} />
        <Route path="/safety" element={<Safety />} />
      </Route>

      {/* Fallback */}
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <AppRoutes />
      </AuthProvider>
    </BrowserRouter>
  );
}
