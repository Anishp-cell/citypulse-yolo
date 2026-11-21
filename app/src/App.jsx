import { Component, lazy, Suspense } from 'react';
import { Route, Routes } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { PublicLayout } from './components/Layouts';
import { Loading } from './components/ui';
import Home from './pages/Home';
import Hazard from './pages/Hazard';
import MyReports from './pages/MyReports';
import City from './pages/City';
import Emergency from './pages/Emergency';
import Settings from './pages/Settings';
import NotFound from './pages/NotFound';
import { Privacy, Terms } from './pages/Legal';
import { ForgotPassword, Login, Register, ResetPassword } from './pages/auth/AuthPages';

const AdminApp = lazy(() => import('./pages/admin/AdminApp'));
const MapPage = lazy(() => import('./pages/MapPage'));
const ReportFlow = lazy(() => import('./pages/report/ReportFlow'));

function Crash() {
  const { t } = useTranslation();
  return (
    <div className="page"><div className="card stack center">
      <h1>{t('common.error_title')}</h1>
      <button className="btn btn-primary" onClick={() => window.location.reload()}>{t('common.retry')}</button>
    </div></div>
  );
}

class ErrorBoundary extends Component {
  state = { failed: false };
  static getDerivedStateFromError() { return { failed: true }; }
  componentDidCatch(error, info) { console.error(error, info); }
  render() { return this.state.failed ? <Crash /> : this.props.children; }
}

export default function App() {
  return (
    <ErrorBoundary>
      <Suspense fallback={<Loading />}>
        <Routes>
          <Route path="/admin/*" element={<AdminApp />} />
          <Route element={<PublicLayout />}>
            <Route index element={<Home />} />
            <Route path="map" element={<MapPage />} />
            <Route path="report" element={<ReportFlow />} />
            <Route path="h/:id" element={<Hazard />} />
            <Route path="me" element={<MyReports />} />
            <Route path="city" element={<City />} />
            <Route path="emergency" element={<Emergency />} />
            <Route path="settings" element={<Settings />} />
            <Route path="privacy" element={<Privacy />} />
            <Route path="terms" element={<Terms />} />
            <Route path="login" element={<Login />} />
            <Route path="register" element={<Register />} />
            <Route path="forgot-password" element={<ForgotPassword />} />
            <Route path="reset-password" element={<ResetPassword />} />
            <Route path="*" element={<NotFound />} />
          </Route>
        </Routes>
      </Suspense>
    </ErrorBoundary>
  );
}
