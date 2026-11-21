import { useState } from 'react';
import { Link, useNavigate, useSearchParams } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { api } from '../../api/client';
import { isStaff, useAuth } from '../../hooks/useAuth';
import { Notice, Spinner } from '../../components/ui';
import { errorMessage } from '../../lib/format';

/** Only allow same-site relative redirects after login. */
function safeNext(next) {
  return next && next.startsWith('/') && !next.startsWith('//') ? next : null;
}

function useSubmit(fn) {
  const { t } = useTranslation();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const run = async (e) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try { await fn(); } catch (err) { setError(errorMessage(t, err)); } finally { setBusy(false); }
  };
  return { busy, error, run };
}

function Field({ id, label, hint, ...props }) {
  return (
    <div className="field">
      <label htmlFor={id}>{label}</label>
      <input id={id} className="input" {...props} />
      {hint && <span className="tiny">{hint}</span>}
    </div>
  );
}

export function Login() {
  const { t } = useTranslation();
  const { login } = useAuth();
  const navigate = useNavigate();
  const [params] = useSearchParams();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const { busy, error, run } = useSubmit(async () => {
    const user = await login(email, password);
    navigate(safeNext(params.get('next')) || (isStaff(user) ? '/admin' : '/'), { replace: true });
  });
  const next = params.get('next') ? `?next=${encodeURIComponent(params.get('next'))}` : '';
  return (
    <form className="card stack" onSubmit={run}>
      <h1>{t('auth.login_title')}</h1>
      {error && <Notice tone="danger">{error}</Notice>}
      <Field id="email" label={t('auth.email')} type="email" autoComplete="email" required value={email}
        onChange={(e) => setEmail(e.target.value)} />
      <Field id="password" label={t('auth.password')} type="password" autoComplete="current-password" required
        value={password} onChange={(e) => setPassword(e.target.value)} />
      <button className="btn btn-primary btn-lg" disabled={busy}>{busy ? <Spinner /> : t('auth.submit_login')}</button>
      <Link to="/forgot-password" className="small">{t('auth.forgot')}</Link>
      <Link to={`/register${next}`} className="small">{t('auth.no_account')}</Link>
    </form>
  );
}

export function Register() {
  const { t, i18n } = useTranslation();
  const { register } = useAuth();
  const navigate = useNavigate();
  const [params] = useSearchParams();
  const [form, setForm] = useState({ full_name: '', email: '', password: '' });
  const set = (k) => (e) => setForm((f) => ({ ...f, [k]: e.target.value }));
  const { busy, error, run } = useSubmit(async () => {
    await register({ ...form, full_name: form.full_name.trim() || null, language: i18n.language });
    navigate(safeNext(params.get('next')) || '/', { replace: true });
  });
  return (
    <form className="card stack" onSubmit={run}>
      <h1>{t('auth.register_title')}</h1>
      <p className="small muted">{t('auth.register_desc')}</p>
      {error && <Notice tone="danger">{error}</Notice>}
      <Field id="name" label={t('auth.name')} autoComplete="name" maxLength={120} value={form.full_name} onChange={set('full_name')} />
      <Field id="email" label={t('auth.email')} type="email" autoComplete="email" required value={form.email} onChange={set('email')} />
      <Field id="password" label={t('auth.password')} type="password" autoComplete="new-password" required minLength={8}
        hint={t('auth.password_hint')} value={form.password} onChange={set('password')} />
      <button className="btn btn-primary btn-lg" disabled={busy}>{busy ? <Spinner /> : t('auth.submit_register')}</button>
      <Link to="/login" className="small">{t('auth.have_account')}</Link>
      <p className="tiny"><Link to="/terms">{t('settings.terms')}</Link> · <Link to="/privacy">{t('settings.privacy')}</Link></p>
    </form>
  );
}

export function ForgotPassword() {
  const { t } = useTranslation();
  const [email, setEmail] = useState('');
  const [sent, setSent] = useState(false);
  const { busy, error, run } = useSubmit(async () => {
    await api.post('/api/auth/forgot-password', { email });
    setSent(true);
  });
  return (
    <form className="card stack" onSubmit={run}>
      <h1>{t('auth.forgot_title')}</h1>
      {sent ? <Notice tone="ok">{t('auth.link_sent')}</Notice> : (
        <>
          <p className="small muted">{t('auth.forgot_desc')}</p>
          {error && <Notice tone="danger">{error}</Notice>}
          <Field id="email" label={t('auth.email')} type="email" autoComplete="email" required value={email}
            onChange={(e) => setEmail(e.target.value)} />
          <button className="btn btn-primary" disabled={busy}>{busy ? <Spinner /> : t('auth.send_link')}</button>
        </>
      )}
      <Link to="/login" className="small">{t('auth.have_account')}</Link>
    </form>
  );
}

export function ResetPassword() {
  const { t } = useTranslation();
  const [params] = useSearchParams();
  const [password, setPassword] = useState('');
  const [done, setDone] = useState(false);
  const { busy, error, run } = useSubmit(async () => {
    await api.post('/api/auth/reset-password', { token: params.get('token') || '', new_password: password });
    setDone(true);
  });
  return (
    <form className="card stack" onSubmit={run}>
      <h1>{t('auth.reset_title')}</h1>
      {done ? <><Notice tone="ok">{t('auth.reset_done')}</Notice><Link className="btn btn-primary" to="/login">{t('nav.login')}</Link></> : (
        <>
          {error && <Notice tone="danger">{error}</Notice>}
          <Field id="password" label={t('auth.new_password')} type="password" autoComplete="new-password" required
            minLength={8} hint={t('auth.password_hint')} value={password} onChange={(e) => setPassword(e.target.value)} />
          <button className="btn btn-primary" disabled={busy}>{busy ? <Spinner /> : t('auth.reset_submit')}</button>
        </>
      )}
    </form>
  );
}
