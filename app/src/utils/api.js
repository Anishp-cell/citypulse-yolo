// API client utility for CityPulse

const API_BASE = '';

export function getToken() {
  return localStorage.getItem('cp_token');
}

export function setToken(token) {
  localStorage.setItem('cp_token', token);
}

export function clearToken() {
  localStorage.removeItem('cp_token');
}

function authHeaders() {
  const token = getToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function request(url, options = {}) {
  const { headers = {}, ...rest } = options;
  const res = await fetch(`${API_BASE}${url}`, {
    headers: { ...authHeaders(), ...headers },
    ...rest,
  });

  if (res.status === 401) {
    clearToken();
    window.location.href = '/login';
    throw new Error('Session expired');
  }

  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.detail || `HTTP ${res.status}`);
  }

  return res.json();
}

export const api = {
  get: (url) => request(url),
  post: (url, body) =>
    request(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  patch: (url, body) =>
    request(url, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  upload: (url, formData) =>
    request(url, { method: 'POST', body: formData }),

  // Auth
  login: async (email, password) => {
    const res = await fetch(`${API_BASE}/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams({ username: email, password }),
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.detail || 'Login failed');
    }
    const data = await res.json();
    setToken(data.access_token);
    return data;
  },

  register: async ({ email, password, phone, role = 'citizen' }) => {
    const res = await fetch(`${API_BASE}/auth/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password, phone: phone || null, role }),
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.detail || 'Registration failed');
    }
    return res.json();
  },

  me: () => request('/auth/me'),
};
