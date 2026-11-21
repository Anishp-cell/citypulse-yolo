// Thin fetch wrapper: cookie session, CSRF header, retries for flaky networks and cold starts,
// and errors carrying a stable `code` that the UI translates.

export class ApiError extends Error {
  constructor(status, code, data) {
    super(code);
    this.status = status;
    this.code = code;
    this.data = data;
  }
}

const RETRY_STATUSES = new Set([502, 503, 504]);
const DEFAULT_RETRY_DELAYS = [1000, 2500, 5000];
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function errorCode(status, data) {
  if (Array.isArray(data?.detail)) return 'validation_error';
  if (typeof data?.detail === 'string') return data.detail;
  if (status === 429) return 'rate_limited';
  if (status >= 500) return 'server_error';
  return `http_${status}`;
}

/**
 * @param {string} path
 * @param {{method?: string, body?: any, form?: FormData, headers?: Record<string,string>,
 *          idempotent?: boolean, retryDelays?: number[], wakeAfterMs?: number, signal?: AbortSignal}} [opts]
 */
export async function request(path, opts = {}) {
  const { method = 'GET', body, form, headers = {}, idempotent = false, signal,
    retryDelays = DEFAULT_RETRY_DELAYS, wakeAfterMs = 3000 } = opts;
  const canRetry = method === 'GET' || idempotent;
  const init = {
    method,
    credentials: 'include',
    signal,
    headers: { 'X-Requested-With': 'citypulse', Accept: 'application/json', ...headers },
  };
  if (form) init.body = form;
  else if (body !== undefined) {
    init.headers['Content-Type'] = 'application/json';
    init.body = JSON.stringify(body);
  }

  const wakeTimer = setTimeout(() => window.dispatchEvent(new CustomEvent('cp:waking')), wakeAfterMs);
  try {
    for (let attempt = 0; ; attempt++) {
      let res;
      try {
        res = await fetch(path, init);
      } catch (e) {
        if (e?.name === 'AbortError') throw e;
        if (canRetry && attempt < retryDelays.length) {
          await sleep(retryDelays[attempt]);
          continue;
        }
        throw new ApiError(0, 'network_error');
      }
      if (RETRY_STATUSES.has(res.status) && canRetry && attempt < retryDelays.length) {
        await sleep(retryDelays[attempt]);
        continue;
      }
      const isJson = (res.headers.get('content-type') || '').includes('json');
      const data = isJson ? await res.json().catch(() => null) : null;
      if (!res.ok) throw new ApiError(res.status, errorCode(res.status, data), data);
      return data;
    }
  } finally {
    clearTimeout(wakeTimer);
    window.dispatchEvent(new CustomEvent('cp:awake'));
  }
}

export const api = {
  get: (path, opts) => request(path, opts),
  post: (path, body, opts) => request(path, { ...opts, method: 'POST', body }),
  patch: (path, body, opts) => request(path, { ...opts, method: 'PATCH', body }),
  del: (path, body, opts) => request(path, { ...opts, method: 'DELETE', body }),
  form: (path, form, opts) => request(path, { ...opts, method: 'POST', form }),
};
