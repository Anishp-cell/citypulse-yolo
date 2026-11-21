import { describe, expect, it, vi } from 'vitest';
import { ApiError, request } from './client';

function respond(status, body) {
  return Promise.resolve(new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } }));
}

const fast = { retryDelays: [0, 0, 0], wakeAfterMs: 1e9 };

describe('request', () => {
  it('sends the CSRF header and cookies', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation(() => respond(200, { ok: true }));
    await request('/api/x', { ...fast });
    const [, init] = fetchMock.mock.calls[0];
    expect(init.credentials).toBe('include');
    expect(init.headers['X-Requested-With']).toBe('citypulse');
  });

  it('sends JSON bodies', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation(() => respond(200, {}));
    await request('/api/x', { method: 'POST', body: { a: 1 }, ...fast });
    const [, init] = fetchMock.mock.calls[0];
    expect(init.headers['Content-Type']).toBe('application/json');
    expect(init.body).toBe('{"a":1}');
  });

  it('retries a 503 three times for GET, then succeeds', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch')
      .mockImplementationOnce(() => respond(503, {}))
      .mockImplementationOnce(() => respond(503, {}))
      .mockImplementationOnce(() => respond(503, {}))
      .mockImplementationOnce(() => respond(200, { ok: 1 }));
    expect(await request('/api/x', fast)).toEqual({ ok: 1 });
    expect(fetchMock).toHaveBeenCalledTimes(4);
  });

  it('does not retry a POST unless marked idempotent', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation(() => respond(503, {}));
    await expect(request('/api/x', { method: 'POST', ...fast })).rejects.toBeInstanceOf(ApiError);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('retries an idempotent POST', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch')
      .mockImplementationOnce(() => Promise.reject(new TypeError('Failed to fetch')))
      .mockImplementationOnce(() => respond(200, { ok: 1 }));
    await request('/api/x', { method: 'POST', idempotent: true, ...fast });
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it('maps the error detail to a code', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation(() => respond(409, { detail: 'already_voted' }));
    await expect(request('/api/x', fast)).rejects.toMatchObject({ status: 409, code: 'already_voted' });
  });

  it('maps pydantic validation errors to validation_error', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation(() => respond(422, { detail: [{ loc: ['body'], msg: 'x' }] }));
    await expect(request('/api/x', fast)).rejects.toMatchObject({ status: 422, code: 'validation_error' });
  });

  it('reports network failures as network_error after retries', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation(() => Promise.reject(new TypeError('offline')));
    await expect(request('/api/x', fast)).rejects.toMatchObject({ status: 0, code: 'network_error' });
  });

  it('announces a slow request so the UI can say the server is waking up', async () => {
    const seen = [];
    window.addEventListener('cp:waking', () => seen.push(1));
    vi.spyOn(globalThis, 'fetch').mockImplementation(
      () => new Promise((resolve) => setTimeout(() => resolve(new Response('{}', { status: 200 })), 30)));
    await request('/api/x', { retryDelays: [], wakeAfterMs: 5 });
    expect(seen.length).toBe(1);
  });
});
