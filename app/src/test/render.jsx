import { render } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { vi } from 'vitest';
import { AuthProvider } from '../hooks/useAuth';
import '../i18n';

/** Mock fetch with a table of `"METHOD /path"` (or "/path" for GET) -> JSON body | (init) => [status, body]. */
export function mockApi(routes) {
  return vi.spyOn(globalThis, 'fetch').mockImplementation(async (url, init = {}) => {
    const path = String(url).split('?')[0];
    const method = init.method || 'GET';
    const handler = routes[`${method} ${path}`] ?? (method === 'GET' ? routes[path] : undefined);
    if (handler === undefined) return new Response(JSON.stringify({ detail: 'not_found' }), { status: 404, headers: { 'Content-Type': 'application/json' } });
    const resolved = typeof handler === 'function' ? handler(init) : handler;
    const [status, body] = Array.isArray(resolved) && typeof resolved[0] === 'number' ? resolved : [200, resolved];
    return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
  });
}

export function renderPage(element, { route = '/', path = '/' } = {}) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <AuthProvider>
        <MemoryRouter initialEntries={[route]}>
          <Routes><Route path={path} element={element} /></Routes>
        </MemoryRouter>
      </AuthProvider>
    </QueryClientProvider>,
  );
}
