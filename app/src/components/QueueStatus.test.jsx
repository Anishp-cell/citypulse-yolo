import { render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { describe, expect, it, vi } from 'vitest';
import { AuthProvider } from '../hooks/useAuth';
import { mockApi } from '../test/render';
import '../i18n';

vi.mock('../lib/offlineQueue', () => ({
  list: vi.fn(async () => [{ id: '1' }]),
  flush: vi.fn(async () => ({ sent: 0, dropped: 0, kept: 1 })),
}));
import { flush } from '../lib/offlineQueue';
import QueueStatus, { RETRY_EVERY_MS } from './QueueStatus';

describe('QueueStatus', () => {
  it('keeps retrying queued reports on a timer, not only on the online event', async () => {
    // Review I9: if the first flush hits a cold server, the queue must not wait for the next "online" event.
    const intervals = vi.spyOn(window, 'setInterval');
    mockApi({ '/api/auth/me': { id: 1, email: 'a@b.in', role: 'citizen', language: 'en' } });
    render(
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <AuthProvider><QueueStatus /></AuthProvider>
      </QueryClientProvider>,
    );
    await waitFor(() => expect(flush).toHaveBeenCalledTimes(1));
    expect(await screen.findByText('1 report waiting to be sent')).toBeInTheDocument();
    const retry = intervals.mock.calls.findLast(([, ms]) => ms === RETRY_EVERY_MS); // the live timer
    expect(retry).toBeTruthy();
    retry[0]();
    await waitFor(() => expect(flush).toHaveBeenCalledTimes(2));
  });
});
