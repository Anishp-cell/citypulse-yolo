import { beforeEach, describe, expect, it, vi } from 'vitest';
import { clear } from 'idb-keyval';
import { ApiError } from '../api/client';
import { enqueue, flush, list } from './offlineQueue';

const item = (n) => ({ idempotencyKey: `key-${n}`, bytes: new Uint8Array([1, 2, 3]).buffer, mime: 'image/jpeg',
  lat: 18.52, lng: 73.85, accuracy: 10, source: 'gps', type: 'pothole', severity: 'medium', note: '' });

beforeEach(async () => { await clear(); });

describe('offline queue', () => {
  it('stores and lists items with an id and timestamp', async () => {
    await enqueue(item(1));
    const items = await list();
    expect(items).toHaveLength(1);
    expect(items[0].id).toBeTruthy();
    expect(items[0].createdAt).toBeTruthy();
    expect(items[0].idempotencyKey).toBe('key-1');
  });

  it('flush sends each item with its original idempotency key and removes it', async () => {
    await enqueue(item(1));
    const analyze = vi.fn().mockResolvedValue({ upload_id: 'u1' });
    const submit = vi.fn().mockResolvedValue({ hazard_id: 5 });
    const res = await flush({ analyze, submit });
    expect(res).toEqual({ sent: 1, dropped: 0, kept: 0 });
    expect(submit).toHaveBeenCalledWith(expect.objectContaining({ idempotencyKey: 'key-1' }), 'u1');
    expect(await list()).toEqual([]);
  });

  it('keeps items on network errors and stops trying the rest', async () => {
    await enqueue(item(1));
    await enqueue(item(2));
    const analyze = vi.fn().mockRejectedValue(new ApiError(0, 'network_error'));
    const res = await flush({ analyze, submit: vi.fn() });
    expect(res).toEqual({ sent: 0, dropped: 0, kept: 2 });
    expect(analyze).toHaveBeenCalledTimes(1);
    expect(await list()).toHaveLength(2);
  });

  it('keeps items on server errors and when signed out', async () => {
    await enqueue(item(1));
    await flush({ analyze: vi.fn().mockRejectedValue(new ApiError(503, 'server_error')), submit: vi.fn() });
    await flush({ analyze: vi.fn().mockRejectedValue(new ApiError(401, 'not_authenticated')), submit: vi.fn() });
    expect(await list()).toHaveLength(1);
  });

  it('drops an item the server rejects (4xx) and reports why', async () => {
    await enqueue(item(1));
    const onDrop = vi.fn();
    const res = await flush({
      analyze: vi.fn().mockResolvedValue({ upload_id: 'u1' }),
      submit: vi.fn().mockRejectedValue(new ApiError(422, 'too_far')),
      onDrop,
    });
    expect(res).toEqual({ sent: 0, dropped: 1, kept: 0 });
    expect(onDrop).toHaveBeenCalledWith(expect.objectContaining({ idempotencyKey: 'key-1' }), 'too_far');
    expect(await list()).toEqual([]);
  });

  it('never runs two flushes at once', async () => {
    await enqueue(item(1));
    let release;
    const analyze = vi.fn(() => new Promise((r) => { release = () => r({ upload_id: 'u' }); }));
    const submit = vi.fn().mockResolvedValue({});
    const first = flush({ analyze, submit });
    const second = await flush({ analyze, submit });
    expect(second).toEqual({ sent: 0, dropped: 0, kept: 1, busy: true });
    await vi.waitFor(() => expect(release).toBeTypeOf('function'));
    release();
    await first;
    expect(analyze).toHaveBeenCalledTimes(1);
  });
});
