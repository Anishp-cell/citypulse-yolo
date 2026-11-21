// Reports made without a connection are kept in IndexedDB and sent later, in order,
// with the idempotency key they were created with (so a retry never creates a duplicate).
import { get, set } from 'idb-keyval';

const KEY = 'cp_offline_reports_v1';
let flushing = false;

async function read() {
  return (await get(KEY)) || [];
}

export async function list() {
  return read();
}

export async function enqueue(item) {
  const items = await read();
  const entry = { ...item, id: crypto.randomUUID(), createdAt: new Date().toISOString() };
  items.push(entry);
  await set(KEY, items);
  window.dispatchEvent(new CustomEvent('cp:queue-changed'));
  return entry;
}

export async function remove(id) {
  await set(KEY, (await read()).filter((i) => i.id !== id));
  window.dispatchEvent(new CustomEvent('cp:queue-changed'));
}

const keepTrying = (err) => !err?.status || err.status >= 500 || err.status === 401 || err.status === 429;

/**
 * @param {{analyze: (item) => Promise<{upload_id: string}>, submit: (item, uploadId: string) => Promise<any>,
 *          onDrop?: (item, code: string) => void}} handlers
 */
export async function flush({ analyze, submit, onDrop }) {
  if (flushing) return { sent: 0, dropped: 0, kept: (await read()).length, busy: true };
  flushing = true;
  const result = { sent: 0, dropped: 0, kept: 0 };
  try {
    const items = await read();
    for (let i = 0; i < items.length; i++) {
      const item = items[i];
      try {
        const { upload_id: uploadId } = await analyze(item);
        await submit(item, uploadId);
        await remove(item.id);
        result.sent++;
      } catch (err) {
        if (keepTrying(err)) { // still offline / server down: try again later, keep the order
          result.kept = items.length - i;
          break;
        }
        await remove(item.id);
        result.dropped++;
        onDrop?.(item, err?.code || 'server_error');
      }
    }
  } finally {
    flushing = false;
  }
  return result;
}
