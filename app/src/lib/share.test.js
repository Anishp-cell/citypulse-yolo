import { afterEach, describe, expect, it, vi } from 'vitest';
import { mailtoUrl, prefetchPdf, shareComplaint, whatsappUrl } from './share';

const original = { share: navigator.share, canShare: navigator.canShare };
afterEach(() => { navigator.share = original.share; navigator.canShare = original.canShare; });

describe('share', () => {
  it('shares text plus a pre-fetched PDF without awaiting the network inside the tap', async () => {
    // Review I7: Safari requires navigator.share() to run directly in the user gesture.
    const fetchSpy = vi.spyOn(globalThis, 'fetch');
    navigator.canShare = vi.fn(() => true);
    navigator.share = vi.fn().mockResolvedValue(undefined);
    const file = new File(['%PDF'], 'c.pdf', { type: 'application/pdf' });
    const r = await shareComplaint({ title: 'T', text: 'Body', file });
    expect(r).toBe('shared');
    expect(fetchSpy).not.toHaveBeenCalled();
    const arg = navigator.share.mock.calls[0][0];
    expect(arg.text).toBe('Body');
    expect(arg.files[0].name).toBe('c.pdf');
  });

  it('prefetchPdf returns a named PDF File', async () => {
    // A plain object, not new Response(): Node's Response cannot read jsdom's Blob ("object.stream is not a function").
    vi.spyOn(globalThis, 'fetch').mockResolvedValue({ ok: true, blob: async () => new Blob(['%PDF'], { type: 'application/pdf' }) });
    const f = await prefetchPdf('/x.pdf', 'c.pdf');
    expect(f.name).toBe('c.pdf');
    expect(f.type).toBe('application/pdf');
  });

  it('falls back to sharing text only when files are not shareable', async () => {
    navigator.canShare = vi.fn(() => false);
    navigator.share = vi.fn().mockResolvedValue(undefined);
    const file = new File(['%PDF'], 'c.pdf', { type: 'application/pdf' });
    expect(await shareComplaint({ title: 'T', text: 'Body', file })).toBe('shared_text');
    expect(navigator.share.mock.calls[0][0].files).toBeUndefined();
  });

  it('reports unsupported when there is no Web Share API', async () => {
    navigator.share = undefined;
    navigator.canShare = undefined;
    expect(await shareComplaint({ title: 'T', text: 'B' })).toBe('unsupported');
  });

  it('treats the user closing the share sheet as cancelled, not an error', async () => {
    navigator.canShare = undefined;
    navigator.share = vi.fn().mockRejectedValue(Object.assign(new Error('x'), { name: 'AbortError' }));
    expect(await shareComplaint({ title: 'T', text: 'B' })).toBe('cancelled');
  });

  it('builds WhatsApp and mailto links with encoded text', () => {
    expect(whatsappUrl('a b&c')).toBe('https://wa.me/?text=a%20b%26c');
    expect(whatsappUrl('hi', '+91 96899 00002')).toBe('https://wa.me/919689900002?text=hi');
    expect(mailtoUrl('x@y.in', 'Sub ject', 'Line1\nLine2'))
      .toBe('mailto:x@y.in?subject=Sub%20ject&body=Line1%0ALine2');
  });
});
