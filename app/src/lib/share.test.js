import { afterEach, describe, expect, it, vi } from 'vitest';
import { mailtoUrl, shareComplaint, whatsappUrl } from './share';

const original = { share: navigator.share, canShare: navigator.canShare };
afterEach(() => { navigator.share = original.share; navigator.canShare = original.canShare; });

describe('share', () => {
  it('shares text plus the PDF file when the browser can share files', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(new Blob(['%PDF'], { type: 'application/pdf' })));
    navigator.canShare = vi.fn(() => true);
    navigator.share = vi.fn().mockResolvedValue(undefined);
    const r = await shareComplaint({ title: 'T', text: 'Body', pdfUrl: '/x.pdf', filename: 'c.pdf' });
    expect(r).toBe('shared');
    const arg = navigator.share.mock.calls[0][0];
    expect(arg.text).toBe('Body');
    expect(arg.files[0].name).toBe('c.pdf');
    expect(arg.files[0].type).toBe('application/pdf');
  });

  it('falls back to sharing text only when files are not shareable', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(new Blob(['%PDF'])));
    navigator.canShare = vi.fn(() => false);
    navigator.share = vi.fn().mockResolvedValue(undefined);
    expect(await shareComplaint({ title: 'T', text: 'Body', pdfUrl: '/x.pdf', filename: 'c.pdf' })).toBe('shared_text');
    expect(navigator.share.mock.calls[0][0].files).toBeUndefined();
  });

  it('reports unsupported when there is no Web Share API', async () => {
    navigator.share = undefined;
    navigator.canShare = undefined;
    expect(await shareComplaint({ title: 'T', text: 'B', pdfUrl: '/x.pdf' })).toBe('unsupported');
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
