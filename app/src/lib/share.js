// Sharing a complaint: native share sheet with the PDF where supported (Android Chrome, iOS Safari),
// otherwise WhatsApp / email links and copy-to-clipboard.

/** Fetch the complaint PDF ahead of time (when the sheet opens); returns a File or null. */
export async function prefetchPdf(pdfUrl, filename = 'complaint.pdf') {
  try {
    const res = await fetch(pdfUrl, { credentials: 'include' });
    if (!res.ok) return null;
    return new File([await res.blob()], filename, { type: 'application/pdf' });
  } catch {
    return null;
  }
}

/**
 * Must be called directly from a click handler with an already-fetched `file`: Safari only allows
 * navigator.share() inside the user gesture, so nothing may be awaited before it.
 */
export async function shareComplaint({ title, text, file }) {
  if (typeof navigator.share !== 'function') return 'unsupported';
  try {
    if (file && typeof navigator.canShare === 'function' && navigator.canShare({ files: [file] })) {
      await navigator.share({ title, text, files: [file] });
      return 'shared';
    }
    await navigator.share({ title, text });
    return 'shared_text';
  } catch (e) {
    if (e?.name === 'AbortError') return 'cancelled';
    return 'unsupported';
  }
}

export function whatsappUrl(text, number) {
  const digits = (number || '').replace(/\D/g, '');
  return `https://wa.me/${digits}?text=${encodeURIComponent(text)}`;
}

export function mailtoUrl(to, subject, body) {
  return `mailto:${to || ''}?subject=${encodeURIComponent(subject)}&body=${encodeURIComponent(body)}`;
}

export async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.setAttribute('readonly', '');
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    const ok = document.execCommand?.('copy') ?? false;
    ta.remove();
    return ok;
  }
}

export async function sharePage({ title, text, url }) {
  if (typeof navigator.share === 'function') {
    try {
      await navigator.share({ title, text, url });
      return 'shared';
    } catch (e) {
      if (e?.name === 'AbortError') return 'cancelled';
    }
  }
  return (await copyText(`${text} ${url}`)) ? 'copied' : 'unsupported';
}
