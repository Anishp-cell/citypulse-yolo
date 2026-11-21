import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Copy, Download, ExternalLink, Mail, MessageCircle, Phone, Share2, TriangleAlert } from 'lucide-react';
import { useComplaint, useFiled } from '../api/hooks';
import { errorMessage } from '../lib/format';
import { copyText, mailtoUrl, shareComplaint, whatsappUrl } from '../lib/share';
import { toast } from './Toaster';
import { ErrorState, Notice, Sheet, Spinner } from './ui';

export function FiledForm({ hazardId, defaultPortal, onSaved }) {
  const { t } = useTranslation();
  const filed = useFiled(hazardId);
  const [portal, setPortal] = useState(defaultPortal || '');
  const [ticket, setTicket] = useState('');
  const submit = async (e) => {
    e.preventDefault();
    try {
      await filed.mutateAsync({ portal: portal.trim(), ticket_id: ticket.trim() });
      toast(t('complaint.filed_saved'));
      setTicket('');
      onSaved?.();
    } catch (err) {
      toast(errorMessage(t, err));
    }
  };
  return (
    <form className="stack" onSubmit={submit}>
      <p className="small muted">{t('complaint.filed_q')}</p>
      <div className="field">
        <label htmlFor="filed-portal">{t('complaint.filed_portal')}</label>
        <input id="filed-portal" className="input" value={portal} maxLength={120} required
          onChange={(e) => setPortal(e.target.value)} />
      </div>
      <div className="field">
        <label htmlFor="filed-ticket">{t('complaint.filed_ticket')}</label>
        <input id="filed-ticket" className="input" value={ticket} maxLength={80} required inputMode="text"
          autoComplete="off" onChange={(e) => setTicket(e.target.value)} />
      </div>
      <button className="btn btn-primary" disabled={filed.isPending || !ticket.trim() || !portal.trim()}>
        {filed.isPending ? <Spinner /> : t('complaint.filed_save')}
      </button>
    </form>
  );
}

export default function ComplaintSheet({ hazardId, open, onClose, canFile = true }) {
  const { t, i18n } = useTranslation();
  const complaint = useComplaint(hazardId, i18n.language, open);
  const c = complaint.data;

  const copy = async (text) => toast((await copyText(text)) ? t('common.copied') : t('errors.server_error'));
  const shareAll = async () => {
    const r = await shareComplaint({ title: c.subject, text: `${c.subject}\n\n${c.body}`, pdfUrl: c.pdf_url,
      filename: `CityPulse-complaint-${hazardId}.pdf` });
    if (r === 'unsupported') copy(`${c.subject}\n\n${c.body}`);
  };

  return (
    <Sheet open={open} onClose={onClose} title={c ? t('complaint.title', { authority: c.authority }) : t('hazard.send_to_city')}>
      {complaint.isPending && <div className="center"><Spinner /></div>}
      {complaint.isError && <ErrorState error={complaint.error} onRetry={complaint.refetch} />}
      {c && (
        <div className="stack">
          <p className="muted small">{t('complaint.why')}</p>
          {!c.contacts_verified && c.portal_url && <Notice tone="warn" icon={TriangleAlert}>{t('complaint.unverified')}</Notice>}
          {!c.portal_url && !c.email_to && <Notice tone="info">{t('complaint.no_portal')}</Notice>}

          <button className="btn btn-primary btn-lg btn-block" onClick={shareAll}><Share2 size={20} />{t('complaint.share_all')}</button>
          <div className="choice-grid">
            <a className="btn btn-outline" href={whatsappUrl(c.whatsapp_text, c.whatsapp_number)} target="_blank" rel="noreferrer">
              <MessageCircle size={18} />{c.whatsapp_number ? t('complaint.whatsapp_city') : t('complaint.whatsapp')}
            </a>
            <a className="btn btn-outline" href={mailtoUrl(c.email_to, c.subject, c.body)}><Mail size={18} />{t('complaint.email')}</a>
            <a className="btn btn-outline" href={c.pdf_url} target="_blank" rel="noreferrer"><Download size={18} />{t('complaint.download_pdf')}</a>
            <button className="btn btn-outline" onClick={() => copy(`${c.subject}\n\n${c.body}`)}><Copy size={18} />{t('complaint.copy_text')}</button>
          </div>
          {c.portal_url && (
            <div className="stack-sm">
              <a className="btn btn-outline btn-block" href={c.portal_url} target="_blank" rel="noreferrer">
                <ExternalLink size={18} />{t('complaint.open_portal', { portal: c.portal_name })}
              </a>
              <span className="tiny center">{t('complaint.portal_hint')}</span>
            </div>
          )}
          {c.helpline && (
            <a className="btn btn-ghost btn-block" href={`tel:${c.helpline}`}><Phone size={18} />{t('complaint.call_helpline', { number: c.helpline })}</a>
          )}

          {canFile && <><hr className="divider" /><FiledForm hazardId={hazardId} defaultPortal={c.portal_name} onSaved={onClose} /></>}

          {c.escalation && (
            <details className="card card-flat">
              <summary><strong>{t('complaint.escalate_title')}</strong></summary>
              <div className="stack" style={{ marginTop: 10 }}>
                <p className="small muted">{t('complaint.escalate_desc', { name: c.escalation.name })}</p>
                <button className="btn btn-outline" onClick={() => copy(c.escalation.body)}><Copy size={18} />{t('complaint.copy_escalation')}</button>
                <a className="btn btn-outline" href={c.escalation.url} target="_blank" rel="noreferrer">
                  <ExternalLink size={18} />{t('complaint.escalate_open', { name: c.escalation.name })}
                </a>
              </div>
            </details>
          )}
        </div>
      )}
    </Sheet>
  );
}
