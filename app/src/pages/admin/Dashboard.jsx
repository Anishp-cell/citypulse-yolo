import { createElement, useMemo } from 'react';
import { Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { Activity, ArrowDownRight, ArrowRight, ArrowUpRight, CircleCheck, Clock, Inbox, Minus, Wrench } from 'lucide-react';
import { useAdminQueue, useAdminStats } from '../../api/hooks';
import { ErrorState } from '../../components/ui';
import MapView from '../../components/MapView';
import { DEFAULT_CENTER } from '../../lib/geo';
import { placeName } from '../../lib/format';
import { typeIcon } from '../../lib/status';
import { Heat, Kpi, PageHeader, Panel, Pipeline, PriorityMeter, PulseChart, SkeletonPanel, TypeBreakdown } from './parts';

const sum = (rows) => rows.reduce((s, r) => s + r.count, 0);

function Trend({ delta }) {
  const { t } = useTranslation();
  if (delta === 0) return <span className="trend"><Minus size={14} aria-hidden="true" />{t('admin.no_change')}</span>;
  const Icon = delta > 0 ? ArrowUpRight : ArrowDownRight;
  return (
    <span className="trend">
      <Icon size={14} aria-hidden="true" />{t('admin.vs_prev_week', { delta: `${delta > 0 ? '+' : '−'}${Math.abs(delta)}` })}
    </span>
  );
}

function NeedsAttention({ pending }) {
  const { t } = useTranslation();
  const queue = useAdminQueue({ sort: 'priority', limit: '5', offset: '0' });
  const hazards = queue.data?.hazards || [];
  return (
    <Panel title={t('admin.needs_attention')} description={t('admin.needs_attention_desc')} bodyClass={null}
      action={<Link className="panel-link" to="/admin/queue">{t('admin.view_queue')}<ArrowRight size={14} aria-hidden="true" /></Link>}>
      <div style={{ height: 12 }} />
      {pending > 0 && (
        <Link className="attn-banner" to="/admin/moderation">
          <Inbox size={16} aria-hidden="true" />
          <span>{t('admin.review_banner', { count: pending })}</span>
          {t('admin.review_now')}<ArrowRight size={14} aria-hidden="true" />
        </Link>
      )}
      {queue.isPending && <div className="panel-body"><div className="skeleton" style={{ height: 180 }} /></div>}
      {!queue.isPending && hazards.length === 0 && (
        <div className="empty-panel"><CircleCheck size={28} aria-hidden="true" /><strong>{t('admin.all_clear')}</strong></div>
      )}
      {hazards.length > 0 && (
        <ol className="rank-list">
          {hazards.map((h) => (
            <li key={h.id}>
              <Link to={`/h/${h.id}`}>
                {h.thumb_url ? <img className="rank-thumb" src={h.thumb_url} alt="" loading="lazy" />
                  : <span className="rank-thumb">{createElement(typeIcon(h.type), { size: 18, 'aria-hidden': true })}</span>}
                <span style={{ minWidth: 0 }}>
                  <span className="rank-title" style={{ display: 'block' }}>{t(`types.${h.type}`)} · {placeName(h)}</span>
                  <span className="rank-meta" style={{ display: 'block' }}>
                    <span className="mono">#{h.id}</span>{h.ward ? ` · ${h.ward}` : ''} · {t('common.days', { count: h.days_open })}
                  </span>
                </span>
                <PriorityMeter score={h.priority_score} />
              </Link>
            </li>
          ))}
        </ol>
      )}
    </Panel>
  );
}

function Hotspots({ heat }) {
  const { t } = useTranslation();
  const center = useMemo(() => (heat.length
    ? { lat: heat.reduce((s, p) => s + p.lat, 0) / heat.length, lng: heat.reduce((s, p) => s + p.lng, 0) / heat.length }
    : DEFAULT_CENTER), [heat]);
  return (
    <Panel title={t('admin.hotspots')} description={t('admin.hotspots_desc')} bodyClass={null}
      action={<Link className="panel-link" to="/admin/map">{t('admin.open_map')}<ArrowRight size={14} aria-hidden="true" /></Link>}>
      <div className="mini-map"><MapView center={center} zoom={12} hazards={[]}><Heat points={heat} /></MapView></div>
    </Panel>
  );
}

export default function Dashboard() {
  const { t } = useTranslation();
  const { data, isPending, error, refetch } = useAdminStats();
  if (error) return <ErrorState error={error} onRetry={refetch} />;

  const header = <PageHeader title={t('admin.dashboard')} description={t('admin.dash_subtitle')} />;
  if (isPending) {
    return (
      <>
        {header}
        <div className="kpis">{[0, 1, 2, 3].map((i) => <SkeletonPanel key={i} height={56} />)}</div>
        <div className="dash-grid"><SkeletonPanel height={190} /><SkeletonPanel lines={5} /></div>
      </>
    );
  }

  const daily = data.daily_reports;
  const last7 = sum(daily.slice(-7));
  const prev7 = sum(daily.slice(-14, -7));
  const total30 = sum(daily);
  const pending = data.by_status.pending_review || 0;

  return (
    <>
      {header}
      <div className="kpis">
        <Kpi icon={Activity} label={t('admin.active')} value={data.totals.active} foot={t('admin.active_desc')} />
        <Kpi as={Link} to="/admin/moderation" icon={Inbox} label={t('admin.pending')} value={pending} attn={pending > 0}
          foot={pending > 0 ? <>{t('admin.review_now')}<ArrowRight size={13} aria-hidden="true" /></> : t('admin.nothing_to_review')} />
        <Kpi icon={Clock} label={t('admin.reports_7d')} value={last7} foot={<Trend delta={last7 - prev7} />} />
        <Kpi icon={Wrench} label={t('admin.avg_fix')} value={data.totals.avg_days_to_fix ?? null}
          foot={data.totals.fixed ? t('admin.fixed_count', { count: data.totals.fixed }) : t('admin.no_fixes_yet')} />
      </div>

      <div className="dash-grid">
        <Panel title={t('admin.pulse_title')} description={t('admin.pulse_desc')}>
          <div className="pulse-summary" style={{ marginBottom: 4 }}>
            <strong className="tabular">{total30}</strong>
            <span className="muted small">{t('admin.pulse_summary', { count: total30 })}</span>
          </div>
          <PulseChart data={daily} label={t('admin.pulse_aria', { count: total30, last7 })} />
        </Panel>
        <NeedsAttention pending={pending} />

        <Panel className="span-2" title={t('admin.pipeline')} description={t('admin.pipeline_desc')}>
          <Pipeline byStatus={data.by_status} />
        </Panel>

        <Hotspots heat={data.heat} />
        <Panel title={t('admin.by_type')} description={t('admin.by_type_desc')}>
          <TypeBreakdown byType={data.by_type} />
        </Panel>
      </div>
    </>
  );
}
