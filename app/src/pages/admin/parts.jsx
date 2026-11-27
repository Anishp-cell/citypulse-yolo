// Building blocks for the officer console: panels, KPIs, the Pulse line, the hazard pipeline.
import { createElement, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import L from 'leaflet';
import 'leaflet.heat';
import { useMap } from 'react-leaflet';
import { statusMeta, typeIcon } from '../../lib/status';
import { formatDate } from '../../lib/format';
import { PIPELINE, useCountUp } from './consoleUtils';

export function PageHeader({ title, description, children }) {
  return (
    <header className="page-head">
      <div>
        <h1>{title}</h1>
        {description && <p>{description}</p>}
      </div>
      {children && <div className="toolbar">{children}</div>}
    </header>
  );
}

export function Panel({ title, description, action, className = '', children, bodyClass = 'panel-body' }) {
  return (
    <section className={`panel ${className}`}>
      {(title || action) && (
        <div className="panel-head">
          <div>
            {title && <h2>{title}</h2>}
            {description && <p>{description}</p>}
          </div>
          {action}
        </div>
      )}
      {bodyClass ? <div className={bodyClass}>{children}</div> : children}
    </section>
  );
}

export function Kpi({ icon, label, value, foot, attn = false, as: As = 'div', ...rest }) {
  const shown = useCountUp(value);
  const text = typeof value !== 'number' ? '—'
    : Number.isInteger(value) ? Math.round(shown).toLocaleString() : shown.toFixed(1);
  return (
    <As className={`panel kpi${attn ? ' attn' : ''}`} {...rest}>
      <span className="kpi-label">{icon && createElement(icon, { size: 16, 'aria-hidden': true })}{label}</span>
      <span className="kpi-value">{text}</span>
      {foot && <span className="kpi-foot">{foot}</span>}
    </As>
  );
}

function useWidth(ref) {
  const [width, setWidth] = useState(0);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return undefined;
    setWidth(el.clientWidth);
    const ro = new ResizeObserver(([e]) => setWidth(e.contentRect.width));
    ro.observe(el);
    return () => ro.disconnect();
  }, [ref]);
  return width;
}

/** A round axis maximum whose midpoint is a whole number (4, 6, 8, 10, 20, 40, 100…). */
function niceMax(n) {
  if (n <= 4) return 4;
  if (n <= 10) return Math.ceil(n / 2) * 2;
  const step = 10 ** Math.floor(Math.log10(n)) / 5;
  return Math.ceil(n / (step * 2)) * step * 2;
}

/** The signature chart: 30 days of reports as one pulse line, today's point beating. */
export function PulseChart({ data, label }) {
  const { i18n } = useTranslation();
  const ref = useRef(null);
  const width = useWidth(ref);
  const [hover, setHover] = useState(null);
  const H = 190, top = 30, bottom = 24, left = 4, right = 34;
  const max = niceMax(Math.max(0, ...data.map((d) => d.count)));
  const geo = useMemo(() => {
    if (!width || data.length < 2) return null;
    const w = width - left - right, h = H - top - bottom;
    const pts = data.map((d, i) => [left + (i / (data.length - 1)) * w, top + h - (d.count / max) * h]);
    const line = pts.map(([x, y], i) => `${i ? 'L' : 'M'}${x.toFixed(1)},${y.toFixed(1)}`).join('');
    return { pts, line, area: `${line}L${pts.at(-1)[0]},${top + h}L${pts[0][0]},${top + h}Z`, w, h };
  }, [width, data, max]);
  const pick = (clientX) => {
    if (!geo) return;
    const x = clientX - ref.current.getBoundingClientRect().left;
    setHover(Math.max(0, Math.min(data.length - 1, Math.round(((x - left) / geo.w) * (data.length - 1)))));
  };
  const onKey = (e) => {
    if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
      e.preventDefault();
      const cur = hover ?? data.length - 1;
      setHover(Math.max(0, Math.min(data.length - 1, cur + (e.key === 'ArrowLeft' ? -1 : 1))));
    }
  };
  const day = (iso) => formatDate(iso, i18n.language, { day: 'numeric', month: 'short' });
  const last = geo?.pts.at(-1);
  const active = hover != null && geo ? { p: geo.pts[hover], d: data[hover] } : null;
  return (
    <div className="pulse-wrap" ref={ref}>
      {geo && (
        <svg role="img" aria-label={label} tabIndex={0} onKeyDown={onKey}
          onPointerMove={(e) => pick(e.clientX)} onPointerLeave={() => setHover(null)} onBlur={() => setHover(null)}>
          <defs>
            <linearGradient id="pulse-fill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="var(--c-brand)" stopOpacity="0.16" />
              <stop offset="100%" stopColor="var(--c-brand)" stopOpacity="0" />
            </linearGradient>
          </defs>
          <g className="pulse-grid" aria-hidden="true">
            {[0, 0.5, 1].map((f) => <line key={f} x1={left} x2={left + geo.w} y1={top + geo.h * (1 - f)} y2={top + geo.h * (1 - f)} />)}
          </g>
          <g className="pulse-axis" aria-hidden="true">
            {[0, 0.5, 1].map((f) => (
              <text key={f} x={left + geo.w + 8} y={top + geo.h * (1 - f) + 4}>{Math.round(max * f)}</text>
            ))}
            {[0, Math.floor((data.length - 1) / 2), data.length - 1].map((i, k) => (
              <text key={i} x={geo.pts[i][0]} y={H - 6} textAnchor={['start', 'middle', 'end'][k]}>{day(data[i].date)}</text>
            ))}
          </g>
          <path className="pulse-area" d={geo.area} />
          <path className="pulse-line pulse-draw" d={geo.line} pathLength="1" />
          {active && <line className="pulse-cursor" x1={active.p[0]} x2={active.p[0]} y1={top - 6} y2={top + geo.h} />}
          {!active && <circle className="pulse-ring" cx={last[0]} cy={last[1]} r="4" />}
          <circle className="pulse-now" cx={(active?.p || last)[0]} cy={(active?.p || last)[1]} r="4" />
        </svg>
      )}
      {active && (
        <div className="pulse-tip" style={{ left: Math.min(Math.max(active.p[0], 60), width - 60) }} aria-live="polite">
          {day(active.d.date)} · {active.d.count}
        </div>
      )}
      <table className="sr-only">
        <tbody>{data.map((d) => <tr key={d.date}><th>{day(d.date)}</th><td>{d.count}</td></tr>)}</tbody>
      </table>
    </div>
  );
}

/** Every hazard's place in the lifecycle, in lifecycle order, as one segmented bar plus a labelled legend. */
export function Pipeline({ byStatus }) {
  const { t } = useTranslation();
  const total = PIPELINE.reduce((s, k) => s + (byStatus[k] || 0), 0);
  return (
    <div>
      <div className="pipeline-bar" role="img"
        aria-label={PIPELINE.map((k) => `${t(`status.${k}`)}: ${byStatus[k] || 0}`).join(', ')}>
        {total > 0 && PIPELINE.filter((k) => byStatus[k]).map((k, i) => (
          <span key={k} title={`${t(`status.${k}`)}: ${byStatus[k]}`}
            style={{ flexGrow: byStatus[k], background: statusMeta(k).color, animationDelay: `${i * 50}ms` }} />
        ))}
      </div>
      <div className="pipeline-legend" aria-hidden="true">
        {PIPELINE.map((k) => (
          <div key={k} className={byStatus[k] ? '' : 'zero'}>
            <i className="dot" style={{ background: statusMeta(k).color }} />
            <span className="ellipsis">{t(`status.${k}`)}</span>
            <strong className="tabular">{byStatus[k] || 0}</strong>
          </div>
        ))}
      </div>
    </div>
  );
}

export function PriorityMeter({ score }) {
  const value = Math.round(score || 0);
  return (
    <span className={`prio${value >= 60 ? ' high' : ''}`}>
      <span className="prio-track" aria-hidden="true"><span style={{ width: `${Math.min(100, value)}%` }} /></span>
      <span className="tabular">{value}</span>
    </span>
  );
}

export function TypeBreakdown({ byType }) {
  const { t } = useTranslation();
  const rows = Object.entries(byType).sort((a, b) => b[1] - a[1]);
  const max = Math.max(1, ...rows.map(([, n]) => n));
  return (
    <div className="type-rows">
      {rows.map(([type, n], i) => (
        <div key={type} className="type-row">
          <span className="label">{createElement(typeIcon(type), { size: 15, 'aria-hidden': true })}<span className="ellipsis">{t(`types.${type}`)}</span></span>
          <span className="bar" aria-hidden="true"><span style={{ width: `${(n / max) * 100}%`, animationDelay: `${i * 60}ms` }} /></span>
          <strong className="tabular">{n}</strong>
        </div>
      ))}
    </div>
  );
}

export function Heat({ points }) {
  const map = useMap();
  useEffect(() => {
    const layer = L.heatLayer(points.map((p) => [p.lat, p.lng, Math.max(0.2, p.weight)]), { radius: 25, blur: 18 });
    layer.addTo(map);
    return () => { map.removeLayer(layer); };
  }, [map, points]);
  return null;
}

export function SkeletonPanel({ height = 120, lines = 0 }) {
  return (
    <div className="panel panel-body" aria-hidden="true">
      <div className="skeleton sk-line" style={{ width: '35%', marginBottom: 14 }} />
      {lines > 0 ? Array.from({ length: lines }, (_, i) => (
        <div key={i} className="skeleton sk-line" style={{ width: `${90 - i * 12}%`, marginTop: 10 }} />
      )) : <div className="skeleton" style={{ height }} />}
    </div>
  );
}
