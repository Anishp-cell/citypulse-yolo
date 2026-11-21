import { useState, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { useAuth } from '../../hooks/useAuth';
import { api } from '../../utils/api';
import { getCurrentPosition, reverseGeocode } from '../../utils/geo';
import { getSeverityMeta } from '../../utils/severity';
import { Upload, MapPin, Crosshair, AlertCircle } from 'lucide-react';
import './PublicPages.css';

export default function Report() {
  const { t } = useTranslation();
  const { user } = useAuth();
  const fileRef = useRef();
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [location, setLocation] = useState('');
  const [coords, setCoords] = useState(null);
  const [gpsLoading, setGpsLoading] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [alertStatus, setAlertStatus] = useState('');
  const [alertSending, setAlertSending] = useState(false);

  function handleFile(f) {
    if (!f) return;
    setFile(f);
    setResult(null);
    const url = URL.createObjectURL(f);
    setPreview(url);
  }

  async function detectGPS() {
    setGpsLoading(true);
    try {
      const pos = await getCurrentPosition();
      setCoords(pos);
      const addr = await reverseGeocode(pos.lat, pos.lng);
      setLocation(addr);
    } catch (err) {
      setLocation('GPS unavailable — type manually');
    } finally {
      setGpsLoading(false);
    }
  }

  async function analyze() {
    if (!file) return;
    setAnalyzing(true);
    setResult(null);
    try {
      const formData = new FormData();
      formData.append('file', file);
      if (location) formData.append('location', location);
      if (coords) {
        formData.append('lat', coords.lat);
        formData.append('lng', coords.lng);
      }
      const isVideo = file.type.startsWith('video/');
      const endpoint = isVideo ? '/api/analyze-video' : '/api/analyze';
      const data = await api.upload(endpoint, formData);
      setResult(data);
    } catch (err) {
      alert(err.message);
    } finally {
      setAnalyzing(false);
    }
  }

  async function sendAlert() {
    if (!result?.incident_id) return;
    setAlertSending(true);
    try {
      const type = result.highest_severity?.includes('pothole') ? 'pothole' : 'accident';
      const res = await api.post('/api/notify', {
        type, location: location || 'Unknown', incident_id: result.incident_id,
      });
      setAlertStatus(res.message || t('report.alert_sent'));
    } catch (err) {
      setAlertStatus('Failed to send alert');
    } finally {
      setAlertSending(false);
    }
  }

  const severity = result ? getSeverityMeta(result.highest_severity) : null;

  return (
    <div className="public-page fade-in">
      <h2 className="pub-page-title">{t('report.title')}</h2>

      {/* Upload */}
      {!result && (
        <>
          <div className="upload-zone-pub" onClick={() => fileRef.current?.click()}>
            {preview ? (
              file?.type.startsWith('video/')
                ? <video src={preview} controls className="upload-preview" />
                : <img src={preview} alt="Preview" className="upload-preview" />
            ) : (
              <div className="upload-placeholder">
                <Upload size={36} />
                <p>{t('report.drag_drop')}</p>
                <span>{t('report.supported')}</span>
              </div>
            )}
            <input ref={fileRef} type="file" accept="image/*,video/*" hidden
              onChange={e => handleFile(e.target.files[0])} />
          </div>

          {file && <button className="btn btn-ghost btn-sm clear-btn" onClick={() => { setFile(null); setPreview(null); }}>✕ Clear</button>}

          {/* Location */}
          <div className="report-location">
            <label className="input-label"><MapPin size={14} /> {t('report.location')}</label>
            <div className="location-row">
              <input className="input" value={location} onChange={e => setLocation(e.target.value)}
                placeholder={t('report.manual_entry')} />
              <button className="btn btn-secondary btn-sm" onClick={detectGPS} disabled={gpsLoading}>
                <Crosshair size={14} />
                {gpsLoading ? '...' : 'GPS'}
              </button>
            </div>
          </div>

          <button className="btn btn-primary btn-lg analyze-btn" onClick={analyze}
            disabled={!file || analyzing || !user}>
            {analyzing
              ? <><span className="spinner" /> {t('report.analyzing')}</>
              : t('report.analyze')}
          </button>
          {!user && <p className="login-hint"><AlertCircle size={14} /> Sign in to analyze</p>}
        </>
      )}

      {/* Results */}
      {result && (
        <div className="report-results fade-in">
          <div className="result-severity-card" style={{ borderColor: severity.color }}>
            <span className="result-sev-icon">{severity.icon}</span>
            <div>
              <strong>{severity.label}</strong>
              <span className="result-sev-class">{(result.highest_severity || '').replace(/_/g, ' ')}</span>
            </div>
          </div>

          {result.image_url && <img src={result.image_url} alt="Result" className="result-image" />}

          {/* Guidance */}
          {result.guidance && result.highest_severity !== 'no_accident' && (
            <div className="result-guidance card">
              <h3>{t('report.guidance')}</h3>
              {result.guidance.llm_enhanced && (
                <div className="ai-summary"><strong>🧠 {t('report.ai_summary')}</strong><p>{result.guidance.llm_enhanced}</p></div>
              )}
              {result.guidance.immediate_actions?.length > 0 && (
                <div className="guidance-section"><h4>🚑 {t('report.immediate_actions')}</h4>
                  <ul>{result.guidance.immediate_actions.map((a,i) => <li key={i}>{a}</li>)}</ul>
                </div>
              )}
              {result.guidance.warning_signs?.length > 0 && (
                <div className="guidance-section warn"><h4>⚠️ {t('report.warning_signs')}</h4>
                  <ul>{result.guidance.warning_signs.map((a,i) => <li key={i}>{a}</li>)}</ul>
                </div>
              )}
            </div>
          )}

          {/* Alert Button */}
          {result.highest_severity !== 'no_accident' && (
            <div className="alert-section">
              <button className="btn btn-danger btn-lg alert-btn" onClick={sendAlert} disabled={alertSending}>
                {alertSending ? <><span className="spinner" /> {t('report.sending')}</> : t('report.send_alert')}
              </button>
              {alertStatus && <p className="alert-status">{alertStatus}</p>}
            </div>
          )}

          <button className="btn btn-ghost" onClick={() => { setResult(null); setFile(null); setPreview(null); }}>
            Report Another Incident
          </button>
        </div>
      )}
    </div>
  );
}
