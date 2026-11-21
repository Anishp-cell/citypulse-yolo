import { useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { Camera, Upload, Phone } from 'lucide-react';
import './PublicPages.css';

export default function Home() {
  const { t } = useTranslation();
  const navigate = useNavigate();

  return (
    <div className="public-page fade-in">
      <div className="pub-hero">
        <div className="pub-hero-glow" />
        <h1 className="pub-hero-title">{t('home.hero_title')}</h1>
        <p className="pub-hero-sub">{t('home.hero_subtitle')}</p>
      </div>

      <div className="pub-actions">
        <button className="pub-action-card" onClick={() => navigate('/report')}>
          <div className="pub-action-icon" style={{ background: 'var(--accent-soft)', color: 'var(--accent-hover)' }}>
            <Camera size={28} />
          </div>
          <div className="pub-action-info">
            <strong>{t('home.quick_report')}</strong>
            <span>{t('home.take_photo')}</span>
          </div>
        </button>
        <button className="pub-action-card" onClick={() => navigate('/report')}>
          <div className="pub-action-icon" style={{ background: 'var(--cyan-soft)', color: 'var(--cyan)' }}>
            <Upload size={28} />
          </div>
          <div className="pub-action-info">
            <strong>{t('home.upload_photo')}</strong>
            <span>{t('report.supported')}</span>
          </div>
        </button>
      </div>

      <div className="pub-emergency card">
        <h3>{t('home.emergency_numbers')}</h3>
        <div className="emergency-list">
          <a href="tel:108" className="emergency-item">
            <div className="emergency-icon" style={{ background: 'var(--red-soft)', color: 'var(--red)' }}>🚑</div>
            <div><strong>{t('safety.ambulance')}</strong><span>108</span></div>
          </a>
          <a href="tel:100" className="emergency-item">
            <div className="emergency-icon" style={{ background: 'var(--blue-soft)', color: 'var(--blue)' }}>🚔</div>
            <div><strong>{t('safety.police')}</strong><span>100</span></div>
          </a>
          <a href="tel:101" className="emergency-item">
            <div className="emergency-icon" style={{ background: 'var(--orange-soft)', color: 'var(--orange)' }}>🚒</div>
            <div><strong>{t('safety.fire')}</strong><span>101</span></div>
          </a>
        </div>
      </div>

      <div className="pub-info card">
        <h3>How It Works</h3>
        <div className="how-steps">
          <div className="how-step"><span className="how-num">1</span><p>Take a photo or upload an image of a road incident</p></div>
          <div className="how-step"><span className="how-num">2</span><p>Our AI analyzes severity using YOLOv8 computer vision</p></div>
          <div className="how-step"><span className="how-num">3</span><p>Get emergency guidance and alert authorities instantly</p></div>
        </div>
      </div>
    </div>
  );
}
