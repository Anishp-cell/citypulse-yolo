import { useTranslation } from 'react-i18next';
import { Phone, Heart, Shield } from 'lucide-react';
import './PublicPages.css';

export default function Safety() {
  const { t } = useTranslation();
  return (
    <div className="public-page fade-in">
      <h2 className="pub-page-title">{t('safety.title')}</h2>

      <div className="card safety-card">
        <h3><Phone size={18} /> {t('safety.emergency_title')}</h3>
        <div className="emergency-grid">
          <a href="tel:108" className="emer-card emer-red"><span className="emer-num">108</span><span>{t('safety.ambulance')}</span></a>
          <a href="tel:100" className="emer-card emer-blue"><span className="emer-num">100</span><span>{t('safety.police')}</span></a>
          <a href="tel:101" className="emer-card emer-orange"><span className="emer-num">101</span><span>{t('safety.fire')}</span></a>
          <a href="tel:112" className="emer-card emer-purple"><span className="emer-num">112</span><span>Universal</span></a>
        </div>
      </div>

      <div className="card safety-card">
        <h3><Heart size={18} /> {t('safety.first_aid_title')}</h3>
        <div className="aid-list">
          <div className="aid-item"><strong>🩸 Bleeding</strong><p>Apply firm pressure with a clean cloth. Elevate the wound above the heart if possible.</p></div>
          <div className="aid-item"><strong>💨 Not Breathing</strong><p>Check airway, tilt head back. If no breathing, begin CPR: 30 chest compressions, 2 rescue breaths.</p></div>
          <div className="aid-item"><strong>🦴 Fracture</strong><p>Do NOT move the injured limb. Immobilize with a splint. Call 108 for ambulance.</p></div>
          <div className="aid-item"><strong>🔥 Burns</strong><p>Cool under running water for 10+ minutes. Do NOT apply ice, butter, or toothpaste.</p></div>
          <div className="aid-item"><strong>🧠 Head Injury</strong><p>Keep person still. Do NOT remove helmet. Watch for vomiting, confusion, or unequal pupils.</p></div>
        </div>
      </div>

      <div className="card safety-card">
        <h3><Shield size={18} /> {t('safety.tips_title')}</h3>
        <ul className="tips-list">
          <li>Always wear seatbelts and helmets</li>
          <li>Never use phone while driving</li>
          <li>Follow speed limits — especially in school zones</li>
          <li>Keep safe distance from the vehicle ahead</li>
          <li>Use indicators before turning or changing lanes</li>
          <li>Check mirrors regularly</li>
          <li>Avoid driving when tired or under influence</li>
          <li>Report potholes and road damage to keep roads safe</li>
        </ul>
      </div>
    </div>
  );
}
