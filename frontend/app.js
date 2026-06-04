/* ===== CityPulse Frontend — app.js ===== */

const API_BASE = ''; // same origin

// DOM refs
const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const previewArea = document.getElementById('previewArea');
const previewImage = document.getElementById('previewImage');
const videoPreviewArea = document.getElementById('videoPreviewArea');
const previewVideo = document.getElementById('previewVideo');
const clearBtn = document.getElementById('clearBtn');
const videoClearBtn = document.getElementById('videoClearBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const resultsSection = document.getElementById('results');

let selectedFile = null;
let currentMode = 'image'; // 'image' or 'video'
let currentIncidentId = null;

// ───────── Auth State ─────────
let authToken = localStorage.getItem('cp_token') || null;

function getAuthHeaders() {
    return authToken ? { 'Authorization': `Bearer ${authToken}` } : {};
}

// ───────── Mode Toggle ─────────

document.querySelectorAll('.mode-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentMode = btn.dataset.mode;
        resetUpload();
    });
});

// ───────── Upload Handling ─────────

uploadZone.addEventListener('click', () => fileInput.click());

uploadZone.addEventListener('dragover', e => {
    e.preventDefault();
    uploadZone.classList.add('drag-over');
});
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
uploadZone.addEventListener('drop', e => {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});

fileInput.addEventListener('change', () => {
    if (fileInput.files.length) handleFile(fileInput.files[0]);
});

clearBtn.addEventListener('click', resetUpload);
videoClearBtn.addEventListener('click', resetUpload);

function handleFile(file) {
    const isVideo = file.type.startsWith('video/');
    const isImage = file.type.startsWith('image/');

    // Auto-detect mode from file type
    if (isVideo) {
        currentMode = 'video';
        document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
        document.getElementById('modeVideoBtn').classList.add('active');
    } else if (isImage) {
        currentMode = 'image';
        document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
        document.getElementById('modeImageBtn').classList.add('active');
    } else {
        return; // unsupported
    }

    selectedFile = file;
    uploadZone.style.display = 'none';

    if (isImage) {
        const reader = new FileReader();
        reader.onload = e => {
            previewImage.src = e.target.result;
            previewArea.style.display = 'block';
            videoPreviewArea.style.display = 'none';
        };
        reader.readAsDataURL(file);
    } else {
        const url = URL.createObjectURL(file);
        previewVideo.src = url;
        videoPreviewArea.style.display = 'block';
        previewArea.style.display = 'none';
    }

    analyzeBtn.disabled = false;
    document.getElementById('analyzeBtnText').textContent =
        isVideo ? 'Analyze Video' : 'Analyze Image';
}

function resetUpload() {
    selectedFile = null;
    fileInput.value = '';
    previewImage.src = '';
    previewVideo.src = '';
    previewArea.style.display = 'none';
    videoPreviewArea.style.display = 'none';
    uploadZone.style.display = 'block';
    analyzeBtn.disabled = true;
    resultsSection.style.display = 'none';
    document.getElementById('analyzeBtnText').textContent = 'Analyze';
}

// ───────── Analyze ─────────

analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    const isVideo = currentMode === 'video';
    const endpoint = isVideo ? '/api/analyze-video' : '/api/analyze';
    const loaderText = document.getElementById('analyzeLoaderText');
    loaderText.textContent = isVideo ? 'Analyzing video… this may take a moment' : 'Analyzing…';

    setBtnLoading(analyzeBtn, true);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        const res = await fetch(`${API_BASE}${endpoint}`, {
            method: 'POST',
            headers: getAuthHeaders(),
            body: formData
        });
        if (res.status === 401) { showAuthModal(); throw new Error('Please sign in to continue.'); }
        if (!res.ok) throw new Error(`Server error ${res.status}`);
        const data = await res.json();
        currentIncidentId = data.incident_id || null;
        renderResults(data, isVideo);
    } catch (err) {
        alert('Analysis failed: ' + err.message);
    } finally {
        setBtnLoading(analyzeBtn, false);
    }
});

// ───────── Render Results ─────────

const SEVERITY_MAP = {
    no_accident: { icon: '✅', label: 'No Incident Detected', cls: 'sev-none', dot: '#22c55e' },
    minor_accident: { icon: '⚠️', label: 'Minor Accident', cls: 'sev-minor', dot: '#f59e0b' },
    moderate_accident: { icon: '🔶', label: 'Moderate Accident', cls: 'sev-moderate', dot: '#f97316' },
    severe_accident: { icon: '🔴', label: 'Severe Accident', cls: 'sev-severe', dot: '#ef4444' },
    totaled_vehicle: { icon: '🚨', label: 'Critical — Totaled', cls: 'sev-critical', dot: '#dc2626' },
    pothole: { icon: '🕳️', label: 'Pothole Detected', cls: 'sev-pothole', dot: '#06b6d4' },
};

function renderResults(data, isVideo = false) {
    const severity = data.highest_severity || 'no_accident';
    const meta = SEVERITY_MAP[severity] || SEVERITY_MAP.no_accident;

    // Severity banner
    const banner = document.getElementById('severityBanner');
    banner.className = 'severity-banner ' + meta.cls;
    document.getElementById('severityIcon').textContent = meta.icon;
    document.getElementById('severityLabel').textContent = meta.label;
    document.getElementById('severityClass').textContent = severity.replace(/_/g, ' ');

    // Update image title
    document.getElementById('resultImageTitle').textContent =
        isVideo ? 'Worst-Severity Keyframe' : 'Annotated Image';

    // Annotated image / keyframe
    document.getElementById('resultImage').src = data.image_url;

    // Video stats
    const videoStats = document.getElementById('videoStats');
    if (isVideo && data.video_info) {
        const vi = data.video_info;
        document.getElementById('statDuration').textContent = vi.duration + 's';
        document.getElementById('statFrames').textContent = vi.frames_analyzed;
        document.getElementById('statIncidents').textContent = vi.total_incidents;
        document.getElementById('statFps').textContent = vi.fps;
        videoStats.style.display = 'flex';
    } else {
        videoStats.style.display = 'none';
    }

    // Timeline
    const timelineCard = document.getElementById('timelineCard');
    const timelineContent = document.getElementById('timelineContent');
    if (isVideo && data.timeline && data.timeline.length > 0) {
        timelineContent.innerHTML = '';
        data.timeline.forEach(t => {
            const tmeta = SEVERITY_MAP[t.severity] || { dot: '#64748b', label: t.severity };
            const el = document.createElement('div');
            el.className = 'timeline-item fade-up';
            el.innerHTML = `
                <span class="timeline-time">${t.timestamp}s</span>
                <span class="timeline-dot" style="background:${tmeta.dot}"></span>
                <span class="timeline-info">
                    <strong>${(tmeta.label || t.severity).replace(/_/g, ' ')}</strong>
                    <span class="timeline-count">${t.count} detection${t.count > 1 ? 's' : ''}</span>
                </span>
            `;
            timelineContent.appendChild(el);
        });
        timelineCard.style.display = 'block';
    } else {
        timelineCard.style.display = 'none';
    }

    // Detections list
    const list = document.getElementById('detectionsList');
    list.innerHTML = '';

    // For video, de-duplicate and show unique detections with timestamps
    const detections = data.detections || [];
    if (detections.length > 0) {
        // Show up to 20 most relevant detections
        const shown = detections.slice(0, 20);
        shown.forEach(d => {
            const dm = SEVERITY_MAP[d.class] || { dot: '#64748b' };
            const item = document.createElement('div');
            item.className = 'det-item fade-up';
            const tsLabel = d.timestamp != null ? `<span class="det-ts">@ ${d.timestamp}s</span>` : '';
            item.innerHTML = `
                <span style="display:flex;align-items:center;gap:.4rem;">
                    <span class="det-dot" style="background:${dm.dot}"></span>
                    <span class="det-class">${d.class.replace(/_/g, ' ')}</span>
                    ${tsLabel}
                </span>
                <span class="det-conf">${(d.confidence * 100).toFixed(1)}%</span>
            `;
            list.appendChild(item);
        });
        if (detections.length > 20) {
            const more = document.createElement('p');
            more.className = 'no-detections';
            more.textContent = `+ ${detections.length - 20} more detections…`;
            list.appendChild(more);
        }
    } else {
        const mediaType = isVideo ? 'video' : 'image';
        list.innerHTML = `<p class="no-detections">No incidents detected in this ${mediaType}.</p>`;
    }

    // Notify section (show if any real detections)
    const notifySection = document.getElementById('notifySection');
    notifySection.style.display = (detections.length > 0) ? 'block' : 'none';

    // Guidance
    renderGuidance(data.guidance, severity);

    // Show results
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ───────── Render Guidance ─────────

function renderGuidance(g, severity) {
    const card = document.getElementById('guidanceCard');
    const content = document.getElementById('guidanceContent');
    content.innerHTML = '';

    if (!g || severity === 'no_accident') {
        card.style.display = 'none';
        return;
    }
    card.style.display = 'block';

    // LLM-enhanced summary
    if (g.llm_enhanced) {
        const div = document.createElement('div');
        div.className = 'llm-summary';
        div.innerHTML = `<strong>🧠 AI Summary:</strong> ${g.llm_enhanced}`;
        content.appendChild(div);
    }

    // Sections
    if (g.immediate_actions && g.immediate_actions.length)
        content.appendChild(makeGuidanceSection('🚑 Immediate Actions', g.immediate_actions, 'actions'));
    if (g.warning_signs && g.warning_signs.length)
        content.appendChild(makeGuidanceSection('⚠️ Warning Signs', g.warning_signs, 'warnings'));
    if (g.do_not_do && g.do_not_do.length)
        content.appendChild(makeGuidanceSection('🚫 Do Not', g.do_not_do, 'donts'));
    if (g.recommendations && g.recommendations.length)
        content.appendChild(makeGuidanceSection('📋 Recommendations', g.recommendations, 'recs'));
}

function makeGuidanceSection(title, items, cls) {
    const div = document.createElement('div');
    div.className = `guidance-section ${cls}`;
    div.innerHTML = `<h4>${title}</h4><ul>${items.map(i => `<li>${i}</li>`).join('')}</ul>`;
    return div;
}

// ───────── Notifications ─────────

document.getElementById('notifyBtn').addEventListener('click', async () => {
    const btn = document.getElementById('notifyBtn');
    const loc = document.getElementById('locationInput').value.trim();
    const status = document.getElementById('notifyStatus');

    if (!loc) { status.textContent = 'Please enter a location.'; status.className = 'notify-status error'; return; }

    setBtnLoading(btn, true);
    status.textContent = '';

    // Determine type from current severity
    const sevClass = document.getElementById('severityClass').textContent.trim();
    const type = sevClass.includes('pothole') ? 'pothole' : 'accident';

    try {
        const res = await fetch(`${API_BASE}/api/notify`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
            body: JSON.stringify({ type, location: loc, incident_id: currentIncidentId })
        });
        if (res.status === 401) { showAuthModal(); return; }
        const data = await res.json();
        status.textContent = data.message || 'Notification sent.';
        status.className = 'notify-status';
    } catch (err) {
        status.textContent = 'Failed to send notification.';
        status.className = 'notify-status error';
    } finally {
        setBtnLoading(btn, false);
    }
});

// ───────── Helpers ─────────

function setBtnLoading(btn, loading) {
    const text = btn.querySelector('.btn-text');
    const loader = btn.querySelector('.btn-loader');
    if (loading) {
        text.style.display = 'none';
        loader.style.display = 'inline-flex';
        btn.disabled = true;
    } else {
        text.style.display = 'inline';
        loader.style.display = 'none';
        btn.disabled = false;
    }
}

// ───────── Auth Modal ─────────

function showAuthModal() {
    document.getElementById('authModal').style.display = 'flex';
}

function hideAuthModal() {
    document.getElementById('authModal').style.display = 'none';
}

function switchAuthTab(tab) {
    document.getElementById('loginForm').style.display = tab === 'login' ? 'block' : 'none';
    document.getElementById('registerForm').style.display = tab === 'register' ? 'block' : 'none';
    document.getElementById('loginTab').classList.toggle('active', tab === 'login');
    document.getElementById('registerTab').classList.toggle('active', tab === 'register');
}

document.getElementById('loginForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const btn = e.target.querySelector('button[type=submit]');
    const errEl = document.getElementById('loginError');
    const email = document.getElementById('loginEmail').value.trim();
    const password = document.getElementById('loginPassword').value;
    setBtnLoading(btn, true);
    errEl.textContent = '';
    try {
        const res = await fetch('/auth/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: new URLSearchParams({ username: email, password }).toString(),
        });
        const data = await res.json();
        if (!res.ok) { errEl.textContent = data.detail || 'Invalid credentials'; return; }
        authToken = data.access_token;
        localStorage.setItem('cp_token', authToken);
        hideAuthModal();
        onAuthSuccess();
    } catch (err) {
        errEl.textContent = 'Login failed. Try again.';
    } finally {
        setBtnLoading(btn, false);
    }
});

document.getElementById('registerForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const btn = e.target.querySelector('button[type=submit]');
    const errEl = document.getElementById('registerError');
    const email = document.getElementById('regEmail').value.trim();
    const password = document.getElementById('regPassword').value;
    const phone = document.getElementById('regPhone').value.trim();
    setBtnLoading(btn, true);
    errEl.textContent = '';
    try {
        const res = await fetch('/auth/register', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, password, phone: phone || null, role: 'citizen' }),
        });
        const data = await res.json();
        if (!res.ok) { errEl.textContent = data.detail || 'Registration failed'; return; }
        // Auto-login after register
        const lr = await fetch('/auth/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: new URLSearchParams({ username: email, password }).toString(),
        });
        const ld = await lr.json();
        if (lr.ok) {
            authToken = ld.access_token;
            localStorage.setItem('cp_token', authToken);
            hideAuthModal();
            onAuthSuccess();
        } else {
            switchAuthTab('login');
            document.getElementById('loginEmail').value = email;
        }
    } catch (err) {
        errEl.textContent = 'Registration failed. Try again.';
    } finally {
        setBtnLoading(btn, false);
    }
});

function logout() {
    authToken = null;
    currentIncidentId = null;
    localStorage.removeItem('cp_token');
    document.getElementById('logoutBtn').style.display = 'none';
    document.getElementById('historyNavPill').style.display = 'none';
    document.getElementById('dashboardNavPill').style.display = 'none';
    document.getElementById('userInfo').textContent = '';
    document.getElementById('history').style.display = 'none';
    document.getElementById('dashboard').style.display = 'none';
    if (_map) { _map.remove(); _map = null; }
    resetUpload();
    showAuthModal();
}

function onAuthSuccess() {
    document.getElementById('logoutBtn').style.display = 'inline-flex';
    document.getElementById('historyNavPill').style.display = 'inline-flex';
    document.getElementById('dashboardNavPill').style.display = 'inline-flex';
    fetch('/auth/me', { headers: getAuthHeaders() })
        .then(r => r.json())
        .then(data => { document.getElementById('userInfo').textContent = data.email || ''; })
        .catch(() => {});
}

// ───────── Incident History ─────────

async function loadHistory() {
    const section = document.getElementById('history');
    section.style.display = 'block';
    section.scrollIntoView({ behavior: 'smooth' });

    const loadingEl = document.getElementById('historyLoading');
    const listEl = document.getElementById('historyList');
    const emptyEl = document.getElementById('historyEmpty');

    loadingEl.style.display = 'flex';
    listEl.innerHTML = '';
    emptyEl.style.display = 'none';

    try {
        const res = await fetch('/api/incidents?limit=20', { headers: getAuthHeaders() });
        if (res.status === 401) { showAuthModal(); return; }
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();

        loadingEl.style.display = 'none';
        if (!data.incidents || data.incidents.length === 0) {
            emptyEl.style.display = 'block';
            return;
        }

        data.incidents.forEach(inc => {
            const meta = SEVERITY_MAP[inc.severity] || { dot: '#64748b', label: inc.severity };
            const date = inc.created_at ? new Date(inc.created_at).toLocaleString() : 'Unknown';
            const el = document.createElement('div');
            el.className = 'history-item glass-card fade-up';
            el.innerHTML = `
                <div class="history-img">
                    ${inc.image_url
                        ? `<img src="${inc.image_url}" alt="Incident ${inc.id}" loading="lazy">`
                        : '<div class="history-no-img">📷</div>'}
                </div>
                <div class="history-info">
                    <div class="history-severity">
                        <span class="det-dot" style="background:${meta.dot};flex-shrink:0;"></span>
                        <span>${(meta.label || inc.severity).replace(/_/g, ' ')}</span>
                    </div>
                    <p class="history-location">${inc.address_text || 'Location not set'}</p>
                    <p class="history-date">${date}</p>
                </div>
                <div class="history-status">
                    <span class="status-badge status-${inc.status || 'detected'}">${(inc.status || 'detected').replace(/_/g, ' ')}</span>
                </div>
            `;
            listEl.appendChild(el);
        });

        if (data.total > 20) {
            const more = document.createElement('p');
            more.className = 'no-detections';
            more.style.marginTop = '.75rem';
            more.textContent = `Showing 20 of ${data.total} total incidents.`;
            listEl.appendChild(more);
        }
    } catch (err) {
        loadingEl.style.display = 'none';
        listEl.innerHTML = `<p class="no-detections" style="color:var(--red)">Failed to load history: ${err.message}</p>`;
    }
}

// ───────── Page Init ─────────

(function init() {
    if (!authToken) {
        showAuthModal();
    } else {
        onAuthSuccess();
    }
})();

// ───────── Dashboard ─────────

let _chartSeverity = null;
let _chartStatus = null;
let _chartDaily = null;
let _map = null;

const CHART_COLORS = {
    no_accident:       'rgba(34,197,94,0.8)',
    minor_accident:    'rgba(245,158,11,0.8)',
    moderate_accident: 'rgba(249,115,22,0.8)',
    severe_accident:   'rgba(239,68,68,0.8)',
    totaled_vehicle:   'rgba(185,28,28,0.8)',
    pothole:           'rgba(6,182,212,0.8)',
};

const STATUS_COLORS = {
    detected:     'rgba(59,130,246,0.8)',
    notified:     'rgba(245,158,11,0.8)',
    acknowledged: 'rgba(168,85,247,0.8)',
    resolved:     'rgba(34,197,94,0.8)',
    closed:       'rgba(100,116,139,0.8)',
};

const CHART_DEFAULTS = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { labels: { color: '#94a3b8', font: { family: 'Inter' } } } },
};

function destroyChart(ref) { if (ref) { ref.destroy(); } return null; }

async function loadDashboard() {
    const section = document.getElementById('dashboard');
    section.style.display = 'block';
    section.scrollIntoView({ behavior: 'smooth' });

    try {
        const [statsRes, incRes] = await Promise.all([
            fetch('/api/stats', { headers: getAuthHeaders() }),
            fetch('/api/incidents?limit=100', { headers: getAuthHeaders() }),
        ]);

        if (statsRes.status === 401 || incRes.status === 401) { showAuthModal(); return; }

        const stats = await statsRes.json();
        const incData = await incRes.json();

        // ── Stat tiles ──
        const open = (stats.by_status.detected || 0) + (stats.by_status.notified || 0) + (stats.by_status.acknowledged || 0);
        document.getElementById('statTotal').textContent = stats.total ?? 0;
        document.getElementById('statOpen').textContent = open;
        document.getElementById('statResolved').textContent = stats.by_status.resolved ?? 0;
        document.getElementById('statCritical').textContent =
            (stats.by_severity.severe_accident || 0) + (stats.by_severity.totaled_vehicle || 0);

        // ── Severity doughnut ──
        _chartSeverity = destroyChart(_chartSeverity);
        const sevLabels = Object.keys(stats.by_severity).map(k => k.replace(/_/g, ' '));
        const sevData   = Object.values(stats.by_severity);
        const sevColors = Object.keys(stats.by_severity).map(k => CHART_COLORS[k] || 'rgba(148,163,184,0.8)');
        _chartSeverity = new Chart(document.getElementById('chartSeverity'), {
            type: 'doughnut',
            data: { labels: sevLabels, datasets: [{ data: sevData, backgroundColor: sevColors, borderWidth: 0 }] },
            options: { ...CHART_DEFAULTS, cutout: '65%' },
        });

        // ── Status doughnut ──
        _chartStatus = destroyChart(_chartStatus);
        const stLabels = Object.keys(stats.by_status);
        const stData   = Object.values(stats.by_status);
        const stColors = stLabels.map(k => STATUS_COLORS[k] || 'rgba(148,163,184,0.8)');
        _chartStatus = new Chart(document.getElementById('chartStatus'), {
            type: 'doughnut',
            data: { labels: stLabels, datasets: [{ data: stData, backgroundColor: stColors, borderWidth: 0 }] },
            options: { ...CHART_DEFAULTS, cutout: '65%' },
        });

        // ── Daily bar chart ──
        _chartDaily = destroyChart(_chartDaily);
        // Fill in missing days with 0
        const dailyMap = {};
        (stats.daily_last_7_days || []).forEach(d => { dailyMap[d.date] = d.count; });
        const dayLabels = [];
        const dayValues = [];
        for (let i = 6; i >= 0; i--) {
            const d = new Date();
            d.setDate(d.getDate() - i);
            const key = d.toISOString().split('T')[0];
            dayLabels.push(key.slice(5)); // MM-DD
            dayValues.push(dailyMap[key] || 0);
        }
        _chartDaily = new Chart(document.getElementById('chartDaily'), {
            type: 'bar',
            data: {
                labels: dayLabels,
                datasets: [{
                    label: 'Incidents',
                    data: dayValues,
                    backgroundColor: 'rgba(59,130,246,0.7)',
                    borderRadius: 4,
                }],
            },
            options: {
                ...CHART_DEFAULTS,
                scales: {
                    x: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                    y: { ticks: { color: '#94a3b8', stepSize: 1 }, grid: { color: 'rgba(255,255,255,0.05)' }, beginAtZero: true },
                },
                plugins: { legend: { display: false } },
            },
        });

        // ── Leaflet map ──
        const incidents = (incData.incidents || []).filter(i => i.lat && i.lng);
        const mapEl = document.getElementById('incidentMap');
        const hintEl = document.getElementById('mapHint');

        if (incidents.length === 0) {
            mapEl.style.display = 'none';
            hintEl.style.display = 'block';
        } else {
            mapEl.style.display = 'block';
            hintEl.style.display = 'none';

            if (_map) { _map.remove(); _map = null; }
            _map = L.map('incidentMap', { zoomControl: true });
            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                attribution: '© OpenStreetMap contributors',
                maxZoom: 18,
            }).addTo(_map);

            const bounds = [];
            incidents.forEach(inc => {
                const meta = SEVERITY_MAP[inc.severity] || { dot: '#64748b', label: inc.severity };
                const color = meta.dot;
                const marker = L.circleMarker([inc.lat, inc.lng], {
                    radius: 9,
                    fillColor: color,
                    color: '#0a0e1a',
                    weight: 2,
                    opacity: 1,
                    fillOpacity: 0.85,
                }).addTo(_map);
                marker.bindPopup(`
                    <strong>${(meta.label || inc.severity).replace(/_/g, ' ')}</strong><br>
                    ${inc.address_text || ''}<br>
                    <small>${inc.created_at ? new Date(inc.created_at).toLocaleString() : ''}</small>
                `);
                bounds.push([inc.lat, inc.lng]);
            });

            if (bounds.length === 1) {
                _map.setView(bounds[0], 13);
            } else {
                _map.fitBounds(bounds, { padding: [40, 40] });
            }
        }

    } catch (err) {
        console.error('Dashboard error:', err);
    }
}
