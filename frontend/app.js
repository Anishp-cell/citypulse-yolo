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
        const res = await fetch(`${API_BASE}${endpoint}`, { method: 'POST', body: formData });
        if (!res.ok) throw new Error(`Server error ${res.status}`);
        const data = await res.json();
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
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ type, location: loc })
        });
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
