/* ── Atlético Intelligence — app.js ──────────────────────────────────────────
   State machine:
     upload → processing → result (offside / onside / no_detect) | error
   ──────────────────────────────────────────────────────────────────────────── */

const API = '/api';
const POLL_INTERVAL_MS = 1500;

// ── State ──────────────────────────────────────────────────────────────────
let currentJobId   = null;
let pollTimer      = null;
let selectedFile   = null;

// ── DOM refs ────────────────────────────────────────────────────────────────
const states = {
  upload:      document.getElementById('state-upload'),
  processing:  document.getElementById('state-processing'),
  result:      document.getElementById('state-result'),
  error:       document.getElementById('state-error'),
};

const uploadZone     = document.getElementById('uploadZone');
const fileInput      = document.getElementById('fileInput');
const browseBtn      = document.getElementById('browseBtn');
const filePreview    = document.getElementById('filePreview');
const fileName       = document.getElementById('fileName');
const fileSize       = document.getElementById('fileSize');
const removeFile     = document.getElementById('removeFile');
const analyzeBtn     = document.getElementById('analyzeBtn');
const uploadError    = document.getElementById('uploadError');

const progressFill   = document.getElementById('progressFill');
const progressPct    = document.getElementById('progressPct');
const progressStage  = document.getElementById('progressStage');
const processingMsg  = document.getElementById('processingMsg');

const verdictBanner  = document.getElementById('verdictBanner');
const verdictLabel   = document.getElementById('verdictLabel');
const verdictSub     = document.getElementById('verdictSub');
const resultVideo    = document.getElementById('resultVideo');
const downloadBtn    = document.getElementById('downloadBtn');
const newAnalysisBtn = document.getElementById('newAnalysisBtn');
const jobIdBadge     = document.getElementById('jobIdBadge');
const summaryVerdict = document.getElementById('summaryVerdict');
const summaryJobId   = document.getElementById('summaryJobId');
const summaryStatus  = document.getElementById('summaryStatus');

const errorMsg       = document.getElementById('errorMsg');
const retryBtn       = document.getElementById('retryBtn');

// ── State switcher ──────────────────────────────────────────────────────────
function showState(name) {
  Object.entries(states).forEach(([key, el]) => {
    if (key === name) {
      el.style.display = 'block';
      el.classList.add('state--active');
    } else {
      el.style.display = 'none';
      el.classList.remove('state--active');
    }
  });
}

// ── File selection ──────────────────────────────────────────────────────────
browseBtn.addEventListener('click', () => fileInput.click());

uploadZone.addEventListener('click', (e) => {
  if (e.target === browseBtn) return;
  fileInput.click();
});

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

// Drag & drop
['dragenter','dragover'].forEach(ev =>
  uploadZone.addEventListener(ev, e => { e.preventDefault(); uploadZone.classList.add('drag-over'); })
);
['dragleave','drop'].forEach(ev =>
  uploadZone.addEventListener(ev, e => { e.preventDefault(); uploadZone.classList.remove('drag-over'); })
);
uploadZone.addEventListener('drop', e => {
  const f = e.dataTransfer.files[0];
  if (f) handleFile(f);
});

removeFile.addEventListener('click', () => {
  selectedFile = null;
  fileInput.value = '';
  filePreview.classList.add('hidden');
  analyzeBtn.disabled = true;
  hideError();
});

function handleFile(file) {
  hideError();

  const allowed = ['video/mp4','video/avi','video/quicktime','video/x-matroska',
                   'video/x-msvideo','video/mov','video/mkv'];
  const ext     = file.name.split('.').pop().toLowerCase();
  const allowedExt = ['mp4','avi','mov','mkv'];

  if (!allowedExt.includes(ext)) {
    showError(`Unsupported file type ".${ext}". Please upload MP4, AVI, MOV or MKV.`);
    return;
  }
  if (file.size > 500 * 1024 * 1024) {
    showError('File too large. Maximum size is 500 MB.');
    return;
  }

  selectedFile = file;
  fileName.textContent = file.name;
  fileSize.textContent = formatBytes(file.size);
  filePreview.classList.remove('hidden');
  analyzeBtn.disabled = false;
}

function formatBytes(bytes) {
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

function showError(msg) {
  uploadError.textContent = msg;
  uploadError.classList.remove('hidden');
}
function hideError() {
  uploadError.classList.add('hidden');
}

// ── Upload & start ──────────────────────────────────────────────────────────
analyzeBtn.addEventListener('click', startAnalysis);

async function startAnalysis() {
  if (!selectedFile) return;
  hideError();

  const formData = new FormData();
  formData.append('file', selectedFile);

  showState('processing');
  setProgress(2, 'Uploading video…');

  try {
    const res  = await fetch(`${API}/upload`, { method: 'POST', body: formData });
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.detail || `Upload failed (${res.status})`);
    }

    currentJobId = data.job_id;
    setProgress(5, 'Queued for processing…');
    startPolling();

  } catch (err) {
    showErrorState(err.message);
  }
}

// ── Polling ─────────────────────────────────────────────────────────────────
function startPolling() {
  stopPolling();
  pollTimer = setInterval(pollStatus, POLL_INTERVAL_MS);
}
function stopPolling() {
  if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
}

const STAGE_LABELS = {
  5:  'Initializing models…',
  10: 'Collecting player crops…',
  20: 'Training team classifier…',
  35: 'Running detection…',
  50: 'Computing homography…',
  65: 'Offside analysis…',
  80: 'Rendering annotations…',
  90: 'Encoding output video…',
  99: 'Finalizing…',
};

function getStageLabel(pct) {
  const keys = Object.keys(STAGE_LABELS).map(Number).sort((a,b) => b-a);
  for (const k of keys) {
    if (pct >= k) return STAGE_LABELS[k];
  }
  return 'Processing…';
}

async function pollStatus() {
  if (!currentJobId) return;
  try {
    const res  = await fetch(`${API}/status/${currentJobId}`);
    const data = await res.json();

    if (!res.ok) throw new Error(data.detail || 'Poll failed');

    const pct = data.progress || 0;
    setProgress(pct, getStageLabel(pct));

    if (data.status === 'done') {
      stopPolling();
      setProgress(100, 'Complete!');
      setTimeout(fetchResult, 400);
    } else if (data.status === 'error') {
      stopPolling();
      showErrorState(data.message || 'Processing failed');
    }
  } catch (err) {
    // network blip — keep polling
    console.warn('Poll error:', err);
  }
}

async function fetchResult() {
  try {
    const res  = await fetch(`${API}/result/${currentJobId}`);
    const data = await res.json();

    if (!res.ok) throw new Error(data.detail || 'Result fetch failed');
    showResult(data);

  } catch (err) {
    showErrorState(err.message);
  }
}

// ── Progress bar ─────────────────────────────────────────────────────────────
function setProgress(pct, stage) {
  progressFill.style.width  = `${pct}%`;
  progressPct.textContent   = `${pct}%`;
  progressStage.textContent = stage || '';
  processingMsg.textContent = stage || '';
  animatePitchDots(pct);
}

// ── Pitch dot animation ──────────────────────────────────────────────────────
let dotAnim = null;
function animatePitchDots(progress) {
  const blue = document.getElementById('dotBlue');
  const pink = document.getElementById('dotPink');
  const ball = document.getElementById('dotBall');
  const line = document.getElementById('offsideLineAnim');

  // random-ish positions that evolve with progress
  const t = progress / 100;
  blue.style.left = `${30 + Math.sin(t * Math.PI * 2) * 15}%`;
  blue.style.top  = `${40 + Math.cos(t * Math.PI * 3) * 20}%`;
  pink.style.left = `${55 + Math.cos(t * Math.PI * 2.5) * 20}%`;
  pink.style.top  = `${50 + Math.sin(t * Math.PI * 1.5) * 20}%`;
  ball.style.left = `${45 + Math.sin(t * Math.PI * 4) * 25}%`;
  ball.style.top  = `${50 + Math.cos(t * Math.PI * 3) * 15}%`;

  if (progress > 60) {
    line.style.opacity = '1';
    line.style.left    = `${58 + Math.sin(t * Math.PI) * 5}%`;
  }
}

// ── Show result ──────────────────────────────────────────────────────────────
function showResult(data) {
  const verdict = (data.verdict || 'UNKNOWN').toUpperCase();

  // Verdict banner
  verdictBanner.className = 'verdict-banner';
  if (verdict === 'OFFSIDE') {
    verdictBanner.classList.add('offside');
    verdictLabel.textContent = 'OFFSIDE';
    verdictSub.textContent   = 'Player detected ahead of the last defender at time of pass.';
  } else if (verdict === 'ONSIDE') {
    verdictBanner.classList.add('onside');
    verdictLabel.textContent = 'ONSIDE';
    verdictSub.textContent   = 'All attacking players were in an onside position.';
  } else {
    verdictBanner.classList.add('no-detect');
    verdictLabel.textContent = 'NO FOOTBALL DETECTED';
    verdictSub.textContent   = 'Could not detect players or a valid pitch in this clip.';
  }

  // Video — use full URL so browser range requests work correctly
  const videoUrl = `${window.location.origin}${data.video_url}`;
  resultVideo.pause();
  resultVideo.removeAttribute('src');
  resultVideo.load();
  resultVideo.src = videoUrl;
  resultVideo.load();

  // Download
  downloadBtn.onclick = () => {
    const a = document.createElement('a');
    a.href     = `${API}/download/${currentJobId}`;
    a.download = `offside_analysis_${currentJobId.slice(0,8)}.mp4`;
    a.click();
  };

  // Summary panel
  jobIdBadge.textContent     = currentJobId.slice(0, 8) + '…';
  summaryVerdict.textContent = verdict;
  summaryVerdict.style.color = verdict === 'OFFSIDE' ? 'var(--red)'
                             : verdict === 'ONSIDE'  ? 'var(--green)'
                             : 'var(--orange)';
  summaryJobId.textContent   = currentJobId.slice(0, 16) + '…';
  summaryStatus.textContent  = 'Complete ✓';
  summaryStatus.style.color  = 'var(--green)';

  showState('result');
}

// ── Error state ──────────────────────────────────────────────────────────────
function showErrorState(msg) {
  stopPolling();
  errorMsg.textContent = msg || 'An unexpected error occurred.';
  showState('error');
}

// ── Reset / new analysis ──────────────────────────────────────────────────────
function resetToUpload() {
  stopPolling();
  currentJobId = null;
  selectedFile = null;
  fileInput.value = '';
  filePreview.classList.add('hidden');
  analyzeBtn.disabled = true;
  hideError();
  setProgress(0, '');
  resultVideo.src = '';
  const line = document.getElementById('offsideLineAnim');
  if (line) line.style.opacity = '0';
  showState('upload');
}

newAnalysisBtn.addEventListener('click', resetToUpload);
retryBtn.addEventListener('click', resetToUpload);

// ── Init ──────────────────────────────────────────────────────────────────────
showState('upload');
