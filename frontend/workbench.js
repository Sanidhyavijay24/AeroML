// -*- coding: utf-8 -*-
/**
 * @file workbench.js
 * @description Frontend controller for drafting board drawing, simulation requests, and optimization
 * @module frontend
 */

// UI Elements Lookup
const fileUpload = document.getElementById('datFileUpload');
const uploadStatus = document.getElementById('fileUploadStatus');
const btnRunPrediction = document.getElementById('btnRunPrediction');
const btnRunOptimization = document.getElementById('btnRunOptimization');
const btnExportDat = document.getElementById('btnExportDat');
const terminalLog = document.getElementById('terminalLog');
const candidateSelect = document.getElementById('candidateSelect');

// Tab Switch Elements
const tabForward = document.getElementById('tabForward');
const tabReverse = document.getElementById('tabReverse');
const panelForwardInputs = document.getElementById('panelForwardInputs');
const panelReverseInputs = document.getElementById('panelReverseInputs');
const panelForwardOutputs = document.getElementById('panelForwardOutputs');
const panelReverseOutputs = document.getElementById('panelReverseOutputs');

let activeMode = 'forward'; // 'forward' | 'reverse'
let isLoading = false;      // Loading animation state

// Handle Mode Switching
function switchMode(mode) {
  activeMode = mode;
  activeGeometry = null; // Clear viewports on switch
  drawViewport();

  if (mode === 'forward') {
    tabForward.classList.add('active');
    tabReverse.classList.remove('active');
    panelForwardInputs.classList.remove('hidden');
    panelForwardOutputs.classList.remove('hidden');
    panelReverseInputs.classList.add('hidden');
    panelReverseOutputs.classList.add('hidden');
    logTerminal('Switched to FORWARD ANALYSIS MODE.', 'info');
  } else {
    tabForward.classList.remove('active');
    tabReverse.classList.add('active');
    panelForwardInputs.classList.add('hidden');
    panelForwardOutputs.classList.add('hidden');
    panelReverseInputs.classList.remove('hidden');
    panelReverseOutputs.classList.remove('hidden');
    logTerminal('Switched to REVERSE OPTIMIZATION MODE.', 'info');
  }
}

tabForward.addEventListener('click', () => switchMode('forward'));
tabReverse.addEventListener('click', () => switchMode('reverse'));

// Sliders and Readouts
const sliders = [
  { slider: 'machPredictSlider', readout: 'machPredictVal', format: (v) => parseFloat(v).toFixed(3) },
  { slider: 'rePredictSlider', readout: 'rePredictVal', format: (v) => parseFloat(v).toFixed(2) },
  { slider: 'targetLdSlider', readout: 'targetLdVal', format: (v) => parseInt(v) },
  { slider: 'targetClSlider', readout: 'targetClVal', format: (v) => parseFloat(v).toFixed(2) },
  { slider: 'targetCdSlider', readout: 'targetCdVal', format: (v) => parseFloat(v).toFixed(3) },
  { slider: 'machOptimizeSlider', readout: 'machOptimizeVal', format: (v) => parseFloat(v).toFixed(3) },
  { slider: 'reOptimizeSlider', readout: 'reOptimizeVal', format: (v) => parseFloat(v).toFixed(2) }
];

// Initialize sliders events
sliders.forEach(({ slider, readout, format }) => {
  const sliderEl = document.getElementById(slider);
  const readoutEl = document.getElementById(readout);
  if (sliderEl && readoutEl) {
    sliderEl.addEventListener('input', (e) => {
      readoutEl.innerText = format(e.target.value);
    });
  }
});

// Canvas Setup
const canvas = document.getElementById('cadCanvas');
const ctx = canvas.getContext('2d');
let activeGeometry = null; // Holds { x, y_upper, y_lower, cl }
let activeCandidates = []; // Holds list of optimization candidates

// Resize handler
function resizeCanvas() {
  const rect = canvas.parentElement.getBoundingClientRect();
  canvas.width = rect.width;
  canvas.height = rect.height;
  drawViewport();
}
window.addEventListener('resize', resizeCanvas);
window.addEventListener('load', resizeCanvas);

// Logger helper
function logTerminal(message, type = 'info') {
  const entry = document.createElement('div');
  entry.className = `log-entry ${type}`;
  entry.innerText = `> ${message}`;
  terminalLog.appendChild(entry);
  terminalLog.scrollTop = terminalLog.scrollHeight;
}

// --------------------------------------------------------------------------
// 📐 CAD Viewport Drawing Functions
// --------------------------------------------------------------------------
function drawViewport() {
  const w = canvas.width;
  const h = canvas.height;

  // Render scanning loader state
  if (isLoading) {
    ctx.fillStyle = '#120f0f';
    ctx.fillRect(0, 0, w, h);

    // Draw technical grid lines
    ctx.strokeStyle = 'rgba(237, 235, 221, 0.05)';
    ctx.lineWidth = 1;
    for (let x = 0; x < w; x += 20) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    }
    for (let y = 0; y < h; y += 20) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }

    // Glowing horizontal laser scanning sweep line
    const scanY = h / 2 + Math.sin(Date.now() / 150) * (h / 2 - 30);
    ctx.fillStyle = 'rgba(129, 1, 0, 0.15)';
    ctx.fillRect(0, scanY - 15, w, 30);
    ctx.strokeStyle = 'var(--cherry-red)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(0, scanY);
    ctx.lineTo(w, scanY);
    ctx.stroke();

    // Render loading labels
    ctx.fillStyle = 'var(--cotton)';
    ctx.font = '13px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    ctx.fillText('SOLVER EXECUTING SIMULATION...', w / 2, h / 2 - 10);
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.fillStyle = 'var(--maroon)';
    ctx.fillText('ESTABLISHING STEADY-STATE VELOCITY VECTORS...', w / 2, h / 2 + 15);

    requestAnimationFrame(drawViewport);
    return;
  }

  // Clear background
  ctx.fillStyle = '#120f0f';
  ctx.fillRect(0, 0, w, h);

  // Draw technical grid lines
  ctx.strokeStyle = 'rgba(237, 235, 221, 0.04)';
  ctx.lineWidth = 0.5;
  const gridSpacing = 20;

  for (let gx = 0; gx < w; gx += gridSpacing) {
    ctx.beginPath();
    ctx.moveTo(gx, 0);
    ctx.lineTo(gx, h);
    ctx.stroke();
  }
  for (let gy = 0; gy < h; gy += gridSpacing) {
    ctx.beginPath();
    ctx.moveTo(0, gy);
    ctx.lineTo(w, gy);
    ctx.stroke();
  }

  // Draw centered axis lines
  ctx.strokeStyle = 'rgba(237, 235, 221, 0.18)';
  ctx.lineWidth = 1;
  // X axis (chord line)
  ctx.beginPath();
  ctx.moveTo(0, h / 2);
  ctx.lineTo(w, h / 2);
  ctx.stroke();

  // If no shape is active, render a default layout/placeholder message
  if (!activeGeometry) {
    ctx.fillStyle = 'rgba(237, 235, 221, 0.3)';
    ctx.font = '12px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    ctx.fillText('NO GEOMETRY LOADED | VIEWPORT IDLE', w / 2, h / 2 - 10);
    return;
  }

  const { x, y_upper, y_lower, cl = 1.0 } = activeGeometry;
  const n = x.length;

  // Compute Pressure offsets
  const cpSuction = x.map(xi => -Math.max(cl, 0.1) * 0.18 * Math.sin(Math.PI * xi) - 0.015);
  const cpCompression = x.map(xi => Math.max(cl, 0.1) * 0.10 * Math.sin(Math.PI * xi) + 0.008);

  // Compute peak vertical extent to prevent clipping dynamically
  const maxUpperVal = Math.max(...y_upper.map((yu, i) => yu - cpSuction[i]));
  const maxLowerVal = Math.max(...y_lower.map((yl, i) => Math.abs(yl - cpCompression[i])));
  const peakVal = Math.max(maxUpperVal, maxLowerVal, 0.15);

  const padding = 50;
  // Use 95% of available horizontal space to fill the viewport
  let scale = (w - 2 * padding) * 0.95;

  // Limit scale to fit height boundaries — 40px vertical margin
  const maxScaleY = (h / 2 - 40) / peakVal;
  if (scale > maxScaleY) {
    scale = maxScaleY;
  }
  const scaleY = scale;

  const mapX = (val) => padding + (w - 2 * padding - scale) / 2 + val * scale;
  const mapY = (val) => h / 2 - val * scaleY;

  // ---- X-axis chord labels (0.0, 0.2, ... 1.0) ----
  ctx.font = '9px "JetBrains Mono", monospace';
  ctx.textAlign = 'center';
  ctx.fillStyle = 'rgba(237, 235, 221, 0.40)';
  ctx.strokeStyle = 'rgba(237, 235, 221, 0.12)';
  ctx.lineWidth = 0.5;
  for (let t = 0; t <= 1.001; t += 0.2) {
    const px = mapX(t);
    ctx.beginPath();
    ctx.moveTo(px, h / 2 + 4);
    ctx.lineTo(px, h / 2 + 10);
    ctx.stroke();
    ctx.fillText(t.toFixed(1), px, h / 2 + 20);
    // Faint vertical guide
    ctx.save();
    ctx.strokeStyle = 'rgba(237, 235, 221, 0.05)';
    ctx.beginPath();
    ctx.moveTo(px, 0);
    ctx.lineTo(px, h);
    ctx.stroke();
    ctx.restore();
  }
  // X-axis label
  ctx.font = '8px "JetBrains Mono", monospace';
  ctx.fillStyle = 'rgba(237, 235, 221, 0.30)';
  ctx.fillText('x/c', mapX(1.0) + 22, h / 2 + 5);

  // ---- Y-axis Cp scale markers ----
  ctx.textAlign = 'right';
  ctx.font = '8px "JetBrains Mono", monospace';
  ctx.fillStyle = 'rgba(237, 235, 221, 0.30)';
  const cpTicks = [-0.2, -0.1, 0.0, 0.1, 0.2];
  cpTicks.forEach(cpVal => {
    const py = mapY(cpVal);
    if (py > 10 && py < h - 10) {
      ctx.beginPath();
      ctx.strokeStyle = 'rgba(237, 235, 221, 0.06)';
      ctx.lineWidth = 0.5;
      ctx.setLineDash([2, 4]);
      ctx.moveTo(padding - 5, py);
      ctx.lineTo(w - padding + 5, py);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillText(cpVal.toFixed(1), padding - 8, py + 3);
    }
  });
  // Cp label
  ctx.save();
  ctx.translate(padding - 32, h / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = 'center';
  ctx.fillText('Cp', 0, 0);
  ctx.restore();

  // ---- SUCTION REGION (Cherry Red — bright, solid boundary) ----
  // Gradient fill
  const gradSuction = ctx.createLinearGradient(0, mapY(peakVal), 0, mapY(0.0));
  gradSuction.addColorStop(0, 'rgba(201, 16, 16, 0.02)');
  gradSuction.addColorStop(0.5, 'rgba(201, 16, 16, 0.12)');
  gradSuction.addColorStop(1, 'rgba(201, 16, 16, 0.25)');

  ctx.fillStyle = gradSuction;
  ctx.beginPath();
  ctx.moveTo(mapX(x[0]), mapY(y_upper[0]));
  for (let i = 0; i < n; i++) {
    ctx.lineTo(mapX(x[i]), mapY(y_upper[i] - cpSuction[i]));
  }
  for (let i = n - 1; i >= 0; i--) {
    ctx.lineTo(mapX(x[i]), mapY(y_upper[i]));
  }
  ctx.closePath();
  ctx.fill();

  // Suction boundary line (cherry-red — long dash, solid feel)
  ctx.strokeStyle = 'rgba(201, 16, 16, 0.9)';
  ctx.setLineDash([8, 3]);
  ctx.lineWidth = 1.8;
  ctx.beginPath();
  ctx.moveTo(mapX(x[0]), mapY(y_upper[0] - cpSuction[0]));
  for (let i = 1; i < n; i++) {
    ctx.lineTo(mapX(x[i]), mapY(y_upper[i] - cpSuction[i]));
  }
  ctx.stroke();

  // 5 contour levels inside suction region (cherry-red, progressively brighter)
  const suctionLevels = [0.15, 0.30, 0.50, 0.70, 0.85];
  suctionLevels.forEach((pct, idx) => {
    const alpha = 0.18 + idx * 0.06;
    ctx.strokeStyle = `rgba(201, 16, 16, ${alpha})`;
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 6]);
    ctx.beginPath();
    ctx.moveTo(mapX(x[0]), mapY(y_upper[0] - cpSuction[0] * pct));
    for (let i = 1; i < n; i++) {
      ctx.lineTo(mapX(x[i]), mapY(y_upper[i] - cpSuction[i] * pct));
    }
    ctx.stroke();
  });

  // ---- COMPRESSION REGION (Maroon — dotted boundary, distinct from suction) ----
  // Gradient fill
  const gradCompression = ctx.createLinearGradient(0, mapY(0.0), 0, mapY(-peakVal));
  gradCompression.addColorStop(0, 'rgba(156, 28, 54, 0.22)');
  gradCompression.addColorStop(0.5, 'rgba(156, 28, 54, 0.10)');
  gradCompression.addColorStop(1, 'rgba(156, 28, 54, 0.02)');

  ctx.fillStyle = gradCompression;
  ctx.beginPath();
  ctx.moveTo(mapX(x[0]), mapY(y_lower[0]));
  for (let i = 0; i < n; i++) {
    ctx.lineTo(mapX(x[i]), mapY(y_lower[i] - cpCompression[i]));
  }
  for (let i = n - 1; i >= 0; i--) {
    ctx.lineTo(mapX(x[i]), mapY(y_lower[i]));
  }
  ctx.closePath();
  ctx.fill();

  // Compression boundary line (maroon — dotted, visually distinct from suction dashes)
  ctx.strokeStyle = 'rgba(156, 28, 54, 0.9)';
  ctx.setLineDash([2, 3]);
  ctx.lineWidth = 1.8;
  ctx.beginPath();
  ctx.moveTo(mapX(x[0]), mapY(y_lower[0] - cpCompression[0]));
  for (let i = 1; i < n; i++) {
    ctx.lineTo(mapX(x[i]), mapY(y_lower[i] - cpCompression[i]));
  }
  ctx.stroke();

  // 5 contour levels inside compression region (maroon, progressively stronger)
  const compressionLevels = [0.15, 0.30, 0.50, 0.70, 0.85];
  compressionLevels.forEach((pct, idx) => {
    const alpha = 0.20 + idx * 0.07;
    ctx.strokeStyle = `rgba(156, 28, 54, ${alpha})`;
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 5]);
    ctx.beginPath();
    ctx.moveTo(mapX(x[0]), mapY(y_lower[0] - cpCompression[0] * pct));
    for (let i = 1; i < n; i++) {
      ctx.lineTo(mapX(x[i]), mapY(y_lower[i] - cpCompression[i] * pct));
    }
    ctx.stroke();
  });

  // Reset line dash
  ctx.setLineDash([]);

  // ---- CAMBER LINE (cotton-alpha dashed line through profile midpoint) ----
  ctx.strokeStyle = 'rgba(237, 235, 221, 0.20)';
  ctx.setLineDash([4, 6]);
  ctx.lineWidth = 0.8;
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const camberY = (y_upper[i] + y_lower[i]) / 2;
    if (i === 0) ctx.moveTo(mapX(x[i]), mapY(camberY));
    else ctx.lineTo(mapX(x[i]), mapY(camberY));
  }
  ctx.stroke();
  ctx.setLineDash([]);

  // ---- THICKNESS MARKERS at 25% and 50% chord ----
  [0.25, 0.50].forEach(xc => {
    let closest = 0;
    let closestDist = Infinity;
    for (let i = 0; i < n; i++) {
      if (Math.abs(x[i] - xc) < closestDist) {
        closestDist = Math.abs(x[i] - xc);
        closest = i;
      }
    }
    const px = mapX(x[closest]);
    const pyTop = mapY(y_upper[closest]);
    const pyBot = mapY(y_lower[closest]);
    const thick = y_upper[closest] - y_lower[closest];

    // Vertical thickness line
    ctx.strokeStyle = 'rgba(237, 235, 221, 0.25)';
    ctx.lineWidth = 0.8;
    ctx.setLineDash([2, 2]);
    ctx.beginPath();
    ctx.moveTo(px, pyTop);
    ctx.lineTo(px, pyBot);
    ctx.stroke();
    ctx.setLineDash([]);

    // Top and bottom tick marks
    ctx.strokeStyle = 'rgba(237, 235, 221, 0.40)';
    ctx.lineWidth = 0.8;
    ctx.beginPath();
    ctx.moveTo(px - 3, pyTop);
    ctx.lineTo(px + 3, pyTop);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(px - 3, pyBot);
    ctx.lineTo(px + 3, pyBot);
    ctx.stroke();

    // Thickness label
    ctx.font = '8px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    ctx.fillStyle = 'rgba(237, 235, 221, 0.45)';
    ctx.fillText(`t=${(thick * 100).toFixed(1)}%`, px + 5, (pyTop + pyBot) / 2 + 3);
  });

  // 2. Draw Solid Airfoil Coordinate Profile
  ctx.fillStyle = 'rgba(27, 23, 23, 0.95)';
  ctx.strokeStyle = '#EDEBDD';
  ctx.lineWidth = 2;

  ctx.beginPath();
  ctx.moveTo(mapX(x[0]), mapY(y_upper[0]));
  for (let i = 1; i < n; i++) {
    ctx.lineTo(mapX(x[i]), mapY(y_upper[i]));
  }
  for (let i = n - 1; i >= 0; i--) {
    ctx.lineTo(mapX(x[i]), mapY(y_lower[i]));
  }
  ctx.closePath();
  ctx.fill();
  ctx.stroke();

  // 3. Highlight leading and trailing edge points
  ctx.fillStyle = '#EDEBDD';
  ctx.beginPath();
  ctx.arc(mapX(x[0]), mapY(y_upper[0]), 3.5, 0, 2 * Math.PI);
  ctx.fill();

  ctx.beginPath();
  ctx.arc(mapX(x[n - 1]), mapY(y_upper[n - 1]), 2.5, 0, 2 * Math.PI);
  ctx.fill();
}

// --------------------------------------------------------------------------
// 📁 File Upload Logic
// --------------------------------------------------------------------------
let uploadedFile = null;

fileUpload.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (file) {
    uploadedFile = file;
    uploadStatus.innerText = `Loaded: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
    uploadStatus.style.display = 'block';
    btnRunPrediction.disabled = false;
    logTerminal(`Loaded local geometry coordinates: ${file.name}`, 'success');
  } else {
    uploadedFile = null;
    uploadStatus.style.display = 'none';
    btnRunPrediction.disabled = true;
  }
});

// --------------------------------------------------------------------------
// 🚀 API Submissions
// --------------------------------------------------------------------------

// Predict Action
btnRunPrediction.addEventListener('click', async () => {
  if (!uploadedFile) return;

  btnRunPrediction.disabled = true;
  logTerminal('Starting forward CFD surrogate simulation...', 'info');

  // Trigger loading screen
  isLoading = true;
  drawViewport();

  const formData = new FormData();
  formData.append('file', uploadedFile);
  formData.append('re', document.getElementById('rePredictSlider').value * 1e6);
  formData.append('mach', document.getElementById('machPredictSlider').value);

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      body: formData
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.message || 'CFD surrogate solver engine runtime error.');
    }

    const data = await res.json();
    logTerminal('Forward prediction completed successfully.', 'success');

    // Update readouts
    updateMetricReadouts(data.predictions, data.uncertainty, 'predict');

    // Set active geometry
    activeGeometry = {
      x: data.geometry.x,
      y_upper: data.geometry.y_upper,
      y_lower: data.geometry.y_lower,
      cl: data.predictions.ClMax
    };

  } catch (err) {
    logTerminal(err.message, 'error');
  } finally {
    isLoading = false;
    btnRunPrediction.disabled = false;
    drawViewport();
  }
});

// Optimize Action
btnRunOptimization.addEventListener('click', async () => {
  btnRunOptimization.disabled = true;
  candidateSelect.disabled = true;
  btnExportDat.disabled = true;
  logTerminal('Initializing multi-restart latent-space search...', 'warning');

  // Trigger loading screen
  isLoading = true;
  drawViewport();

  const payload = {
    ldmax: parseFloat(document.getElementById('targetLdSlider').value),
    clmax: parseFloat(document.getElementById('targetClSlider').value),
    cdmin: parseFloat(document.getElementById('targetCdSlider').value),
    re: parseFloat(document.getElementById('reOptimizeSlider').value) * 1e6,
    mach: parseFloat(document.getElementById('machOptimizeSlider').value)
  };

  try {
    const res = await fetch('/api/optimize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.message || 'Optimization solver runtime execution error.');
    }

    const data = await res.json();
    activeCandidates = data.candidates;

    logTerminal(`Search completed. Found ${activeCandidates.length} convergent profiles.`, 'success');

    // Populate select candidates list
    candidateSelect.innerHTML = '';
    activeCandidates.forEach((c, idx) => {
      const opt = document.createElement('option');
      opt.value = idx;
      opt.innerText = `${c.label} (Obj: ${c.objective.toFixed(4)})`;
      candidateSelect.appendChild(opt);
    });

    candidateSelect.disabled = false;
    candidateSelect.selectedIndex = 0;
    
    // Select first candidate
    selectCandidate(0);

  } catch (err) {
    logTerminal(err.message, 'error');
  } finally {
    isLoading = false;
    btnRunOptimization.disabled = false;
    drawViewport();
  }
});

// Handle Select Candidate
candidateSelect.addEventListener('change', (e) => {
  if (e.target.value !== '') {
    selectCandidate(parseInt(e.target.value));
  }
});

function selectCandidate(idx) {
  const c = activeCandidates[idx];
  if (!c) return;

  logTerminal(`Rendering candidate profile: ${c.label}`, 'info');

  updateMetricReadouts(c.predictions, c.uncertainty, 'optimize');

  activeGeometry = {
    x: c.geometry.x,
    y_upper: c.geometry.y_upper,
    y_lower: c.geometry.y_lower,
    cl: c.predictions.ClMax
  };

  drawViewport();
  btnExportDat.disabled = false;
}

// Update UI metrics helper
function updateMetricReadouts(preds, unc, prefix) {
  const isForward = prefix === 'predict';
  const ldId = isForward ? 'readoutPredictLd' : 'readoutOptimizeLd';
  const clId = isForward ? 'readoutPredictCl' : 'readoutOptimizeCl';
  const cdId = isForward ? 'readoutPredictCd' : 'readoutOptimizeCd';

  const barLdId = isForward ? 'barPredictLd' : 'barOptimizeLd';
  const barClId = isForward ? 'barPredictCl' : 'barOptimizeCl';
  const barCdId = isForward ? 'barPredictCd' : 'barOptimizeCd';

  document.getElementById(ldId).innerText = preds.LDMax.toFixed(2);
  document.getElementById(clId).innerText = preds.ClMax.toFixed(4);
  document.getElementById(cdId).innerText = preds.CdMin.toFixed(5);

  const computeConfidence = (std, scale) => Math.max(10, Math.min(100, Math.round((1.0 - (std / scale)) * 100)));

  const confLd = computeConfidence(unc.LDMax_std, 12.0);
  const confCl = computeConfidence(unc.ClMax_std, 0.15);
  const confCd = computeConfidence(unc.CdMin_std, 0.005);

  document.getElementById(barLdId).style.width = `${confLd}%`;
  document.getElementById(barClId).style.width = `${confCl}%`;
  document.getElementById(barCdId).style.width = `${confCd}%`;
}

// --------------------------------------------------------------------------
// 📥 Export .dat Geometry Coordinates File
// --------------------------------------------------------------------------
btnExportDat.addEventListener('click', () => {
  if (!activeGeometry) return;

  const { x, y_upper, y_lower } = activeGeometry;
  const n = x.length;

  const x_upper_rev = x.slice().reverse();
  const y_upper_rev = y_upper.slice().reverse();

  let datContent = '';
  for (let i = 0; i < n; i++) {
    datContent += `${x_upper_rev[i].toFixed(8)}\t${y_upper_rev[i].toFixed(8)}\n`;
  }
  for (let i = 1; i < n; i++) {
    datContent += `${x[i].toFixed(8)}\t${y_lower[i].toFixed(8)}\n`;
  }

  const blob = new Blob([datContent], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  
  const selectedIdx = candidateSelect.value;
  const label = selectedIdx !== '' ? activeCandidates[selectedIdx].label : 'optimized';
  a.download = `aeroml_design_${label}.dat`;
  
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
  
  logTerminal(`Exported airfoil coordinates: ${a.download}`, 'success');
});
