// -*- coding: utf-8 -*-
/**
 * @file landing.js
 * @description Real-time canvas particle flow field simulation and landing telemetry calculations
 * @module frontend
 */

// Selig format coordinates for SD7062 airfoil (TE -> upper -> LE -> lower -> TE)
const RAW_COORDS = [
  [0.99973, 0.00327], [0.99625, 0.00384], [0.98607, 0.00569], [0.96977, 0.00898], [0.94791, 0.01363],
  [0.92100, 0.01942], [0.88940, 0.02616], [0.85354, 0.03376], [0.81400, 0.04212], [0.77139, 0.05105],
  [0.72635, 0.06029], [0.67946, 0.06953], [0.63127, 0.07844], [0.58227, 0.08674], [0.53295, 0.09416],
  [0.48375, 0.10050], [0.43513, 0.10556], [0.38752, 0.10919], [0.34132, 0.11128], [0.29693, 0.11173],
  [0.25469, 0.11049], [0.21494, 0.10755], [0.17798, 0.10292], [0.14405, 0.09667], [0.11336, 0.08889],
  [0.08610, 0.07974], [0.06242, 0.06943], [0.04245, 0.05818], [0.02623, 0.04627], [0.01383, 0.03408],
  [0.00535, 0.02198], [0.00076, 0.01038], [0.00000, 0.00000],
  [0.00000, 0.00000], [0.00398, -0.00788], [0.01387, -0.01374], [0.02964, -0.01878], [0.05072, -0.02296],
  [0.07683, -0.02621], [0.10771, -0.02852], [0.14301, -0.02992], [0.18238, -0.03040], [0.22543, -0.03000],
  [0.27172, -0.02879], [0.32078, -0.02683], [0.37211, -0.02420], [0.42520, -0.02099], [0.47952, -0.01737],
  [0.53450, -0.01353], [0.58948, -0.00968], [0.64376, -0.00600], [0.69665, -0.00266], [0.74741, 0.00020],
  [0.79530, 0.00248], [0.83963, 0.00414], [0.87971, 0.00516], [0.91491, 0.00557], [0.94464, 0.00543],
  [0.96837, 0.00490], [0.98566, 0.00419], [0.99619, 0.00354], [0.99974, 0.00327]
];

const upperPts = RAW_COORDS.slice(0, 33).slice().reverse(); // asc 0 -> 1
const lowerPts = RAW_COORDS.slice(33); // asc 0 -> 1

function interp(pts, x) {
  x = Math.max(0, Math.min(1, x));
  for (let i = 0; i < pts.length - 1; i++) {
    const [x0, y0] = pts[i], [x1, y1] = pts[i + 1];
    if (x >= x0 && x <= x1) {
      const t = x1 === x0 ? 0 : (x - x0) / (x1 - x0);
      return y0 + t * (y1 - y0);
    }
  }
  return pts[pts.length - 1][1];
}

const upperY = x => interp(upperPts, x);
const lowerY = x => interp(lowerPts, x);
const camberY = x => (upperY(x) + lowerY(x)) / 2;
const thickness = x => upperY(x) - lowerY(x);
const maxThickness = Math.max(...upperPts.map(p => thickness(p[0])));

const canvas = document.getElementById('field');
const ctx = canvas.getContext('2d');
let W, H, chordPx, airfoilLeft, airfoilCenterY;

function resize() {
  const rect = canvas.parentElement.getBoundingClientRect();
  W = canvas.width = rect.width;
  H = canvas.height = rect.height;
  chordPx = Math.min(W * 0.52, H * 1.3);
  airfoilLeft = W * 0.28;
  airfoilCenterY = H * 0.50;
  ctx.fillStyle = '#1B1717';
  ctx.fillRect(0, 0, W, H);
}
window.addEventListener('resize', resize);
resize();

// --- Particles Simulation Setup ---
const N_PARTICLES = 400;
const particles = [];

function spawnParticle(staggered) {
  const sign = Math.random() < 0.5 ? -1 : 1;
  const magnitude = 0.10 + Math.random() * 0.42;
  const lane = sign * magnitude;
  return {
    lane,
    x: staggered ? Math.random() * W : -Math.random() * 200,
    y: airfoilCenterY + lane * chordPx * 0.55,
    speed: 1.8 + Math.random() * 1.2,
    wobbleSeed: Math.random() * 1000,
    isUpper: lane > 0
  };
}

for (let i = 0; i < N_PARTICLES; i++) {
  particles.push(spawnParticle(true));
}

function drawGrid() {
  ctx.strokeStyle = 'rgba(237, 235, 221, 0.06)';
  ctx.lineWidth = 0.5;
  const step = 48;
  for (let x = 0; x < W; x += step) {
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
  }
  for (let y = 0; y < H; y += step) {
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
  }
}

function drawAirfoil() {
  ctx.save();
  ctx.beginPath();
  upperPts.forEach((p, i) => {
    const px = airfoilLeft + p[0] * chordPx;
    const py = airfoilCenterY - p[1] * chordPx;
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  });
  lowerPts.slice().reverse().forEach(p => {
    const px = airfoilLeft + p[0] * chordPx;
    const py = airfoilCenterY - p[1] * chordPx;
    ctx.lineTo(px, py);
  });
  ctx.closePath();
  ctx.fillStyle = 'rgba(84, 1, 1, 0.28)'; // Maroon alpha body
  ctx.fill();
  ctx.strokeStyle = '#EDEBDD'; // Cotton border outline
  ctx.lineWidth = 1.4;
  ctx.stroke();
  ctx.restore();

  // Chord baseline + coordinate ticks (blueprint / drafting-paper motif)
  ctx.strokeStyle = 'rgba(237, 235, 221, 0.22)';
  ctx.lineWidth = 0.5;
  ctx.beginPath();
  ctx.moveTo(airfoilLeft - chordPx * 0.15, airfoilCenterY);
  ctx.lineTo(airfoilLeft + chordPx * 1.15, airfoilCenterY);
  ctx.stroke();

  ctx.font = '10px "JetBrains Mono", monospace';
  ctx.fillStyle = 'rgba(237, 235, 221, 0.6)';
  for (let t = 0; t <= 1.0001; t += 0.2) {
    const px = airfoilLeft + t * chordPx;
    ctx.beginPath();
    ctx.moveTo(px, airfoilCenterY + 6);
    ctx.lineTo(px, airfoilCenterY + 12);
    ctx.stroke();
    ctx.fillText(t.toFixed(1), px - 8, airfoilCenterY + 24);
  }
}

let frameCount = 0;

function step() {
  frameCount++;
  ctx.fillStyle = 'rgba(27, 23, 23, 0.08)'; // Trail effect
  ctx.fillRect(0, 0, W, H);
  drawGrid();

  for (const p of particles) {
    const cx = (p.x - airfoilLeft) / chordPx;
    let targetY;
    if (cx >= 0 && cx <= 1) {
      const spread = 1 + 2.4 * (thickness(cx) / maxThickness);
      targetY = airfoilCenterY - camberY(cx) * chordPx + p.lane * chordPx * 0.55 * spread;
    } else {
      targetY = airfoilCenterY + p.lane * chordPx * 0.55;
    }
    const wobble = Math.sin(frameCount * 0.02 + p.wobbleSeed) * 0.6;
    p.y += (targetY - p.y) * 0.08 + wobble * 0.15;
    p.x += p.speed;

    const insideBody = cx >= 0 && cx <= 1 &&
      p.y > airfoilCenterY - upperY(cx) * chordPx &&
      p.y < airfoilCenterY - lowerY(cx) * chordPx;

    if (!insideBody) {
      const nearBody = cx >= -0.05 && cx <= 1.05;
      // Adapted flow line colors: Suction = Cherry Red, Pressure = Maroon
      const color = p.isUpper ? [201, 16, 16] : [156, 28, 54];
      const alpha = nearBody ? 0.90 : 0.45;
      ctx.strokeStyle = `rgba(${color[0]},${color[1]},${color[2]},${alpha})`;
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.moveTo(p.x - p.speed * 2.5, p.y);
      ctx.lineTo(p.x, p.y);
      ctx.stroke();
    }

    if (p.x > W + 20) {
      Object.assign(p, spawnParticle(false));
    }
  }

  drawAirfoil();
  updateSensors();

  requestAnimationFrame(step);
}

// Telemetry fluctuations
const sensorElRe = document.getElementById('telemetryRe');
const sensorElMach = document.getElementById('telemetryMach');
const sensorElCl = document.getElementById('telemetryCl');
const sensorElCd = document.getElementById('telemetryCd');
const sensorElLd = document.getElementById('telemetryLd');

function updateSensors() {
  if (frameCount % 10 === 0) {
    const reBase = 3000000;
    const reDelta = Math.floor((Math.random() - 0.5) * 4500);
    if (sensorElRe) {
      sensorElRe.innerText = (reBase + reDelta).toLocaleString();
    }

    const machBase = 0.150;
    const machDelta = (Math.random() - 0.5) * 0.002;
    if (sensorElMach) {
      sensorElMach.innerText = (machBase + machDelta).toFixed(4);
    }

    const clBase = 1.4284;
    const clDelta = (Math.random() - 0.5) * 0.006;
    const clVal = clBase + clDelta;
    if (sensorElCl) {
      sensorElCl.innerText = clVal.toFixed(4);
    }

    const cdBase = 0.00842;
    const cdDelta = (Math.random() - 0.5) * 0.00008;
    const cdVal = cdBase + cdDelta;
    if (sensorElCd) {
      sensorElCd.innerText = cdVal.toFixed(5);
    }

    if (sensorElLd) {
      sensorElLd.innerText = (clVal / cdVal).toFixed(2);
    }
  }
}

// Repository Atlas Node click handlers
const nodeData = {
  root: {
    title: "AeroML/",
    path: "./ (Project Root)",
    purpose: "The core orchestration root of the AeroML project housing optimization modules, data split pipelines, and flow visualizer interfaces.",
    modules: "README.md, requirements.txt, LICENSE, context.md, pyproject.toml",
    size: "1.2 MB",
    status: "active"
  },
  data: {
    title: "Data_Cache/",
    path: "Data_Cache/",
    purpose: "Preprocessed training datasets, test splits, and split manifest arrays.",
    modules: "aeroml_xfoil_n9_dataset.npz, split_manifest.csv",
    size: "45.2 MB",
    status: "cached"
  },
  outputs: {
    title: "Forward_outputs/",
    path: "Forward_outputs/",
    purpose: "Serialized TensorFlow Keras neural network model weights, hyperparameters, and logs from training runs.",
    modules: "seed_42.keras, seed_101.keras, seed_2023.keras, metrics.json",
    size: "84.1 MB",
    status: "cached"
  },
  src: {
    title: "src/",
    path: "src/aeroml/",
    purpose: "Main source codebase container holding the neural network architectures, solvers, and prediction pipelines.",
    modules: "forward.py, reverse.py, features.py, models.py, train.py, evaluate.py",
    size: "184 KB",
    status: "active"
  },
  scripts: {
    title: "scripts/",
    path: "scripts/",
    purpose: "Command-line execution engines for retraining the ensemble models and running standalone design searches.",
    modules: "train_forward.py, run_reverse.py, api_bridge.py",
    size: "24 KB",
    status: "stable"
  },
  tests: {
    title: "tests/",
    path: "tests/",
    purpose: "Verification utilities to validate data drift and test prediction accuracy on local airfoil datasets.",
    modules: "test_forward_drift.py, test_data/",
    size: "12 KB",
    status: "stable"
  },
  backend: {
    title: "backend/",
    path: "backend/",
    purpose: "High-performance Hono static server and Zod rate-limited gateway bridging the front-end requests to the neural network cores.",
    modules: "server.ts, package.json, tsconfig.json",
    size: "18 KB",
    status: "active"
  },
  frontend: {
    title: "frontend/",
    path: "frontend/",
    purpose: "Interactive user interfaces containing flow visualizers, dither background controllers, and drafting canvas viewports.",
    modules: "index.html, workbench.html, style.css, workbench.js, landing.js, dither-bg.js",
    size: "135 KB",
    status: "active"
  }
};

function setupAtlasInspector() {
  const nodes = document.querySelectorAll('.map-node');
  const inspectTitle = document.getElementById('inspectTitle');
  const inspectPath = document.getElementById('inspectPath');
  const inspectPurpose = document.getElementById('inspectPurpose');
  const inspectModules = document.getElementById('inspectModules');
  const inspectSize = document.getElementById('inspectSize');
  const inspectStatus = document.getElementById('inspectStatus');

  if (!inspectTitle) return;

  nodes.forEach(node => {
    node.addEventListener('click', () => {
      nodes.forEach(n => n.classList.remove('active'));
      node.classList.add('active');

      const dataKey = node.getAttribute('data-node');
      const info = nodeData[dataKey];
      if (info) {
        inspectTitle.innerText = info.title;
        inspectPath.innerText = info.path;
        inspectPurpose.innerText = info.purpose;
        inspectModules.innerText = info.modules;
        inspectSize.innerText = info.size;
        inspectStatus.innerText = info.status;

        // Reset and assign tag class mapping
        inspectStatus.className = 'inspector-badge';
        if (info.status === 'active') {
          inspectStatus.classList.add('active');
        } else if (info.status === 'stable') {
          inspectStatus.classList.add('stable');
        } else if (info.status === 'cached') {
          inspectStatus.classList.add('cached');
        }

        const fields = [inspectTitle, inspectPath, inspectPurpose, inspectModules, inspectSize, inspectStatus];
        fields.forEach(el => {
          el.classList.remove('animate-fade-in');
          void el.offsetWidth;
          el.classList.add('animate-fade-in');
        });
      }
    });
  });
}

// Start simulation on load
window.addEventListener('load', () => {
  step();
  setupAtlasInspector();
});
