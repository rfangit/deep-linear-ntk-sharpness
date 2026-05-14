// ntk-app.js  — NTK widget logic

// ─────────────────────────────────────────────────────
// 1. Seeded PRNG (mulberry32)
// ─────────────────────────────────────────────────────
function mulberry32(seed) {
  return function () {
    seed |= 0; seed = seed + 0x6D2B79F5 | 0;
    let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

function randn(rng) {
  // Box-Muller
  const u = 1 - rng(), v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// ─────────────────────────────────────────────────────
// 2. Small linear-algebra helpers (2×2 and 4-vector)
// ─────────────────────────────────────────────────────

// Random 2×2 orthogonal matrix
function randOrth2(rng) {
  const a = randn(rng), b = randn(rng);
  const n = Math.hypot(a, b);
  const c = a / n, s = b / n;
  return [[c, -s], [s, c]];  // columns are orthonormal
}

// mat-vec: M (2×2) times v (len 2)
function mv2(M, v) {
  return [M[0][0] * v[0] + M[0][1] * v[1],
          M[1][0] * v[0] + M[1][1] * v[1]];
}

// dot product
function dot(a, b) { return a.reduce((s, x, i) => s + x * b[i], 0); }

// ─────────────────────────────────────────────────────
// 3. NTK for a 2-layer network: W2 W1 x
//    W1: m×d,  W2: n×m
//    Theta(x, x') = <x,x'> W2 W2^T  +  <W1 x, W1 x'> I_n
//    Returns n×n matrix as flat array [row-major]
// ─────────────────────────────────────────────────────
function computeNTK(W1, W2, x, xp) {
  // W1: m×d arrays, W2: n×m arrays
  const d = x.length, m = W1.length, n = W2.length;
  // <x, x'>
  const xxp = dot(x, xp);
  // W2 W2^T  (n×n)
  const W2W2T = Array.from({length: n}, (_, i) =>
    Array.from({length: n}, (__, j) => dot(W2[i], W2[j]))
  );
  // W1 x  and  W1 x'
  const W1x  = W1.map(row => dot(row, x));
  const W1xp = W1.map(row => dot(row, xp));
  const W1xdotW1xp = dot(W1x, W1xp);
  // Theta = <x,x'> W2W2^T + <W1x, W1x'> I
  return Array.from({length: n}, (_, i) =>
    Array.from({length: n}, (__, j) =>
      xxp * W2W2T[i][j] + (i === j ? W1xdotW1xp : 0)
    )
  );
}

// Build W1, W2 from seed (2D in, 2D out, hidden 2)
function buildRandomWeights(seed) {
  const rng = mulberry32(seed * 1234567 + 1);
  const m = 2, d = 2, n = 2;
  const W1 = Array.from({length: m}, () => Array.from({length: d}, () => randn(rng)));
  const W2 = Array.from({length: n}, () => Array.from({length: m}, () => randn(rng)));
  return {W1, W2};
}

// Build analytic weights from U, Sigma, V, O
// W1 = O^T Sigma^{1/2} V^T,  W2 = U Sigma^{1/2} O
function buildAnalyticWeights(U, sigma, V, O) {
  // All 2×2
  // V^T: rows are V's columns, i.e. V^T[i][j] = V[j][i]
  const sqS = sigma.map(Math.sqrt);
  // W1[i][j] = sum_k O^T[i][k] * sqS[k] * V^T[k][j]
  //          = sum_k O[k][i]   * sqS[k] * V[j][k]
  const W1 = [[0,0],[0,0]];
  for (let i = 0; i < 2; i++)
    for (let j = 0; j < 2; j++)
      for (let k = 0; k < 2; k++)
        W1[i][j] += O[k][i] * sqS[k] * V[j][k];
  // W2[i][j] = sum_k U[i][k] * sqS[k] * O[k][j]
  const W2 = [[0,0],[0,0]];
  for (let i = 0; i < 2; i++)
    for (let j = 0; j < 2; j++)
      for (let k = 0; k < 2; k++)
        W2[i][j] += U[i][k] * sqS[k] * O[k][j];
  return {W1, W2};
}

// ─────────────────────────────────────────────────────
// 4. Saxe analytic singular value trajectory
//    sigma_i(t) = s_i / (1 + (s_i/sigma0 - 1) * exp(-2*s_i*t))
// ─────────────────────────────────────────────────────
const SIGMA0 = 0.001;
const T_MAX_RAW = 12; // real time units mapped to slider 0..1000

function saxeSigma(s, t) {
  if (s < 1e-12) return 0;
  const ratio = s / SIGMA0;
  return s / (1 + (ratio - 1) * Math.exp(-2 * s * t));
}

function sliderToTime(v) { return (v / 1000) * T_MAX_RAW; }

// Loss = sum_i (s_i - sigma_i(t))^2  (whitened, per mode)
function saxeLoss(sv, sigmas) {
  return sv.reduce((acc, s, i) => acc + 0.5 * (s - sigmas[i]) ** 2, 0);
}

// ─────────────────────────────────────────────────────
// 5. NTK matrix renderer
// ─────────────────────────────────────────────────────
// Cell coloring: white at val=0, blending to a soft pastel at val=±maxAbs.
// Saturated endpoints (#0000ff, #ff0000) make the cell text hard to read,
// so we use desaturated endpoints. A gamma > 1 on |t| pushes mid-range
// values toward white, keeping color reserved for genuinely large entries.
function heatColor(val, maxAbs) {
  if (maxAbs < 1e-12) return '#fff';
  const t = Math.max(-1, Math.min(1, val / maxAbs));
  const mag = Math.pow(Math.abs(t), 1.3); // gamma softens the middle
  // Endpoint colors (kept pastel for text legibility).
  const posEnd = [80, 130, 220];  // soft blue
  const negEnd = [220, 100, 100]; // soft red
  const end = t >= 0 ? posEnd : negEnd;
  // Linear interpolate from white -> endpoint by `mag`.
  const r = Math.round(255 + (end[0] - 255) * mag);
  const g = Math.round(255 + (end[1] - 255) * mag);
  const b = Math.round(255 + (end[2] - 255) * mag);
  return `rgb(${r},${g},${b})`;
}

// blocks: [{label: "Θ(x,x)", matrix: [[…],[…]]}, …]
function renderNTKGrid(container, blocks, inputLabels, options = {}) {
  // blocks is an array of {label, matrix (n×n)}
  // Lay them out in a 2×2 grid for our 2 inputs × 2 inputs
  //
  // options.cellW / options.cellH: pixel size of each NTK matrix cell.
  // Defaults match widget 1 (compact). Widget 2 (time-evolution) passes
  // larger values so the matrix fills the height of its plot column.
  const cellW    = options.cellW    ?? 80;
  const cellH    = options.cellH    ?? 52;
  const fontSize = options.fontSize ?? 13;
  const labelFontSize = options.labelFontSize ?? 12;

  container.innerHTML = '';
  const allVals = blocks.flatMap(b => b.matrix.flat());
  const maxAbs = Math.max(...allVals.map(Math.abs), 1e-9);

  const wrapper = document.createElement('div');
  wrapper.style.cssText = 'display:inline-block;';

  const n = blocks[0].matrix.length; // output dims

  // Column headers
  const colHeaderRow = document.createElement('div');
  colHeaderRow.style.cssText = 'display:flex; align-items:flex-end; margin-bottom:4px; margin-left:70px;';
  inputLabels.forEach(lbl => {
    const h = document.createElement('div');
    h.style.cssText = `width:${n * cellW}px; text-align:center; font-size:${labelFontSize}px; color:#888; font-family:Georgia,serif;`;
    h.innerHTML = `$x' = ${lbl}$`;
    colHeaderRow.appendChild(h);
  });
  wrapper.appendChild(colHeaderRow);

  // Row of blocks
  inputLabels.forEach((rowLbl, ri) => {
    const row = document.createElement('div');
    row.style.cssText = 'display:flex; align-items:center; margin-bottom:2px;';

    // Row label
    const rl = document.createElement('div');
    rl.style.cssText = `width:70px; text-align:right; padding-right:10px; font-size:${labelFontSize}px; color:#888; font-family:Georgia,serif;`;
    rl.innerHTML = `$x = ${rowLbl}$`;
    row.appendChild(rl);

    inputLabels.forEach((colLbl, ci) => {
      const blk = blocks[ri * inputLabels.length + ci];
      const tbl = document.createElement('table');
      tbl.style.cssText = 'border-collapse:collapse; margin-right:2px;';
      for (let i = 0; i < n; i++) {
        const tr = document.createElement('tr');
        for (let j = 0; j < n; j++) {
          const val = blk.matrix[i][j];
          const td = document.createElement('td');
          td.style.cssText = `
            width:${cellW}px; height:${cellH}px; text-align:center; vertical-align:middle;
            border:1px solid #ddd; font-family:'Courier New',monospace;
            font-size:${fontSize}px; font-weight:600;
            color:#222;
            background:${heatColor(val, maxAbs)};
            transition: background 0.3s;
          `;
          td.textContent = val.toFixed(3);
          tr.appendChild(td);
        }
        tbl.appendChild(tr);
      }
      row.appendChild(tbl);
    });
    wrapper.appendChild(row);
  });

  // Output dimension label
  const dimNote = document.createElement('div');
  dimNote.style.cssText = 'font-size:11px; color:#bbb; text-align:center; margin-top:6px;';
  dimNote.textContent = `Each block: ${n}×${n} matrix over output dimensions`;
  wrapper.appendChild(dimNote);

  container.appendChild(wrapper);

  // Trigger MathJax on newly inserted content
  if (window.MathJax && window.mathJaxReady) {
    MathJax.typesetPromise([container]).catch(() => {});
  }
}

// ─────────────────────────────────────────────────────
// 6. Simple canvas 2D plotter
// ─────────────────────────────────────────────────────
function drawPlot(canvas, series, opts = {}) {
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth, H = canvas.clientHeight;
  canvas.width = W * dpr; canvas.height = H * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);

  const pad = {l: 42, r: 12, t: 16, b: 30};
  const pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, W, H);

  if (!series.length || !series[0].ys.length) return;

  const allY = series.flatMap(s => s.ys).filter(isFinite);
  const allX = series[0].xs;
  const xMin = allX[0], xMax = allX[allX.length - 1];
  let yMin = opts.yMin ?? Math.min(0, ...allY);
  let yMax = opts.yMax ?? Math.max(...allY) * 1.08;
  if (yMax <= yMin) yMax = yMin + 1;

  const px = x => pad.l + (x - xMin) / (xMax - xMin) * pw;
  const py = y => pad.t + (1 - (y - yMin) / (yMax - yMin)) * ph;

  // Grid & axes
  ctx.strokeStyle = '#eee'; ctx.lineWidth = 1;
  const nYTicks = 4;
  for (let i = 0; i <= nYTicks; i++) {
    const y = yMin + (yMax - yMin) * i / nYTicks;
    const yp = py(y);
    ctx.beginPath(); ctx.moveTo(pad.l, yp); ctx.lineTo(pad.l + pw, yp); ctx.stroke();
    ctx.fillStyle = '#aaa'; ctx.font = `10px sans-serif`; ctx.textAlign = 'right';
    ctx.fillText(y < 1 ? y.toFixed(2) : y.toFixed(1), pad.l - 4, yp + 3);
  }
  const nXTicks = 4;
  for (let i = 0; i <= nXTicks; i++) {
    const x = xMin + (xMax - xMin) * i / nXTicks;
    const xp = px(x);
    ctx.beginPath(); ctx.moveTo(xp, pad.t); ctx.lineTo(xp, pad.t + ph); ctx.stroke();
    ctx.fillStyle = '#aaa'; ctx.font = `10px sans-serif`; ctx.textAlign = 'center';
    ctx.fillText(x < 1 ? x.toFixed(1) : x.toFixed(0), xp, pad.t + ph + 14);
  }

  // Title
  if (opts.title) {
    ctx.fillStyle = '#888'; ctx.font = '11px sans-serif'; ctx.textAlign = 'left';
    ctx.fillText(opts.title, pad.l + 4, pad.t + 11);
  }

  // Marker line (current t)
  if (opts.markerX !== undefined) {
    const mx = px(opts.markerX);
    ctx.save();
    ctx.strokeStyle = 'rgba(0,0,0,0.3)'; ctx.lineWidth = 1.5;
    ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(mx, pad.t); ctx.lineTo(mx, pad.t + ph); ctx.stroke();
    ctx.restore();
  }

  // Series
  series.forEach(s => {
    ctx.strokeStyle = s.color || '#3366cc';
    ctx.lineWidth = s.lineWidth || 1.8;
    ctx.setLineDash(s.dash || []);
    ctx.beginPath();
    s.xs.forEach((x, i) => {
      const xp = px(x), yp = py(s.ys[i]);
      i === 0 ? ctx.moveTo(xp, yp) : ctx.lineTo(xp, yp);
    });
    ctx.stroke();
    ctx.setLineDash([]);
  });
}

// ─────────────────────────────────────────────────────
// 7. Bar chart renderer
// ─────────────────────────────────────────────────────
const BAR_COLORS = [
  '#4e79a7', '#f28e2b', '#e15759', '#76b7b2',
  '#59a14f', '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac'
];

function drawBars(canvas, values, opts = {}) {
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth, H = canvas.clientHeight;
  canvas.width = W * dpr; canvas.height = H * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);

  const pad = {l: 44, r: 12, t: 24, b: 28};
  const pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, W, H);

  const n = values.length;
  const yMax = opts.yMax ?? (Math.max(...values, 1e-9) * 1.15);
  const barW = pw / (n * 1.5);
  const spacing = pw / n;

  // Y axis ticks
  const nT = 4;
  for (let i = 0; i <= nT; i++) {
    const y = yMax * i / nT;
    const yp = pad.t + (1 - i / nT) * ph;
    ctx.strokeStyle = '#eee'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad.l, yp); ctx.lineTo(pad.l + pw, yp); ctx.stroke();
    ctx.fillStyle = '#aaa'; ctx.font = '10px sans-serif'; ctx.textAlign = 'right';
    ctx.fillText(y < 1 ? y.toFixed(2) : y.toFixed(1), pad.l - 4, yp + 3);
  }

  // Title — supports _{...} for subscripts, e.g. "λ_{α}" or "|r · e_{α}|"
  if (opts.title) {
    const baseFont = '12px sans-serif';
    const subFont = '9px sans-serif';
    const baseY = pad.t + 12;
    const subY = baseY + 3; // drop subscript slightly below baseline
    ctx.fillStyle = '#666';
    ctx.textAlign = 'left';
    let x = pad.l + 4;
    // Tokenize title into runs of normal text and subscript runs.
    const parts = opts.title.split(/(_\{[^}]*\})/g);
    for (const part of parts) {
      if (!part) continue;
      const m = part.match(/^_\{([^}]*)\}$/);
      if (m) {
        ctx.font = subFont;
        ctx.fillText(m[1], x, subY);
        x += ctx.measureText(m[1]).width;
      } else {
        ctx.font = baseFont;
        ctx.fillText(part, x, baseY);
        x += ctx.measureText(part).width;
      }
    }
  }

  // Bars
  values.forEach((v, i) => {
    const x = pad.l + (i + 0.5) * spacing - barW / 2;
    const h = (v / yMax) * ph;
    const y = pad.t + ph - h;
    ctx.fillStyle = BAR_COLORS[i % BAR_COLORS.length];
    ctx.fillRect(x, y, barW, h);

    // Label
    ctx.fillStyle = '#999'; ctx.font = '9px sans-serif'; ctx.textAlign = 'center';
    ctx.fillText(i + 1, pad.l + (i + 0.5) * spacing, pad.t + ph + 12);
  });
}

// ─────────────────────────────────────────────────────
// 8. Eigenvalue/eigenvector computation (2×2 symmetric)
//    For our 4×4 NTK (2 data points × 2 output dims)
//    we do a simple power-iteration / analytic approach.
//    For the full widget we build the 4×4 Theta matrix.
// ─────────────────────────────────────────────────────

// Build the full Theta matrix (4×4) from W1, W2
// Data: x1=(1,0), x2=(0,1)  — 2 data points, 2 outputs
const X_DATA = [[1, 0], [0, 1]];

function buildFullTheta(W1, W2) {
  // Returns 4×4 matrix
  // Index: (a=0..1, j=0..1) -> a*2 + j
  const N = 4;
  const T = Array.from({length: N}, () => new Float64Array(N));
  for (let a = 0; a < 2; a++) {
    for (let b = 0; b < 2; b++) {
      const blk = computeNTK(W1, W2, X_DATA[a], X_DATA[b]);
      // blk[i][j] goes into T[a*2+i][b*2+j]
      for (let i = 0; i < 2; i++)
        for (let j = 0; j < 2; j++)
          T[a * 2 + i][b * 2 + j] = blk[i][j];
    }
  }
  return T;
}

// Symmetric eigendecomposition via Jacobi for small matrices
function jacobiEig(A) {
  const n = A.length;
  // Copy
  const M = A.map(r => Float64Array.from(r));
  const V = Array.from({length: n}, (_, i) => {
    const r = new Float64Array(n); r[i] = 1; return r;
  });
  for (let iter = 0; iter < 200; iter++) {
    // Find max off-diagonal
    let p = 0, q = 1, maxVal = 0;
    for (let i = 0; i < n; i++)
      for (let j = i + 1; j < n; j++)
        if (Math.abs(M[i][j]) > maxVal) { maxVal = Math.abs(M[i][j]); p = i; q = j; }
    if (maxVal < 1e-12) break;
    const theta = (M[q][q] - M[p][p]) / (2 * M[p][q]);
    const t = Math.sign(theta) / (Math.abs(theta) + Math.sqrt(1 + theta * theta));
    const c = 1 / Math.sqrt(1 + t * t), s = t * c;
    // Update M
    const Mpp = M[p][p], Mqq = M[q][q], Mpq = M[p][q];
    M[p][p] = c*c*Mpp - 2*s*c*Mpq + s*s*Mqq;
    M[q][q] = s*s*Mpp + 2*s*c*Mpq + c*c*Mqq;
    M[p][q] = M[q][p] = 0;
    for (let r = 0; r < n; r++) {
      if (r === p || r === q) continue;
      const Mrp = M[r][p], Mrq = M[r][q];
      M[r][p] = M[p][r] = c*Mrp - s*Mrq;
      M[r][q] = M[q][r] = s*Mrp + c*Mrq;
    }
    // Update V (eigenvectors in columns)
    for (let r = 0; r < n; r++) {
      const Vrp = V[r][p], Vrq = V[r][q];
      V[r][p] = c*Vrp - s*Vrq;
      V[r][q] = s*Vrp + c*Vrq;
    }
  }
  // Eigenvalues are M[i][i]
  const vals = Array.from({length: n}, (_, i) => M[i][i]);
  // Sort descending
  const idx = vals.map((v, i) => i).sort((a, b) => vals[b] - vals[a]);
  return {
    values: idx.map(i => vals[i]),
    vectors: idx.map(i => Array.from({length: n}, (_, r) => V[r][i]))
  };
}

// ─────────────────────────────────────────────────────
// WIDGET 1: Random NTK
// ─────────────────────────────────────────────────────
function initW1() {
  const container = document.getElementById('w1-ntk-matrix');
  const seedInput = document.getElementById('w1-model-seed');
  const toggleBtn = document.getElementById('w1-toggle-details');
  const detailsDiv = document.getElementById('w1-details');
  if (!container) return;

  let showDetails = false;

  function render() {
    const seed = parseInt(seedInput.value) || 0;
    const {W1, W2} = buildRandomWeights(seed);

    const blocks = [];
    for (const xa of X_DATA) {
      for (const xb of X_DATA) {
        blocks.push({matrix: computeNTK(W1, W2, xa, xb)});
      }
    }
    renderNTKGrid(container, blocks, ['(1,0)', '(0,1)']);

    if (showDetails) {
      detailsDiv.innerHTML =
        `<span>W₁ = [[${W1[0].map(v=>v.toFixed(3)).join(', ')}], [${W1[1].map(v=>v.toFixed(3)).join(', ')}]]</span>` +
        `<span>W₂ = [[${W2[0].map(v=>v.toFixed(3)).join(', ')}], [${W2[1].map(v=>v.toFixed(3)).join(', ')}]]</span>`;
    }
  }

  seedInput.addEventListener('input', render);
  toggleBtn.addEventListener('click', () => {
    showDetails = !showDetails;
    toggleBtn.textContent = showDetails ? 'hide weights' : 'show weights';
    detailsDiv.style.display = showDetails ? 'flex' : 'none';
    render();
  });

  render();
}

// ─────────────────────────────────────────────────────
// WIDGET 2: NTK time evolution (Widget 3 in HTML)
// ─────────────────────────────────────────────────────
function initW3() {
  const matContainer = document.getElementById('w3-ntk-matrix');
  const sigmaCanvas = document.getElementById('w3-sigma-plot');
  const lossCanvas = document.getElementById('w3-loss-plot');
  const tSlider = document.getElementById('w3-t-slider');
  const tDisplay = document.getElementById('w3-t-display');
  const s1Input = document.getElementById('w3-s1');
  const s2Input = document.getElementById('w3-s2');
  const seedInput = document.getElementById('w3-basis-seed');
  const toggleBtn = document.getElementById('w3-toggle-details');
  const detailsDiv = document.getElementById('w3-details');
  if (!matContainer) return;

  let showDetails = false;

  // Pre-compute trajectory
  const N_STEPS = 1001;
  const ts = Array.from({length: N_STEPS}, (_, i) => sliderToTime(i));

  function getParams() {
    const s1 = parseFloat(s1Input.value) || 1.0;
    const s2 = parseFloat(s2Input.value) || 0.4;
    const seed = parseInt(seedInput.value) || 0;
    return {s1, s2, seed};
  }

  function buildUV(seed) {
    const rng = mulberry32(seed * 9999 + 7);
    const U = randOrth2(rng);
    const V = randOrth2(rng);
    return {U, V};
  }

  function getAnalyticWeights(s1, s2, seed, t) {
    const sig = [saxeSigma(s1, t), saxeSigma(s2, t)];
    const {U, V} = buildUV(seed);
    const rng2 = mulberry32(seed * 31337 + 3);
    const O = randOrth2(rng2);
    return buildAnalyticWeights(U, sig, V, O);
  }

  let lastParams = null;
  let sigmaTrajectory = null, lossTrajectory = null;

  function recomputeTrajectory() {
    const {s1, s2} = getParams();
    const sv = [s1, s2];
    sigmaTrajectory = ts.map(t => sv.map(s => saxeSigma(s, t)));
    lossTrajectory = ts.map((t, i) => saxeLoss(sv, sigmaTrajectory[i]));
  }

  function render() {
    const {s1, s2, seed} = getParams();
    const sv = [s1, s2];
    const tIdx = parseInt(tSlider.value);
    const t = sliderToTime(tIdx);
    if (tDisplay) tDisplay.textContent = `t = ${t.toFixed(2)}`;

    // Recompute trajectory only if params changed
    const paramKey = `${s1},${s2}`;
    if (paramKey !== lastParams) {
      lastParams = paramKey;
      recomputeTrajectory();
    }

    // Build weights at this t
    const {W1, W2} = getAnalyticWeights(s1, s2, seed, t);

    // NTK blocks
    const blocks = [];
    for (const xa of X_DATA) {
      for (const xb of X_DATA) {
        blocks.push({matrix: computeNTK(W1, W2, xa, xb)});
      }
    }
    renderNTKGrid(matContainer, blocks, ['(1,0)', '(0,1)'],
      { cellW: 130, cellH: 84, fontSize: 15, labelFontSize: 13 });

    // Sigma plot
    if (sigmaCanvas) {
      const colors = ['#cc3333', '#3366cc', '#33aa55'];
      const sigSeries = sv.map((s, i) => ({
        xs: ts, ys: sigmaTrajectory.map(row => row[i]),
        color: colors[i], lineWidth: 1.8
      }));
      // Target values as dashed horizontal lines
      sv.forEach((s, i) => sigSeries.push({
        xs: [ts[0], ts[ts.length-1]], ys: [s, s],
        color: colors[i], dash: [4, 4], lineWidth: 1
      }));
      drawPlot(sigmaCanvas, sigSeries, {
        yMin: 0, yMax: Math.max(...sv) * 1.15,
        markerX: t, title: 'singular values σᵢ(t)'
      });
    }

    // Loss plot
    if (lossCanvas) {
      drawPlot(lossCanvas, [{
        xs: ts, ys: lossTrajectory, color: '#555', lineWidth: 1.8
      }], {yMin: 0, markerX: t, title: 'training loss'});
    }

    // Details
    if (showDetails) {
      const sig = sv.map(s => saxeSigma(s, t));
      detailsDiv.innerHTML =
        `<span>σ(t) = [${sig.map(v=>v.toFixed(4)).join(', ')}]</span>` +
        `<span>W₁ = [[${W1[0].map(v=>v.toFixed(3)).join(', ')}], [${W1[1].map(v=>v.toFixed(3)).join(', ')}]]</span>` +
        `<span>W₂ = [[${W2[0].map(v=>v.toFixed(3)).join(', ')}], [${W2[1].map(v=>v.toFixed(3)).join(', ')}]]</span>`;
    }
  }

  toggleBtn?.addEventListener('click', () => {
    showDetails = !showDetails;
    toggleBtn.textContent = showDetails ? 'hide weights' : 'show weights';
    detailsDiv.style.display = showDetails ? 'flex' : 'none';
    render();
  });

  tSlider.addEventListener('input', render);
  [s1Input, s2Input, seedInput].forEach(el => el.addEventListener('change', render));

  // Resize observer to redraw canvases on size change
  if (sigmaCanvas && lossCanvas) {
    const ro = new ResizeObserver(() => render());
    ro.observe(sigmaCanvas);
    ro.observe(lossCanvas);
  }

  recomputeTrajectory();
  render();
}

// ─────────────────────────────────────────────────────
// WIDGET 3: Eigenstructure (Widget 4 in HTML)
// ─────────────────────────────────────────────────────
function initW4() {
  const eigCanvas = document.getElementById('w4-eig-bars');
  const residCanvas = document.getElementById('w4-resid-bars');
  const tSlider = document.getElementById('w4-t-slider');
  const tDisplay = document.getElementById('w4-t-display');
  const s1Input = document.getElementById('w4-s1');
  const s2Input = document.getElementById('w4-s2');
  const seedInput = document.getElementById('w4-basis-seed');
  if (!eigCanvas) return;

  // Residual at time t:
  // r_aj = f(x_a)_j - M x_a  _j
  //      = (U Sigma(t) V^T - M) x_a  component j
  function getResidual(W1, W2, M_target) {
    // M_target: 2×2
    // f(x) = W2 W1 x, residual = f(x) - M_target x
    // Returns length-4 vector: [(a=0,j=0), (a=0,j=1), (a=1,j=0), (a=1,j=1)]
    const res = new Float64Array(4);
    for (let a = 0; a < 2; a++) {
      const x = X_DATA[a];
      const W1x = [dot(W1[0], x), dot(W1[1], x)];
      const fx = [dot(W2[0], W1x), dot(W2[1], W1x)];
      const Mx = [dot(M_target[0], x), dot(M_target[1], x)];
      for (let j = 0; j < 2; j++) res[a * 2 + j] = fx[j] - Mx[j];
    }
    return res;
  }

  function getParams() {
    const s1 = parseFloat(s1Input.value) || 1.0;
    const s2 = parseFloat(s2Input.value) || 0.4;
    const seed = parseInt(seedInput.value) || 0;
    return {s1, s2, seed};
  }

  // Build M_target from U, Sigma_target, V
  function buildMTarget(s1, s2, seed) {
    const rng = mulberry32(seed * 9999 + 7);
    const U = randOrth2(rng);
    const V = randOrth2(rng);
    // M = U diag(s1,s2) V^T
    const M = [[0,0],[0,0]];
    for (let i = 0; i < 2; i++)
      for (let j = 0; j < 2; j++)
        M[i][j] = U[i][0] * s1 * V[j][0] + U[i][1] * s2 * V[j][1];
    return M;
  }

  // Fixed y-maxima computed once per param set so the scale doesn't jump with t
  let fixedEigYMax = null;
  let fixedResidYMax = null;
  // Eigenvector ordering fixed at t_max for stable bar positions
  let refEig = null;

  function recomputeFixed() {
    const {s1, s2, seed} = getParams();
    const sv = [s1, s2];

    const rng = mulberry32(seed * 9999 + 7);
    const U = randOrth2(rng);
    const V = randOrth2(rng);
    const rng2 = mulberry32(seed * 31337 + 3);
    const O = randOrth2(rng2);

    // Fix eigenvector ordering at t_max (slider = 1000)
    const tMax = sliderToTime(1000);
    const sigMax = sv.map(s => saxeSigma(s, tMax));
    const {W1: W1max, W2: W2max} = buildAnalyticWeights(U, sigMax, V, O);
    refEig = jacobiEig(buildFullTheta(W1max, W2max));

    // Fixed eigenvalue y-max: largest eigenvalue at t_max
    fixedEigYMax = Math.max(...refEig.values) * 1.15;

    // Fixed residual y-max: magnitude of residual at t=0
    // At t=0, sigma~SIGMA0, so f(x)~0 and r~-M*x
    const M_target = buildMTarget(s1, s2, seed);
    const sig0 = sv.map(s => saxeSigma(s, 0));
    const {W1: W1_0, W2: W2_0} = buildAnalyticWeights(U, sig0, V, O);
    const r0 = getResidual(W1_0, W2_0, M_target);
    const residProj0 = refEig.vectors.map(ev => Math.abs(dot(Array.from(r0), ev)));
    fixedResidYMax = Math.max(...residProj0) * 1.15;
  }

  function render() {
    const {s1, s2, seed} = getParams();
    const sv = [s1, s2];
    const tIdx = parseInt(tSlider.value);
    const t = sliderToTime(tIdx);
    if (tDisplay) tDisplay.textContent = `t = ${t.toFixed(2)}`;

    const rng = mulberry32(seed * 9999 + 7);
    const U = randOrth2(rng);
    const V = randOrth2(rng);
    const rng2 = mulberry32(seed * 31337 + 3);
    const O = randOrth2(rng2);

    const sig = sv.map(s => saxeSigma(s, t));
    const {W1, W2} = buildAnalyticWeights(U, sig, V, O);
    const Theta = buildFullTheta(W1, W2);
    const eig = jacobiEig(Theta);

    // Reorder eigenvalues/residuals to match reference ordering (stable bars)
    const reorder = refEig.vectors.map(refV => {
      // Find the eig vector most aligned with this reference vector
      let best = 0, bestDot = -1;
      eig.vectors.forEach((v, i) => {
        const d = Math.abs(dot(refV, v));
        if (d > bestDot) { bestDot = d; best = i; }
      });
      return best;
    });

    const orderedEigVals = reorder.map(i => Math.max(0, eig.values[i]));
    const orderedEigVecs = reorder.map(i => eig.vectors[i]);

    // Residual at current t — project onto reference eigenvectors (fixed at t_max),
    // which have well-defined identity even when Theta is nearly degenerate at t=0
    const M_target = buildMTarget(s1, s2, seed);
    const r = getResidual(W1, W2, M_target);
    const residProj = refEig.vectors.map(ev => Math.abs(dot(Array.from(r), ev)));

    drawBars(eigCanvas, orderedEigVals,
      {title: 'NTK eigenvalues λ_{α}', yMax: fixedEigYMax});
    drawBars(residCanvas, residProj,
      {title: '|r · e_{α}|  (residual projection)', yMax: fixedResidYMax});
  }

  tSlider.addEventListener('input', render);
  [s1Input, s2Input, seedInput].forEach(el => el.addEventListener('change', () => {
    recomputeFixed();
    render();
  }));

  const ro = new ResizeObserver(() => render());
  ro.observe(eigCanvas); ro.observe(residCanvas);

  recomputeFixed();
  render();
}

// ─────────────────────────────────────────────────────
// Boot
// ─────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initW1();
  initW3();
  initW4();
});
