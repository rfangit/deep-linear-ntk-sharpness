// ============================================================================
// BAR CHARTS - Side-by-side bar charts for the NTK eigenvalue widget.
// ============================================================================
// Two render functions, one helper for computing residuals. Designed to live
// alongside timeEvolutionPlots.js: same canvas-resize-to-CSS-box pattern,
// same drawing style.
//
// Both bar charts share the same call signature:
//   renderBarChart(canvas, values, options)
//
// where options carries yMax (fixed scale across t), per-bar colors,
// per-bar labels, an optional hovered index, and y-axis label.

const PAD_LEFT = 50;
const PAD_RIGHT = 18;
const PAD_BOTTOM = 38;
const PAD_TOP = 18;

// Lazy DPR lookup so this module imports cleanly in node (for tests). In the
// browser this resolves to window.devicePixelRatio on first canvas operation.
function getDpr() {
  return (typeof window !== 'undefined' && window.devicePixelRatio) || 1;
}

/** Sets the canvas's internal pixel dimensions to its CSS size × DPR. */
export function resizeBarCanvas(canvas) {
  const dpr = getDpr();
  const r = canvas.getBoundingClientRect();
  canvas.width = Math.round(r.width * dpr);
  canvas.height = Math.round(r.height * dpr);
}

/**
 * Render a vertical bar chart.
 *
 * @param {HTMLCanvasElement} canvas
 * @param {number[]} values         Bar heights. Must be ≥ 0.
 * @param {object}   options
 * @param {number}   options.yMax        Fixed upper y-bound. Caller decides.
 * @param {string[]} options.colors      One color per bar.
 * @param {string[]} options.labels      One x-axis label per bar.
 * @param {string}   options.yAxisLabel  Rotated label on the left.
 * @param {number|null} [options.hoveredIndex=null]
 *                   Bar to highlight (others fade). null = no highlight.
 */
export function renderBarChart(canvas, values, options) {
  const dpr = getDpr();
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const yMax = options.yMax > 0 ? options.yMax : 1;
  const n = values.length;
  const padL = PAD_LEFT * dpr;
  const padR = PAD_RIGHT * dpr;
  const padT = PAD_TOP * dpr;
  const padB = PAD_BOTTOM * dpr;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  // Axes (left + bottom).
  ctx.strokeStyle = '#ccc';
  ctx.lineWidth = 1 * dpr;
  ctx.beginPath();
  ctx.moveTo(padL, padT);
  ctx.lineTo(padL, H - padB);
  ctx.lineTo(W - padR, H - padB);
  ctx.stroke();

  // Y-axis ticks: 0, ½ yMax, yMax. Plain so the user can read magnitudes.
  ctx.fillStyle = '#555';
  ctx.font = `${11 * dpr}px system-ui, sans-serif`;
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  const tickValues = [0, yMax * 0.5, yMax];
  for (const tv of tickValues) {
    const ty = (H - padB) - (tv / yMax) * plotH;
    // Tick mark.
    ctx.strokeStyle = '#ccc';
    ctx.beginPath();
    ctx.moveTo(padL - 4 * dpr, ty);
    ctx.lineTo(padL, ty);
    ctx.stroke();
    // Faint gridline across the plot.
    ctx.strokeStyle = '#eee';
    ctx.beginPath();
    ctx.moveTo(padL, ty);
    ctx.lineTo(W - padR, ty);
    ctx.stroke();
    ctx.fillStyle = '#666';
    ctx.fillText(formatTick(tv), padL - 7 * dpr, ty);
  }

  // Y-axis label (rotated).
  ctx.save();
  ctx.fillStyle = '#444';
  ctx.font = `${14 * dpr}px system-ui, sans-serif`;
  ctx.translate(15 * dpr, padT + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(options.yAxisLabel, 0, 0);
  ctx.restore();

  // Bars.
  const slotW = plotW / n;
  const barW = slotW * 0.6;
  const hovered = options.hoveredIndex == null ? -1 : options.hoveredIndex;

  for (let i = 0; i < n; i++) {
    const v = Math.max(0, values[i]);
    const cx = padL + (i + 0.5) * slotW;
    const x = cx - barW / 2;
    const barH = (v / yMax) * plotH;
    const y = (H - padB) - barH;

    let alpha = 1;
    if (hovered >= 0 && hovered !== i) alpha = 0.25;

    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.fillStyle = options.colors[i];
    ctx.fillRect(x, y, barW, barH);
    // Outline for hovered bar.
    if (hovered === i) {
      ctx.globalAlpha = 1;
      ctx.strokeStyle = '#222';
      ctx.lineWidth = 2 * dpr;
      ctx.strokeRect(x, y, barW, barH);
    }
    ctx.restore();

    // X-axis label.
    ctx.fillStyle = '#444';
    ctx.font = `${12 * dpr}px system-ui, sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText(options.labels[i], cx, H - padB + 6 * dpr);
  }
}

/**
 * Given a click/hover position in canvas-local CSS pixels, return which bar
 * (if any) is under the pointer. Returns null when not on a bar.
 *
 * The geometry mirrors renderBarChart's slot layout exactly. Callers should
 * convert mouse events with getBoundingClientRect() before calling.
 */
export function barIndexAt(canvas, cssX, cssY, n) {
  const r = canvas.getBoundingClientRect();
  if (cssX < 0 || cssX > r.width || cssY < 0 || cssY > r.height) return null;
  const padL_css = PAD_LEFT;
  const padR_css = PAD_RIGHT;
  const padT_css = PAD_TOP;
  const padB_css = PAD_BOTTOM;
  const plotW_css = r.width - padL_css - padR_css;
  // Bars span the full plot vertically — hovering anywhere in the column is
  // fine, friendlier than requiring the bar's actual painted region.
  if (cssX < padL_css || cssX > r.width - padR_css) return null;
  if (cssY < padT_css || cssY > r.height - padB_css) return null;
  const slot = plotW_css / n;
  const idx = Math.floor((cssX - padL_css) / slot);
  if (idx < 0 || idx >= n) return null;
  return idx;
}

function formatTick(v) {
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a >= 100) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1);
  if (a >= 0.1) return v.toFixed(2);
  return v.toExponential(1);
}

// ============================================================================
// RESIDUAL COMPUTATION
// ============================================================================
// For target M = U Σ_target V^T and snapshot model with singular values σ_i(t),
// the residual on data point x_a is
//   r_a = (U Σ(t) V^T - U Σ_target V^T) x_a = U (Σ(t) - Σ_target) V^T x_a.
// We stack r_{a,j} into a flat vector in (data-point, output-dim)-major
// order — same convention as ntkMatrix — so projection against an NTK
// eigenvector indexes consistently.

/**
 * Build the residual vector at a snapshot.
 *
 * @param {object} args
 * @param {number[][]} args.U              outputDim × outputDim
 * @param {number[][]} args.V              inputDim × inputDim
 * @param {number[]}   args.sigmas         current σ_i(t), length r
 * @param {number[]}   args.targetSigmas   target s_i, length r
 * @param {number[][]} args.dataPoints     N inputs, each length inputDim
 *
 * @returns {number[]}  Flat vector of length N · outputDim, ordered
 *   (a=0,j=0), (a=0,j=1), ..., (a=1,j=0), ...
 */
export function residualVector({ U, V, sigmas, targetSigmas, dataPoints }) {
  const outputDim = U.length;
  const inputDim = V.length;
  const N = dataPoints.length;
  const r = sigmas.length;

  // Δ_l = σ_l(t) - s_l on the diagonal, padded to min(out, in).
  const diag = new Array(Math.min(outputDim, inputDim)).fill(0);
  for (let l = 0; l < r; l++) diag[l] = sigmas[l] - targetSigmas[l];

  const result = new Array(N * outputDim).fill(0);

  for (let a = 0; a < N; a++) {
    const x = dataPoints[a];

    // V^T x (length inputDim).
    const Vtx = new Array(inputDim).fill(0);
    for (let l = 0; l < inputDim; l++) {
      let s = 0;
      for (let k = 0; k < inputDim; k++) s += V[k][l] * x[k];
      Vtx[l] = s;
    }

    // Σ V^T x (length outputDim, with non-square Σ padded by zeros).
    const SVtx = new Array(outputDim).fill(0);
    for (let l = 0; l < diag.length; l++) SVtx[l] = diag[l] * Vtx[l];

    // U (Σ V^T x) — the residual for data point a.
    for (let j = 0; j < outputDim; j++) {
      let s = 0;
      for (let l = 0; l < outputDim; l++) s += U[j][l] * SVtx[l];
      result[a * outputDim + j] = s;
    }
  }

  return result;
}

/**
 * Project a vector onto a list of orthonormal eigenvectors and return
 * absolute coefficients.
 *
 * @param {number[]}   vec
 * @param {number[][]} eigenvectors  list of unit vectors
 * @returns {number[]}                |vec · e_α| for each α
 */
export function projectAbs(vec, eigenvectors) {
  const out = new Array(eigenvectors.length).fill(0);
  for (let a = 0; a < eigenvectors.length; a++) {
    const e = eigenvectors[a];
    let dot = 0;
    for (let i = 0; i < vec.length; i++) dot += vec[i] * e[i];
    out[a] = Math.abs(dot);
  }
  return out;
}
