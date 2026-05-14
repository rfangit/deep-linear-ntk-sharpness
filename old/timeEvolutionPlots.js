// ============================================================================
// TIME EVOLUTION PLOTS - Loss(t) and σᵢ(t) curves with current-time marker.
// ============================================================================
// Follows the same canvas-handling pattern as experiments2.js:
//   - resizeCanvas(canvas) reads getBoundingClientRect and sets canvas.width /
//     .height to that × devicePixelRatio.
//   - Caller invokes it once at init and on window resize.
//   - Each render reads canvas.width / .height, computes scales, draws.
// The plots' display size is determined entirely by CSS on their containers
// (which have a fixed height set by HTML). The canvases just fill them.

import { sigmaAtTime } from './dynamics.js';

const PAD = 38;            // padding on left, right, bottom (axis labels live here)
const PAD_TOP = 14;        // smaller — no axis label or tick text on top
const LINE_WIDTH = 2.0;
const SAMPLES = 200;

const SIGMA_COLORS = ['#cc4455', '#3366cc', '#22aa66', '#aa55cc'];
const LOSS_COLOR = '#222';
const TARGET_COLOR = '#aaa';
const GHOST_ALPHA = 0.12;

function makeScaleX(dataMin, dataMax, pixelW, pad) {
  return v => pad + (v - dataMin) / (dataMax - dataMin) * (pixelW - 2 * pad);
}
// Y scale takes separate top and bottom paddings so we can trim the top margin
// without sacrificing room for the x-axis label at the bottom.
function makeScaleY(dataMin, dataMax, pixelH, padTop, padBot) {
  return v => (pixelH - padBot) - (v - dataMin) / (dataMax - dataMin) * (pixelH - padTop - padBot);
}

const dpr = window.devicePixelRatio || 1;

/**
 * Draw a curve in two passes: faint full curve, then solid past portion up
 * to tCurrent (with linear interpolation so it ends exactly under the marker).
 */
function drawCurve(ctx, ts, ys, scX, scY, color, lineWidth, tCurrent) {
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;

  // Faint full curve.
  ctx.save();
  ctx.globalAlpha = GHOST_ALPHA;
  ctx.beginPath();
  for (let i = 0; i < ts.length; i++) {
    const sx = scX(ts[i]);
    const sy = scY(ys[i]);
    if (i === 0) ctx.moveTo(sx, sy);
    else ctx.lineTo(sx, sy);
  }
  ctx.stroke();
  ctx.restore();

  // Solid past portion. Find largest index with ts[i] <= tCurrent.
  let lastIdx = -1;
  for (let i = 0; i < ts.length; i++) {
    if (ts[i] <= tCurrent) lastIdx = i;
    else break;
  }
  if (lastIdx < 0) return;

  ctx.beginPath();
  for (let i = 0; i <= lastIdx; i++) {
    const sx = scX(ts[i]);
    const sy = scY(ys[i]);
    if (i === 0) ctx.moveTo(sx, sy);
    else ctx.lineTo(sx, sy);
  }
  if (lastIdx < ts.length - 1) {
    const t0 = ts[lastIdx], t1 = ts[lastIdx + 1];
    const span = t1 - t0;
    if (span > 0) {
      const frac = (tCurrent - t0) / span;
      const yInterp = ys[lastIdx] + frac * (ys[lastIdx + 1] - ys[lastIdx]);
      ctx.lineTo(scX(tCurrent), scY(yInterp));
    }
  }
  ctx.stroke();
}

/** Sets the canvas's internal pixel dimensions to its CSS size × DPR. */
export function resizeCanvas(canvas) {
  const r = canvas.getBoundingClientRect();
  canvas.width  = Math.round(r.width  * dpr);
  canvas.height = Math.round(r.height * dpr);
}

export function computeCurves(targets, sigma0, tMax) {
  const ts = new Array(SAMPLES);
  const losses = new Array(SAMPLES);
  const sigmas = targets.map(() => new Array(SAMPLES));

  let lossMax = 0;
  let sigmaMax = 0;
  for (let i = 0; i < SAMPLES; i++) {
    const t = (i / (SAMPLES - 1)) * tMax;
    ts[i] = t;
    let l = 0;
    for (let k = 0; k < targets.length; k++) {
      const sig = sigmaAtTime(targets[k], sigma0, t);
      sigmas[k][i] = sig;
      const r = targets[k] - sig;
      l += 0.5 * r * r;
      if (sig > sigmaMax) sigmaMax = sig;
    }
    losses[i] = l;
    if (l > lossMax) lossMax = l;
  }

  if (lossMax === 0) lossMax = 1;
  if (sigmaMax === 0) sigmaMax = 1;

  return { ts, losses, sigmas, lossMax, sigmaMax, tMax };
}

function drawAxes(ctx, W, H, xLabel, yLabel) {
  ctx.strokeStyle = '#ccc';
  ctx.lineWidth = 1 * dpr;
  ctx.beginPath();
  ctx.moveTo(PAD * dpr, H - PAD * dpr);
  ctx.lineTo(W - PAD * dpr, H - PAD * dpr);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(PAD * dpr, PAD_TOP * dpr);
  ctx.lineTo(PAD * dpr, H - PAD * dpr);
  ctx.stroke();

  ctx.fillStyle = '#444';
  // Larger x-axis label.
  ctx.font = `${15 * dpr}px system-ui, sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'bottom';
  ctx.fillText(xLabel, W / 2, H - 8 * dpr);
  // Y-label rotated.
  ctx.save();
  ctx.font = `${15 * dpr}px system-ui, sans-serif`;
  ctx.translate(14 * dpr, H / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textBaseline = 'top';
  ctx.fillText(yLabel, 0, 0);
  ctx.restore();
}

function drawCurrentTimeLine(ctx, scX, t, H) {
  const sx = scX(t);
  ctx.save();
  ctx.globalAlpha = 0.6;
  ctx.strokeStyle = '#888';
  ctx.lineWidth = 1.2 * dpr;
  ctx.setLineDash([4 * dpr, 4 * dpr]);
  ctx.beginPath();
  ctx.moveTo(sx, PAD_TOP * dpr);
  ctx.lineTo(sx, H - PAD * dpr);
  ctx.stroke();
  ctx.restore();
}

export function renderLossPlot(canvas, curves, tCurrent) {
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const scX = makeScaleX(0, curves.tMax, W, PAD * dpr);
  const scY = makeScaleY(0, curves.lossMax, H, PAD_TOP * dpr, PAD * dpr);

  drawAxes(ctx, W, H, 't', 'loss L(t)');

  drawCurve(ctx, curves.ts, curves.losses, scX, scY,
            LOSS_COLOR, LINE_WIDTH * dpr, tCurrent);

  drawCurrentTimeLine(ctx, scX, tCurrent, H);
}

export function renderSigmaPlot(canvas, curves, targets, tCurrent) {
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  let yMax = curves.sigmaMax;
  for (const s of targets) if (s > yMax) yMax = s;
  yMax *= 1.1;
  if (yMax === 0) yMax = 1;

  const scX = makeScaleX(0, curves.tMax, W, PAD * dpr);
  const scY = makeScaleY(0, yMax, H, PAD_TOP * dpr, PAD * dpr);

  drawAxes(ctx, W, H, 't', 'singular values σᵢ(t)');

  // Target reference lines.
  ctx.save();
  ctx.strokeStyle = TARGET_COLOR;
  ctx.lineWidth = 1 * dpr;
  ctx.setLineDash([3 * dpr, 3 * dpr]);
  for (let k = 0; k < targets.length; k++) {
    if (targets[k] <= 0) continue;
    const sy = scY(targets[k]);
    ctx.beginPath();
    ctx.moveTo(PAD * dpr, sy);
    ctx.lineTo(W - PAD * dpr, sy);
    ctx.stroke();
  }
  ctx.restore();

  for (let k = 0; k < curves.sigmas.length; k++) {
    drawCurve(ctx, curves.ts, curves.sigmas[k], scX, scY,
              SIGMA_COLORS[k % SIGMA_COLORS.length],
              LINE_WIDTH * dpr, tCurrent);
  }

  drawCurrentTimeLine(ctx, scX, tCurrent, H);

  // Legend.
  const legendX = W - PAD * dpr - 70 * dpr;
  let legendY = PAD_TOP * dpr + 6 * dpr;
  ctx.font = `${11 * dpr}px system-ui, sans-serif`;
  ctx.textAlign = 'left';
  ctx.textBaseline = 'middle';
  for (let k = 0; k < targets.length; k++) {
    ctx.fillStyle = SIGMA_COLORS[k % SIGMA_COLORS.length];
    ctx.fillRect(legendX, legendY - 5 * dpr, 10 * dpr, 10 * dpr);
    ctx.fillStyle = '#444';
    ctx.fillText(`σ${k + 1}, s=${targets[k].toFixed(2)}`, legendX + 14 * dpr, legendY);
    legendY += 16 * dpr;
  }
}
