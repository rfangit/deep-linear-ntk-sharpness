// ============================================================================
// VISUALIZATION - Chart.js loss and sharpness plotting
// ============================================================================
// Both LossChart and RightChart accept an options object in their constructor
// to control which series are included. Only declared series get datasets and
// legend entries.
//
// LossChart options:
//   showTrain:   bool (default true)  - train loss (raw + EMA)
//   showEma:     bool (default true)  - enable EMA smoothing support
//
// RightChart options:
//   kEigs:         number (default 3) - how many eigenvalue curves (1-3)
//   showThreshold: bool   (default true) - the 2/η dashed line
//   showPrediction: bool  (default true) - overlay Saxe theory predictions
//                                          (kEigs dotted black curves)

import { IncrementalCache } from './incremental-cache.js';
import { formatTickLabel, baseChartOptions, CHART_FONT } from './chart-utils.js';
import {
  aggregatePooled, aggregatePerClass, GN_GROUPS, FULL_GROUPS, GROUP_LABELS
} from './theory-aggregate.js';

const MAX_PLOT_POINTS = 1000;

// Per-class colors for the per-class theory view (residuals.html). index.html's
// pooled view ignores these (it uses neutral black). Keyed by group name.
const CLASS_COLORS = {
  aligned:      'rgb(40, 130, 180)',   // blue
  aligned_null: 'rgb(80, 180, 80)',    // green
  cross:        'rgb(220, 50, 50)',    // red
  single_value: 'rgb(150, 80, 200)',   // purple
  hidden_null:  'rgb(230, 150, 30)',   // orange — hidden null (±), nonzero branches
  idle_null:    'rgb(140, 140, 140)'   // grey — hidden null (0), the flat zero line
};
const NEUTRAL_PRED_COLOR = 'rgb(0, 0, 0)';
// Stroke patterns distinguish the two theories when both are shown.
const GN_DASH   = [2, 4];        // dotted   (matches the original theory line)
const FULL_DASH = [6, 3, 2, 3];  // dash-dot
// (The exact dense spectrum is no longer a distinct overlay — when selected it
// is drawn through the measured eigenvalue curves in their normal solid style;
// see RightChart.eigenvalueSource and update().)

// ============================================================================
// LOSS CHART
// ============================================================================

export class LossChart {
  /**
   * @param {string} canvasId
   * @param {object} [options]
   * @param {boolean} [options.showTrain=true]
   * @param {boolean} [options.showEma=true]
   * @param {boolean} [options.showPrediction=true]  — overlay Saxe theory loss
   */
  constructor(canvasId, options = {}) {
    this.showTrain = options.showTrain !== false;
    this.showEma   = options.showEma   !== false;
    this.showPrediction = options.showPrediction !== false;

    this.logScale = false;
    this.logScaleX = false;
    this.useEffectiveTime = false;
    this.eta = 0.01;
    this.emaWindow = 1;

    this.cache = new IncrementalCache(this.emaWindow, MAX_PLOT_POINTS, 'loss', { loss: 0.5 });

    // Separate cache for the predicted loss — same downsampling logic but no
    // EMA (the theory curve is deterministic, no smoothing needed). Sharing
    // MAX_PLOT_POINTS keeps the two curves at comparable density on screen.
    this.predCache = new IncrementalCache(1, MAX_PLOT_POINTS, 'loss', { loss: 0 });

    // Build datasets dynamically. Track indices by name.
    this.idx = {};
    const datasets = [];

    if (this.showTrain) {
      this.idx.trainRaw = datasets.length;
      datasets.push({
        label: this.showEma ? 'train (raw)' : 'train',
        data: [],
        borderColor: this.showEma ? 'rgba(40, 130, 130, 0.3)' : 'rgb(40, 130, 130)',
        backgroundColor: this.showEma ? 'rgba(40, 130, 130, 0.05)' : 'rgba(40, 130, 130, 0.1)',
        borderWidth: 2,
        pointRadius: 0,
        tension: 0,
        order: 4
      });
    }

    if (this.showTrain && this.showEma) {
      this.idx.trainEma = datasets.length;
      datasets.push({
        label: 'train (ema)',
        data: [],
        borderColor: 'rgb(40, 130, 130)',
        backgroundColor: 'rgba(40, 130, 130, 0.1)',
        borderWidth: 2,
        pointRadius: 0,
        tension: 0,
        order: 3,
        hidden: true
      });
    }

    // Theory prediction curve — black dotted, matching the sharpness chart's
    // overlay style so the two prediction series read as one thing visually.
    this.idx.pred = datasets.length;
    datasets.push({
      label: 'theory',
      data: [],
      borderColor: 'rgb(0, 0, 0)',
      borderWidth: 1.5,
      borderDash: [2, 4],
      pointRadius: 0,
      tension: 0,
      order: 1,                  // draw on top of the training curves
      hidden: !this.showPrediction
    });

    const ctx = document.getElementById(canvasId).getContext('2d');
    this.chart = new Chart(ctx, {
      type: 'line',
      data: { datasets },
      options: baseChartOptions()
    });

    // Custom legend: only show series that have data or are relevant. We
    // append a 'theory' entry whenever the prediction overlay is on.
    const chartRef = this;
    this.chart.options.plugins.legend.labels.generateLabels = function(chart) {
      const labels = [];
      if (chartRef.showTrain) {
        const emaOn = chartRef.showEma && chartRef.emaWindow > 1;
        if (emaOn) {
          labels.push({
            text: 'train (raw)',
            strokeStyle: 'rgba(40, 130, 130, 0.3)',
            fillStyle: 'rgba(40, 130, 130, 0.3)',
            lineWidth: 2, hidden: false
          });
          labels.push({
            text: 'train (ema)',
            strokeStyle: 'rgb(40, 130, 130)',
            fillStyle: 'rgb(40, 130, 130)',
            lineWidth: 2, hidden: false
          });
        } else {
          labels.push({
            text: 'train',
            strokeStyle: 'rgb(40, 130, 130)',
            fillStyle: 'rgb(40, 130, 130)',
            lineWidth: 2, hidden: false
          });
        }
      }
      if (chartRef.showPrediction) {
        labels.push({
          text: 'theory',
          strokeStyle: 'rgb(0, 0, 0)',
          fillStyle: 'rgb(0, 0, 0)',
          lineWidth: 1.5, hidden: false,
          lineDash: [2, 4]
        });
      }
      return labels;
    };
  }

  setLogScale(useLog) {
    this.logScale = useLog;
    this.chart.options.scales.y.type = useLog ? 'logarithmic' : 'linear';
    this.chart.update('none');
  }

  setLogScaleX(useLogX) {
    this.logScaleX = useLogX;
    this.chart.options.scales.x.type = useLogX ? 'logarithmic' : 'linear';
    this.chart.options.scales.x.min = useLogX ? 1 : 0;
    this.chart.update('none');
  }

  setEffectiveTime(useEffTime, eta) {
    this.useEffectiveTime = useEffTime;
    this.eta = eta;
    if (this.logScaleX) {
      this.chart.options.scales.x.min = useEffTime ? 0.001 : 1;
    }
  }

  setEmaWindow(window) {
    this.emaWindow = window;
    this.cache.setEmaWindow(window);
  }

  setInitialLoss(initialLoss) {
    this.cache.initEmaValues = { loss: initialLoss };
    this.cache.lastEmaValues = { loss: initialLoss };
  }

  setShowPrediction(show) {
    this.showPrediction = show;
    if (this.idx.pred !== undefined) {
      this.chart.data.datasets[this.idx.pred].hidden = !show;
    }
    this.chart.update('none');
  }

  /**
   * Update the chart.
   * @param {Array<{iteration, loss}>} lossHistory     — measured training loss
   * @param {number} eta                                — learning rate (η·step axis)
   * @param {Array<{iteration, loss}>} [predictedLossHistory]
   *   Saxe theory loss. Optional — if omitted or empty, the theory curve
   *   stays empty (but the dataset stays in the chart so showPrediction can
   *   be toggled later).
   */
  update(lossHistory, eta = this.eta, predictedLossHistory = null) {
    if (lossHistory.length === 0) return;
    this.eta = eta;

    const { downsampledRaw, downsampledSmoothed, max } = this.cache.update(lossHistory);
    const toX = (iter) => this.useEffectiveTime ? iter * this.eta : iter;

    // Train raw
    if (this.idx.trainRaw !== undefined) {
      const rawData = downsampledRaw.map(p => ({ x: toX(p.iteration), y: p.loss }));
      this.chart.data.datasets[this.idx.trainRaw].data = rawData;

      // Style depends on whether EMA is active
      if (this.showEma && this.emaWindow > 1) {
        this.chart.data.datasets[this.idx.trainRaw].borderColor = 'rgba(40, 130, 130, 0.3)';
        this.chart.data.datasets[this.idx.trainRaw].backgroundColor = 'rgba(40, 130, 130, 0.05)';
      } else {
        this.chart.data.datasets[this.idx.trainRaw].borderColor = 'rgb(40, 130, 130)';
        this.chart.data.datasets[this.idx.trainRaw].backgroundColor = 'rgba(40, 130, 130, 0.1)';
      }
    }

    // Train EMA
    if (this.idx.trainEma !== undefined) {
      if (this.emaWindow > 1) {
        const smoothedData = downsampledSmoothed.map(p => ({ x: toX(p.iteration), y: p.loss }));
        this.chart.data.datasets[this.idx.trainEma].data = smoothedData;
        this.chart.data.datasets[this.idx.trainEma].hidden = false;
      } else {
        this.chart.data.datasets[this.idx.trainEma].data = [];
        this.chart.data.datasets[this.idx.trainEma].hidden = true;
      }
    }

    // Theory prediction. Uses its own IncrementalCache for downsampling but no
    // EMA (the theory curve is deterministic). Hidden flag governed by
    // showPrediction; the data is always populated when available so toggling
    // the flag is cheap.
    let predMax = 0;
    if (this.idx.pred !== undefined) {
      if (predictedLossHistory && predictedLossHistory.length > 0) {
        const predCacheResult = this.predCache.update(predictedLossHistory);
        const predData = predCacheResult.downsampledRaw.map(p => ({
          x: toX(p.iteration), y: p.loss
        }));
        this.chart.data.datasets[this.idx.pred].data = predData;
        predMax = predCacheResult.max.loss;
      } else {
        this.chart.data.datasets[this.idx.pred].data = [];
      }
    }

    // X-axis max — extend to cover the predicted curve too if it runs further.
    let lastIteration = lossHistory[lossHistory.length - 1].iteration;
    if (this.showPrediction && predictedLossHistory && predictedLossHistory.length > 0) {
      const pLast = predictedLossHistory[predictedLossHistory.length - 1].iteration;
      if (pLast > lastIteration) lastIteration = pLast;
    }
    this.chart.options.scales.x.max = toX(lastIteration);

    // Y-axis: include the predicted curve in the auto-scale only when it's
    // being shown, so toggling the overlay off doesn't leave headroom that
    // makes the measured curve look squished.
    if (!this.logScale) {
      let yMax = max.loss * 1.4;
      if (this.showPrediction && predMax > 0) {
        yMax = Math.max(yMax, predMax * 1.4);
      }
      this.chart.options.scales.y.max = yMax;
      this.chart.options.scales.y.ticks.callback = function(value) {
        if (Math.abs(value - yMax) < 1e-10) return '';
        return formatTickLabel(value);
      };
    } else {
      this.chart.options.scales.y.max = undefined;
      this.chart.options.scales.y.ticks.callback = function(value) {
        return formatTickLabel(value);
      };
    }

    this.chart.update('none');
  }

  clear() {
    for (const ds of this.chart.data.datasets) {
      ds.data = [];
    }
    this.chart.options.scales.x.max = undefined;
    this.cache.clear();
    this.predCache.clear();
    this.chart.update('none');
  }
}


// ============================================================================
// RIGHT CHART - Hessian eigenvalues + 2/η stability threshold
// ============================================================================
// Plots up to kEigs measured eigenvalue curves (top-k from each Lanczos call,
// sorted descending so ds 0 = λ₁ = largest), plus a theory overlay DERIVED at
// plot time from the analytic σ trajectory (simulation.js sigmaHistory) via
// theory_GN / theory_Hessian_full.
//
// The overlay is controlled by options.predictionConfig (and the live set*
// methods). Two independent knob axes:
//   theories ⊆ ['gn','full']  — which theory function(s) to evaluate. GN and
//        full are different objects: when both are on they are aggregated and
//        drawn SEPARATELY (GN dotted, full dash-dot), never merged.
//   strategy 'pooled'|'perClass' — pooled = flatten a theory's groups, sort,
//        top-k; perClass = independent top-k per class (group), colored by class.
// Defaults reproduce the original index.html overlay exactly: GN only, pooled
// top-kEigs, neutral black dotted, shown.

export class RightChart {
  /**
   * @param {string} canvasId
   * @param {object} [options]
   * @param {number}  [options.kEigs=3] - number of measured eigenvalue curves
   * @param {boolean} [options.showThreshold=true] - show the 2/η dashed line
   * @param {boolean} [options.clipSharpness=true] - cap y-max at 3·(2/η)
   * @param {boolean} [options.clipToEos=true] - force y-axis to include 2/η
   * @param {number}  [options.maxPredDatasets=64] - reusable theory-curve pool
   *        size; caps total theory curves drawn at once (summed over theories
   *        and classes). Raise for dense perClass views.
   * @param {object}  [options.predictionConfig] - initial overlay config:
   *        { theories:['gn'|'full'...], strategy:'pooled'|'perClass',
   *          k:number, perClassK:{group:k}, show:bool }. Defaults to
   *        { theories:['gn'], strategy:'pooled', k:kEigs, show:true } — i.e.
   *        original index.html behavior.
   */
  constructor(canvasId, options = {}) {
    this.kEigs = options.kEigs || 3;
    // Max measured curves the chart can EVER show. The chart pre-allocates this
    // many measured-curve datasets up front, so the live displayed count
    // (this.kEigs) can be raised anywhere up to this ceiling — and the run-time
    // "Lanczos eigenvalues tracked" knob can grow — without rebuilding datasets
    // or resetting. We default to a generous ceiling rather than the initial
    // display count so raising the count later always has datasets to populate.
    // (Hidden datasets cost nothing visually.) Honors an explicit maxDisplayEigs
    // when given, but never below the initial display count.
    const ceiling = (typeof options.maxDisplayEigs === 'number')
      ? options.maxDisplayEigs : 20;
    this.maxDisplayEigs = Math.max(ceiling, this.kEigs);
    this.showThreshold = options.showThreshold !== false;
    this.clipSharpness = options.clipSharpness !== false; // default: true
    this.clipToEos = options.clipToEos !== false; // default: true

    // ── Prediction view configuration ────────────────────────────────────────
    // The theory overlay is derived at plot time from σ-history (see update()).
    // Two independent knob axes:
    //   theories:   which theory functions to evaluate — any subset of
    //               ['gn','full']. Drawn separately (GN dotted, full dash-dot).
    //   strategy:   'pooled' | 'perClass' — how to aggregate each theory.
    //   k:          pooled top-k (per theory).
    //   perClassK:  { group: k } for the perClass strategy.
    //   show:       master visibility.
    //
    // Defaults reproduce the original index.html behavior exactly: GN only,
    // pooled top-kEigs, neutral black dotted, shown.
    const pc = options.predictionConfig || {};
    this.predictionConfig = {
      theories:     Array.isArray(pc.theories) ? pc.theories.slice() : ['gn'],
      strategy:     pc.strategy === 'perClass' ? 'perClass' : 'pooled',
      k:            (typeof pc.k === 'number' && pc.k > 0) ? pc.k : this.kEigs,
      perClassK:    pc.perClassK ? { ...pc.perClassK } : {},
      show:         pc.show !== false
    };
    // Reusable prediction-dataset pool size. Caps total theory curves drawn at
    // once (summed over theories and classes). Raise for dense perClass views.
    this.maxPredDatasets = (typeof options.maxPredDatasets === 'number' && options.maxPredDatasets > 0)
      ? options.maxPredDatasets : 64;

    this.logScale = false;
    this.logScaleX = false;
    this.useEffectiveTime = false;
    this.eta = 0.01;

    // Which eigenvalue history feeds the measured (solid colored) curves:
    //   'lanczos' — the Lanczos top-k (default; always available)
    //   'exact'   — the dense-diagonalization spectrum (only when a run computed
    //               it; falls back to Lanczos if exact history is absent).
    // Both are computed by the simulation; this only chooses what's drawn, and
    // is switched live from the run-controls panel (like the displayed count).
    this.eigenvalueSource = 'lanczos';

    const eigColors = [
      'rgb(220, 50, 50)',    // λ₁ (largest) - red
      'rgb(230, 120, 30)',   // λ₂ - orange
      'rgb(80, 180, 80)',    // λ₃ - green
      'rgb(150, 80, 200)',   // λ₄ - purple
      'rgb(40, 130, 180)',   // λ₅ - blue
      'rgb(200, 100, 150)',  // λ₆ - pink
      'rgb(120, 100, 60)',   // λ₇ - olive/brown
      'rgb(60, 160, 160)',   // λ₈ - teal
      'rgb(180, 160, 40)',   // λ₉ - mustard
      'rgb(100, 100, 200)',  // λ₁₀ - indigo
      'rgb(170, 70, 90)',    // λ₁₁ - maroon
      'rgb(70, 140, 110)',   // λ₁₂ - sea green
      'rgb(150, 110, 200)',  // λ₁₃ - lavender
      'rgb(210, 150, 70)',   // λ₁₄ - tan
      'rgb(90, 170, 210)',   // λ₁₅ - sky
      'rgb(190, 90, 160)',   // λ₁₆ - magenta
      'rgb(110, 150, 70)',   // λ₁₇ - moss
      'rgb(140, 120, 100)',  // λ₁₈ - taupe
      'rgb(80, 120, 180)',   // λ₁₉ - steel
      'rgb(160, 80, 120)',   // λ₂₀ - rose
    ];
    // Subscript-digit helper so labels read λ₁..λ₂₀ (and degrade gracefully past).
    const SUBS = ['₀','₁','₂','₃','₄','₅','₆','₇','₈','₉'];
    const sub = (n) => String(n).split('').map(c => SUBS[+c] ?? c).join('');
    const eigLabel = (i) => `λ${sub(i + 1)}`;

    // Stash deterministic style info on the instance so _legendItems() can build
    // measured + threshold entries from KNOWN config without reading back
    // this.chart.data.datasets. (Critical: during the first render Chart.js calls
    // generateLabels from inside `new Chart(...)`, before `this.chart` has been
    // assigned — so the legend must not depend on this.chart existing.)
    this._eigColors = eigColors;
    this._eigLabel = eigLabel;
    this._thresholdStyle = { color: 'rgb(0, 0, 0)', width: 3.5, dash: [8, 4] };

    // Build datasets dynamically. Track indices by name.
    this.idx = {};
    const datasets = [];

    if (this.showThreshold) {
      this.idx.threshold = datasets.length;
      datasets.push({
        label: '2/η',
        data: [],
        borderColor: 'rgb(0, 0, 0)',
        borderWidth: 3.5,
        borderDash: [8, 4],
        pointRadius: 0,
        tension: 0,
        order: 0
      });
    }

    // Measured eigenvalue curves (solid, colored). Allocate up to
    // maxDisplayEigs (= tracked count) so the displayed count can be raised
    // live without rebuilding datasets. All hidden until update() populates them.
    this.idx.eigs = [];
    for (let i = 0; i < this.maxDisplayEigs; i++) {
      this.idx.eigs.push(datasets.length);
      datasets.push({
        label: eigLabel(i),
        data: [],
        borderColor: eigColors[i] || eigColors[i % eigColors.length],
        borderWidth: 2,
        pointRadius: 0,
        tension: 0,
        order: i + 1,
        hidden: true
      });
    }

    // Exact (dense-diagonalization) eigenvalue curves — VESTIGIAL. Exact
    // eigenvalues are now drawn through the measured slots above (selected live
    // via eigenvalueSource), so these datasets are never populated; update()
    // keeps them hidden/empty. The allocation is retained only so downstream
    // dataset indices (predPool) are unchanged. Safe to remove in a future pass
    // that also reindexes predPool.
    this.idx.exactEigs = [];
    for (let i = 0; i < this.maxDisplayEigs; i++) {
      this.idx.exactEigs.push(datasets.length);
      datasets.push({
        label: '',
        data: [],
        borderColor: eigColors[i] || eigColors[i % eigColors.length],
        borderWidth: 2,
        pointRadius: 0,
        tension: 0,
        order: i + 1,
        hidden: true
      });
    }

    // Prediction overlay pool. A fixed bank of generic line datasets whose
    // data/color/dash/label are reassigned each update from the derived theory
    // curves (see _repaintPrediction). Pre-allocated once so switching views
    // never adds/removes datasets — only repaints. Drawn above measured curves.
    this.idx.predPool = [];
    for (let i = 0; i < this.maxPredDatasets; i++) {
      this.idx.predPool.push(datasets.length);
      datasets.push({
        label: '',
        data: [],
        borderColor: NEUTRAL_PRED_COLOR,
        borderWidth: 1.5,
        borderDash: GN_DASH,
        pointRadius: 0,
        tension: 0,
        order: 10 + i,
        hidden: true
      });
    }

    const ctx = document.getElementById(canvasId).getContext('2d');

    // Custom legend generator. Built from the chart's CONFIGURED INTENT — the
    // display count, threshold flag, and prediction config — NOT from whether
    // data has streamed in yet. This means the correct legend shows immediately
    // at construction (e.g. λ₁,λ₂,λ₃ + 2/η + theory for index.html), and stays
    // stable across init / plot / reset. Reads live state via chartRef so
    // setDisplayK / theory-config setters update it automatically.
    //
    // Installed into the options object BEFORE `new Chart(...)` so the very
    // first render uses it (assigning it post-construction left the initial
    // render using Chart.js's default generator → the big phantom legend).
    const chartRef = this;
    const chartOptions = baseChartOptions();
    chartOptions.plugins.legend.labels.generateLabels = function() {
      return chartRef._legendItems();
    };

    this.chart = new Chart(ctx, {
      type: 'line',
      data: { datasets },
      options: chartOptions
    });
  }

  /**
   * Compute the legend entries from configured intent (independent of whether
   * data has been plotted yet), so the legend is correct at init and after
   * reset. Mirrors what update()/_buildPredictionCurves actually draw:
   *   • measured λ₁..λ_kEigs (the live display count), using their dataset colors
   *   • 2/η threshold (if enabled)
   *   • theory entries from predictionConfig: one per active theory in pooled
   *     mode, or one per (theory × enabled class) in perClass mode, with the
   *     full theory dropped when L ≠ 2 (matches the plotting gate).
   */
  _legendItems() {
    const out = [];

    // Measured eigenvalue curves: the top kEigs, in their fixed colors. Built
    // from stashed style (NOT this.chart.data.datasets), so this works during
    // the first render when this.chart isn't assigned yet. The label notes the
    // active source (Lanczos vs exact) so the curves are never ambiguous.
    const exactActive = this.eigenvalueSource === 'exact' &&
                        this._lastExactHistory && this._lastExactHistory.length > 0;
    const srcTag = exactActive ? ' (exact)' : '';
    for (let i = 0; i < this.kEigs && i < this.maxDisplayEigs; i++) {
      const color = this._eigColors[i] || this._eigColors[i % this._eigColors.length];
      out.push({
        text: this._eigLabel(i) + (i === 0 ? srcTag : ''),
        strokeStyle: color, fillStyle: color,
        lineWidth: 2, lineDash: [],
        hidden: false,
        datasetIndex: this.idx.eigs ? this.idx.eigs[i] : undefined
      });
    }

    // 2/η threshold.
    if (this.showThreshold && this.idx.threshold !== undefined) {
      const t = this._thresholdStyle;
      out.push({
        text: '2/η', strokeStyle: t.color, fillStyle: t.color,
        lineWidth: t.width, lineDash: t.dash,
        hidden: false, datasetIndex: this.idx.threshold
      });
    }

    // Theory overlay entries, derived from predictionConfig (so they show
    // before any plotting). Mirror _buildPredictionCurves' grouping + L gate.
    const cfg = this.predictionConfig;
    if (cfg && cfg.show) {
      const L = this._lastL || 2;
      const theories = cfg.theories.filter(t => t !== 'full' || L === 2);
      for (const theory of theories) {
        const dash = theory === 'full' ? FULL_DASH : GN_DASH;
        const theoryLabel = theory === 'full' ? 'full' : 'GN';
        if (cfg.strategy === 'perClass') {
          const groupNames = theory === 'full' ? FULL_GROUPS : GN_GROUPS;
          for (const group of groupNames) {
            if ((cfg.perClassK[group] || 0) <= 0) continue;
            out.push({
              text: `${GROUP_LABELS[group]} (${theoryLabel})`,
              strokeStyle: CLASS_COLORS[group] || NEUTRAL_PRED_COLOR,
              fillStyle: CLASS_COLORS[group] || NEUTRAL_PRED_COLOR,
              lineWidth: 1.5, lineDash: dash, hidden: false
            });
          }
        } else {
          if (cfg.k > 0) out.push({
            text: `theory (${theoryLabel})`,
            strokeStyle: NEUTRAL_PRED_COLOR, fillStyle: NEUTRAL_PRED_COLOR,
            lineWidth: 1.5, lineDash: dash, hidden: false
          });
        }
      }
    }

    // (The exact spectrum, when selected, is drawn through the measured curves
    // above and annotated "(exact)" on the λ₁ legend label — no separate entry.)

    return out;
  }
  // All setters mutate predictionConfig and repaint from the last σ-history
  // handed to update(), so the UI updates immediately without a new sim frame.

  /** Master on/off for the theory overlay. */
  setShowPrediction(show) {
    this.predictionConfig.show = show;
    this._repaintPrediction();
  }

  /**
   * Choose which eigenvalue history drives the measured curves: 'lanczos' or
   * 'exact'. Live — repaints from cached history without a new sim frame, same
   * as setDisplayK. 'exact' silently falls back to Lanczos if no exact history
   * is present (e.g. exact diagonalization wasn't enabled for this run).
   */
  setEigenvalueSource(source) {
    this.eigenvalueSource = source === 'exact' ? 'exact' : 'lanczos';
    if (this._lastEigHistory) {
      this.update(
        this._lastEigHistory, this.eta, this._lastSigmaHistory,
        this._lastSigmaStar, this._lastN, this._lastD, this._lastM, this._lastL,
        { exactEigenvalueHistory: this._lastExactHistory }
      );
    } else {
      this.chart.update('none');
    }
  }

  /**
   * Set the number of MEASURED eigenvalue curves displayed (live). Clamped to
   * [0, maxDisplayEigs] (the tracked count). Repaints from the cached measured
   * history without needing a new simulation frame.
   */
  setDisplayK(k) {
    const clamped = Math.max(0, Math.min(Math.floor(k), this.maxDisplayEigs));
    this.kEigs = clamped;
    if (this._lastEigHistory) {
      this.update(
        this._lastEigHistory, this.eta, this._lastSigmaHistory,
        this._lastSigmaStar, this._lastN, this._lastD, this._lastM, this._lastL,
        { exactEigenvalueHistory: this._lastExactHistory }
      );
    } else {
      // No plot yet — still refresh so the (intent-based) legend reflects the
      // new display count immediately.
      this.chart.update('none');
    }
  }

  /** Which theories to draw: array subset of ['gn','full']. */
  setTheories(theories) {
    this.predictionConfig.theories = Array.isArray(theories) ? theories.slice() : [];
    this._repaintPrediction();
  }

  /** Convenience: toggle a single theory on/off. */
  setTheoryEnabled(theory, on) {
    const set = new Set(this.predictionConfig.theories);
    if (on) set.add(theory); else set.delete(theory);
    this.predictionConfig.theories = [...set];
    this._repaintPrediction();
  }

  /** Aggregation strategy: 'pooled' | 'perClass'. */
  setPredictionStrategy(strategy) {
    this.predictionConfig.strategy = strategy === 'perClass' ? 'perClass' : 'pooled';
    this._repaintPrediction();
  }

  /** Pooled top-k (per theory). */
  setPooledK(k) {
    this.predictionConfig.k = Math.max(0, Math.floor(k));
    this._repaintPrediction();
  }

  /** Per-class top-k. group -> k. */
  setClassK(group, k) {
    this.predictionConfig.perClassK[group] = Math.max(0, Math.floor(k));
    this._repaintPrediction();
  }

  setColorByClass(on) {
    // Retained as a no-op for API stability; pooled curves are always neutral
    // and perClass curves are always colored by class.
    void on;
  }

  setLogScale(useLog) {
    this.logScale = useLog;
    this.chart.options.scales.y.type = useLog ? 'logarithmic' : 'linear';
    this.chart.update('none');
  }

  setLogScaleX(useLogX) {
    this.logScaleX = useLogX;
    this.chart.options.scales.x.type = useLogX ? 'logarithmic' : 'linear';
    this.chart.options.scales.x.min = useLogX ? 1 : 0;
    this.chart.update('none');
  }

  setEffectiveTime(useEffTime, eta) {
    this.useEffectiveTime = useEffTime;
    this.eta = eta;
  }

  setClipSharpness(clip) {
    this.clipSharpness = clip;
    // Force redraw on next update
    this.chart.update('none');
  }

  setClipToEos(clip) {
    this.clipToEos = clip;
    this.chart.update('none');
  }

  /**
   * Update the chart with measured eigenvalues and the analytic σ trajectory.
   * Theory eigenvalues are DERIVED here from σ via theory_GN / theory_Hessian_full
   * according to the active predictionConfig.
   *
   * @param {Array<{iteration:number, eigs:number[]}>} eigenvalueHistory
   *   Measured eigenvalues (Lanczos), sorted ascending within each `eigs`.
   * @param {number} eta - learning rate (2/η line + x-axis).
   * @param {Array<{iteration:number, sigmas:number[]}>} [sigmaHistory]
   *   Analytic singular-value trajectory (simulation.js sigmaHistory).
   * @param {number[]} [sigmaStar] - target singular values (theory input).
   * @param {number} [n] - output dim (theory input).
   * @param {number} [d] - input dim (theory input).
   * @param {number} [m] - hidden width (theory input; full-Hessian hidden_null
   *   class multiplicity 2·r·(m−r)). Only used at L = 2.
   * @param {number} [L=2] - depth (weight-matrix count). When L !== 2 the full
   *   Hessian theory is unavailable and is never evaluated; only GN is shown.
   */
  update(eigenvalueHistory, eta = this.eta, sigmaHistory = null,
         sigmaStar = null, n = null, d = null, m = null, L = 2, opts = {}) {
    this.eta = eta;
    if (!eigenvalueHistory || eigenvalueHistory.length === 0) return;

    // Cache theory inputs so config-change setters can repaint between frames.
    this._lastEigHistory = eigenvalueHistory;
    this._lastSigmaHistory = sigmaHistory;
    this._lastSigmaStar = sigmaStar;
    this._lastN = n;
    this._lastD = d;
    this._lastM = m;
    this._lastL = (typeof L === 'number' && L > 0) ? L : 2;
    // Exact dense-diagonalization history (opt-in). Cached for live re-slice on
    // setDisplayK. May be null/empty when exact diagonalization isn't active.
    this._lastExactHistory = opts.exactEigenvalueHistory || null;

    const toX = (iter) => this.useEffectiveTime ? iter * this.eta : iter;
    const threshold = 2 / this.eta;

    // Choose which history drives the measured (solid colored) curves. 'exact'
    // uses the dense spectrum when present; otherwise we fall back to Lanczos so
    // the plot is never empty. Both are computed each run — this just picks one.
    const exactHist = this._lastExactHistory;
    const useExact = this.eigenvalueSource === 'exact' && exactHist && exactHist.length > 0;
    const drawHistory = useExact ? exactHist : eigenvalueHistory;
    const dataKEigs = drawHistory[0].eigs.length;

    // 2/η threshold line — spans the union of measured and σ x-ranges.
    if (this.idx.threshold !== undefined) {
      let firstX = toX(eigenvalueHistory[0].iteration);
      let lastX = toX(eigenvalueHistory[eigenvalueHistory.length - 1].iteration);
      if (sigmaHistory && sigmaHistory.length > 0) {
        const pFirst = toX(sigmaHistory[0].iteration);
        const pLast = toX(sigmaHistory[sigmaHistory.length - 1].iteration);
        if (pFirst < firstX) firstX = pFirst;
        if (pLast > lastX) lastX = pLast;
      }
      this.chart.data.datasets[this.idx.threshold].data = [
        { x: firstX, y: threshold },
        { x: lastX, y: threshold }
      ];
    }

    // Measured eigenvalue curves: eigs ascending, so eigs[last] = λ₁. Draw the
    // first this.kEigs (the live display count) from the SELECTED source
    // (Lanczos or exact — see drawHistory above), bounded by what the data
    // holds; hide any remaining allocated datasets (display count may have been
    // lowered, or the data has fewer than kEigs eigenvalues).
    for (let eigIdx = 0; eigIdx < this.idx.eigs.length; eigIdx++) {
      const dsIdx = this.idx.eigs[eigIdx];
      if (eigIdx < this.kEigs && eigIdx < dataKEigs) {
        const eigArrayIdx = dataKEigs - 1 - eigIdx;
        this.chart.data.datasets[dsIdx].data = drawHistory.map(point => ({
          x: toX(point.iteration),
          y: point.eigs[eigArrayIdx]
        }));
        this.chart.data.datasets[dsIdx].hidden = false;
      } else {
        this.chart.data.datasets[dsIdx].data = [];
        this.chart.data.datasets[dsIdx].hidden = true;
      }
    }

    // The legacy separate exact-overlay datasets (idx.exactEigs) are retired:
    // exact eigenvalues now flow through the measured slots above (selected via
    // eigenvalueSource), so they render in the normal solid colored style rather
    // than as a distinct overlay. Keep these datasets permanently hidden/empty.
    for (const dsIdx of this.idx.exactEigs) {
      this.chart.data.datasets[dsIdx].data = [];
      this.chart.data.datasets[dsIdx].hidden = true;
    }
    let lastX = toX(eigenvalueHistory[eigenvalueHistory.length - 1].iteration);
    if (sigmaHistory && sigmaHistory.length > 0) {
      const pLast = toX(sigmaHistory[sigmaHistory.length - 1].iteration);
      if (pLast > lastX) lastX = pLast;
    }
    this.chart.options.scales.x.max = lastX;

    this._repaintPrediction({ skipChartUpdate: true });
    this._rescaleY();
    this.chart.update('none');
  }

  /**
   * Derive and paint the theory overlay from the cached σ-history under the
   * active predictionConfig. Safe to call on config changes between frames.
   */
  _repaintPrediction({ skipChartUpdate = false } = {}) {
    const pool = this.idx.predPool;
    for (const dsIdx of pool) {
      const ds = this.chart.data.datasets[dsIdx];
      ds.data = [];
      ds.hidden = true;
      ds.label = '';
    }

    const cfg = this.predictionConfig;
    const sh = this._lastSigmaHistory;
    const haveInputs = sh && sh.length > 0 && this._lastSigmaStar &&
                       this._lastN != null && this._lastD != null;

    if (cfg.show && haveInputs) {
      const toX = (iter) => this.useEffectiveTime ? iter * this.eta : iter;
      const curves = this._buildPredictionCurves(cfg, toX);
      const count = Math.min(curves.length, pool.length);
      for (let i = 0; i < count; i++) {
        const ds = this.chart.data.datasets[pool[i]];
        ds.data = curves[i].points;
        ds.borderColor = curves[i].color;
        ds.borderDash = curves[i].dash;
        ds.label = curves[i].label || '';
        ds.hidden = false;
      }
      if (curves.length > pool.length) {
        console.warn(`[RightChart] ${curves.length} theory curves requested but ` +
                     `pool holds ${pool.length}; extras dropped. Raise maxPredDatasets.`);
      }
    }

    if (!skipChartUpdate) {
      this._rescaleY();
      this.chart.update('none');
    }
  }

  /**
   * Build curve descriptors { points, color, dash, label } for the active
   * config. Each selected theory ('gn'|'full') is aggregated independently and
   * drawn with its own stroke (GN dotted, full dash-dot). Only the first curve
   * of a legend group carries a label.
   */
  _buildPredictionCurves(cfg, toX) {
    const out = [];
    const sh = this._lastSigmaHistory;
    const sStar = this._lastSigmaStar, n = this._lastN, d = this._lastD;
    const m = this._lastM;
    const L = this._lastL || 2;

    // The full-Hessian theory is 2-layer-only; never evaluate it for L != 2.
    const theories = cfg.theories.filter(t => t !== 'full' || L === 2);

    for (const theory of theories) {
      const dash = theory === 'full' ? FULL_DASH : GN_DASH;
      const theoryLabel = theory === 'full' ? 'full' : 'GN';

      if (cfg.strategy === 'perClass') {
        const perClass = aggregatePerClass(theory, sh, sStar, n, d, m, cfg.perClassK, L);
        const groupNames = theory === 'full' ? FULL_GROUPS : GN_GROUPS;
        for (const group of groupNames) {
          const fam = perClass[group];
          if (!fam) continue;
          const color = CLASS_COLORS[group] || NEUTRAL_PRED_COLOR;
          let first = true;
          for (const curve of fam) {
            if (curve.length === 0) continue;
            out.push({
              points: curve.map(p => ({ x: toX(p.x), y: p.y })),
              color, dash,
              label: first ? `${GROUP_LABELS[group]} (${theoryLabel})` : ''
            });
            first = false;
          }
        }
      } else {
        // pooled — always neutral black; class identity isn't meaningful in a
        // pooled ranking (a curve's origin class can change between timepoints).
        const pooled = aggregatePooled(theory, sh, sStar, n, d, m, cfg.k, L);
        let first = true;
        for (const curve of pooled) {
          if (curve.length === 0) continue;
          out.push({
            points: curve.map(p => ({ x: toX(p.x), y: p.y })),
            color: NEUTRAL_PRED_COLOR, dash,
            label: first ? `theory (${theoryLabel})` : ''
          });
          first = false;
        }
      }
    }
    return out;
  }

  /** Y-axis auto-scale, shared by update() and _repaintPrediction(). */
  _rescaleY() {
    const eigenvalueHistory = this._lastEigHistory;
    if (!eigenvalueHistory || eigenvalueHistory.length === 0) return;
    const threshold = 2 / this.eta;

    if (this.logScale) {
      this.chart.options.scales.y.max = undefined;
      return;
    }

    // Scale to whichever source is actually drawn (see update()): exact when
    // selected and available, else Lanczos.
    const exactHist = this._lastExactHistory;
    const useExact = this.eigenvalueSource === 'exact' && exactHist && exactHist.length > 0;
    const drawHistory = useExact ? exactHist : eigenvalueHistory;

    let maxEig = this.clipToEos ? threshold : 0;
    for (const point of drawHistory) {
      for (const e of point.eigs) if (e > maxEig) maxEig = e;
    }
    if (this.predictionConfig.show) {
      for (const dsIdx of this.idx.predPool) {
        const ds = this.chart.data.datasets[dsIdx];
        if (ds.hidden) continue;
        for (const pt of ds.data) if (pt.y > maxEig) maxEig = pt.y;
      }
    }
    let yMax = maxEig * 1.3;
    if (this.clipSharpness && this.clipToEos) {
      yMax = Math.min(yMax, threshold * 3);
    }
    this.chart.options.scales.y.max = yMax;

    // The exact spectrum is indefinite (can have negative eigenvalues). When the
    // drawn source goes negative, drop the zero-floor so the negative branch is
    // visible; otherwise keep the default zero-floor.
    let minEig = 0;
    for (const point of drawHistory) {
      for (const e of point.eigs) if (e < minEig) minEig = e;
    }
    if (minEig < 0) {
      this.chart.options.scales.y.beginAtZero = false;
      this.chart.options.scales.y.min = minEig * 1.3;
    } else {
      this.chart.options.scales.y.beginAtZero = true;
      this.chart.options.scales.y.min = undefined;
    }
  }

  clear() {
    for (const ds of this.chart.data.datasets) {
      ds.data = [];
    }
    this.chart.options.scales.x.max = undefined;
    this.chart.update('none');
  }
}
