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

const MAX_PLOT_POINTS = 1000;

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
      label: 'theory (Saxe)',
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
    // append a 'theory (Saxe)' entry whenever the prediction overlay is on.
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
          text: 'theory (Saxe)',
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
// sorted descending so ds 0 = λ₁ = largest). Optionally overlays the Saxe
// theory prediction: kEigs additional curves drawn as black dotted lines
// (one per measured curve so the k-th-largest predicted lines up visually
// with the k-th-largest measured).

export class RightChart {
  /**
   * @param {string} canvasId
   * @param {object} [options]
   * @param {number} [options.kEigs=3] - number of eigenvalue curves to show (1-3)
   * @param {boolean} [options.showThreshold=true] - show the 2/η dashed line
   * @param {boolean} [options.showPrediction=true] - overlay Saxe theory prediction
   * @param {boolean} [options.clipSharpness=true] - cap y-max at 3·(2/η) to
   *                                                 prevent one huge spike from
   *                                                 squashing the rest
   * @param {boolean} [options.clipToEos=true] - force the y-axis to include the
   *                                              2/η threshold. Turn off when η
   *                                              is small enough that 2/η is far
   *                                              above the actual eigenvalues
   *                                              and you want to zoom in on the
   *                                              curves themselves.
   */
  constructor(canvasId, options = {}) {
    this.kEigs = options.kEigs || 3;
    this.showThreshold = options.showThreshold !== false;
    this.clipSharpness = options.clipSharpness !== false; // default: true
    this.clipToEos = options.clipToEos !== false; // default: true
    this.showPrediction = options.showPrediction !== false; // default: true

    this.logScale = false;
    this.logScaleX = false;
    this.useEffectiveTime = false;
    this.eta = 0.01;

    const eigColors = [
      'rgb(220, 50, 50)',    // λ₁ (largest) - red
      'rgb(230, 120, 30)',   // λ₂ - orange
      'rgb(80, 180, 80)',    // λ₃ - green
      'rgb(150, 80, 200)',   // λ₄ - purple
      'rgb(40, 130, 180)',   // λ₅ - blue
    ];
    const eigLabels = ['λ₁', 'λ₂', 'λ₃', 'λ₄', 'λ₅'];

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

    // Measured eigenvalue curves (solid, colored).
    this.idx.eigs = [];
    for (let i = 0; i < this.kEigs; i++) {
      this.idx.eigs.push(datasets.length);
      datasets.push({
        label: eigLabels[i] || `λ${i + 1}`,
        data: [],
        borderColor: eigColors[i] || eigColors[0],
        borderWidth: 2,
        pointRadius: 0,
        tension: 0,
        order: i + 1
      });
    }

    // Theory prediction curves (dotted, black). One per measured curve so the
    // k-th-largest predicted overlays the k-th-largest measured. We always
    // build the datasets — visibility is toggled via .hidden.
    this.idx.pred = [];
    for (let i = 0; i < this.kEigs; i++) {
      this.idx.pred.push(datasets.length);
      datasets.push({
        label: i === 0 ? 'theory (Saxe)' : '',  // single legend entry
        data: [],
        borderColor: 'rgb(0, 0, 0)',
        borderWidth: 1.5,
        borderDash: [2, 4],
        pointRadius: 0,
        tension: 0,
        order: 10 + i,                          // draw on top of colored curves
        hidden: !this.showPrediction
      });
    }

    const ctx = document.getElementById(canvasId).getContext('2d');
    this.chart = new Chart(ctx, {
      type: 'line',
      data: { datasets },
      options: baseChartOptions()
    });

    // Custom legend filter so empty-label theory entries don't show up.
    this.chart.options.plugins.legend.labels.filter = function(item) {
      return !!item.text;
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

  setShowPrediction(show) {
    this.showPrediction = show;
    for (const dsIdx of this.idx.pred) {
      this.chart.data.datasets[dsIdx].hidden = !show;
    }
    this.chart.update('none');
  }

  /**
   * Update the chart with eigenvalue history.
   * @param {Array<{iteration: number, eigs: number[]}>} eigenvalueHistory
   *   Measured eigenvalues from Lanczos. Sorted ascending within each `eigs`.
   * @param {number} eta - learning rate (for 2/η line)
   * @param {Array<{iteration: number, eigs: number[]}>} [predictedEigenvalueHistory]
   *   Saxe theory predictions. Sorted *descending* within each `eigs`
   *   (matching theory.js predictedEigenvalues). Optional — if omitted or
   *   empty, the theory curves stay empty.
   */
  update(eigenvalueHistory, eta = this.eta, predictedEigenvalueHistory = null) {
    this.eta = eta;
    if (!eigenvalueHistory || eigenvalueHistory.length === 0) return;

    const toX = (iter) => this.useEffectiveTime ? iter * this.eta : iter;
    const threshold = 2 / this.eta;

    // Number of eigenvalues in the data (may differ from this.kEigs)
    const dataKEigs = eigenvalueHistory[0].eigs.length;

    // 2/η threshold line — spans the union of measured and predicted x-ranges.
    if (this.idx.threshold !== undefined) {
      let firstX = toX(eigenvalueHistory[0].iteration);
      let lastX = toX(eigenvalueHistory[eigenvalueHistory.length - 1].iteration);
      if (predictedEigenvalueHistory && predictedEigenvalueHistory.length > 0) {
        const pFirst = toX(predictedEigenvalueHistory[0].iteration);
        const pLast = toX(predictedEigenvalueHistory[predictedEigenvalueHistory.length - 1].iteration);
        if (pFirst < firstX) firstX = pFirst;
        if (pLast > lastX) lastX = pLast;
      }
      this.chart.data.datasets[this.idx.threshold].data = [
        { x: firstX, y: threshold },
        { x: lastX, y: threshold }
      ];
    }

    // Measured eigenvalue curves: eigs sorted ascending in the data, so
    // eigs[dataKEigs - 1] is the largest (= λ₁). The "k-th largest" goes
    // into dataset slot k (so ds 0 = λ₁, ds 1 = λ₂, etc.).
    for (let eigIdx = 0; eigIdx < this.kEigs; eigIdx++) {
      const dsIdx = this.idx.eigs[eigIdx];
      if (eigIdx < dataKEigs) {
        const eigArrayIdx = dataKEigs - 1 - eigIdx;
        const data = eigenvalueHistory.map(point => ({
          x: toX(point.iteration),
          y: point.eigs[eigArrayIdx]
        }));
        this.chart.data.datasets[dsIdx].data = data;
        this.chart.data.datasets[dsIdx].hidden = false;
      } else {
        this.chart.data.datasets[dsIdx].data = [];
        this.chart.data.datasets[dsIdx].hidden = true;
      }
    }

    // Theory prediction curves. The predicted `eigs` arrays come pre-sorted
    // *descending* (eigs[0] = largest), so the k-th largest is at index k —
    // a small asymmetry with the measured data, which lives ascending. Each
    // theory dataset corresponds to the same rank as the colored curve it
    // overlays.
    if (predictedEigenvalueHistory && predictedEigenvalueHistory.length > 0) {
      const predK = predictedEigenvalueHistory[0].eigs.length;
      for (let eigIdx = 0; eigIdx < this.kEigs; eigIdx++) {
        const dsIdx = this.idx.pred[eigIdx];
        if (eigIdx < predK) {
          const data = predictedEigenvalueHistory.map(point => ({
            x: toX(point.iteration),
            y: point.eigs[eigIdx]
          }));
          this.chart.data.datasets[dsIdx].data = data;
          // hidden controlled by showPrediction, not by data presence
        } else {
          this.chart.data.datasets[dsIdx].data = [];
        }
      }
    } else {
      // No prediction supplied: clear the theory series but leave them in
      // the dataset list so showPrediction can be toggled later.
      for (const dsIdx of this.idx.pred) {
        this.chart.data.datasets[dsIdx].data = [];
      }
    }

    // X-axis max — extend to cover predicted curves too if they run further.
    let lastX = toX(eigenvalueHistory[eigenvalueHistory.length - 1].iteration);
    if (predictedEigenvalueHistory && predictedEigenvalueHistory.length > 0) {
      const pLast = toX(predictedEigenvalueHistory[predictedEigenvalueHistory.length - 1].iteration);
      if (pLast > lastX) lastX = pLast;
    }
    this.chart.options.scales.x.max = lastX;

    // Y-axis auto-scale.
    //
    // Two independent toggles govern the range:
    //   clipToEos     — force the y-axis to include the 2/η threshold.
    //                    Default ON. Turn off when η is small enough that 2/η
    //                    is far above the actual eigenvalues and would
    //                    squash the curves into the bottom of the plot.
    //   clipSharpness — cap y-max at 3·(2/η). Default ON. Prevents one giant
    //                    spike from squashing everything else; only meaningful
    //                    when 2/η is a reasonable reference scale, so the cap
    //                    is skipped when clipToEos is off.
    //
    // Predicted eigenvalues participate in the auto-scale only when the
    // overlay is visible, so toggling it off recovers a tighter view.
    if (!this.logScale) {
      let maxEig = this.clipToEos ? threshold : 0;
      for (const point of eigenvalueHistory) {
        for (const e of point.eigs) {
          if (e > maxEig) maxEig = e;
        }
      }
      if (this.showPrediction && predictedEigenvalueHistory) {
        for (const point of predictedEigenvalueHistory) {
          // predicted eigs are sorted desc; check just the top entry per step.
          if (point.eigs.length > 0 && point.eigs[0] > maxEig) maxEig = point.eigs[0];
        }
      }
      let yMax = maxEig * 1.3;
      if (this.clipSharpness && this.clipToEos) {
        yMax = Math.min(yMax, threshold * 3);
      }
      this.chart.options.scales.y.max = yMax;
    } else {
      this.chart.options.scales.y.max = undefined;
    }

    this.chart.update('none');
  }

  clear() {
    for (const ds of this.chart.data.datasets) {
      ds.data = [];
    }
    this.chart.options.scales.x.max = undefined;
    this.chart.update('none');
  }
}
