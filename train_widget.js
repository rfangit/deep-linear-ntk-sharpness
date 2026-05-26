// ============================================================================
// TRAIN WIDGET — streamlined deep-linear playground
// ============================================================================
// Exported as initWidget(prefix, options) so the same file drives every
// embedded copy on the blog pages:
//
//   • index.html  widget1 — initWidget('tw1', { showLayers: false, simple: true, ... })
//   • index.html  widget2 — initWidget('tw2', { showLayers: true,  simple: true, ... })
//   • residuals.html       — initWidget('tw',  { showLayers: true,  simple: true, ... })
//
// `prefix` is prepended to every element ID with a '-' separator, e.g.
// prefix 'tw1' turns 'lossChart' into 'tw1-lossChart'. Calling with an empty
// prefix leaves IDs unchanged and uses the storage key
// 'mlp-trainer-state-train-widget'; this path is still supported but no
// current entry point uses it.
//
// `options`:
//   storageKey       — localStorage key (defaults to prefix-scoped key)
//   showLayers       — show second/third layer toggles (default true)
//   simple           — when true: hides EMA slider, logscale checkboxes,
//                       step/teff toggle, init-weights panel, and preset button;
//                       locks theory prediction + clip-sharpness on; resets to
//                       defaults in-place instead of reloading the page.
//                       (default false)
//   defaultDepth     — 1, 2, or 3 hidden layers on first load only (ignored if
//                       the widget has persisted state in localStorage).
//   defaultEta       — learning rate on first load only.
//   defaultInitScale — initial-weight scale ε on first load only.
//   defaultAlignedInit — boolean; true to start with weights aligned to the
//                       SVD basis (otherwise random Gaussian).
//   showTheoryControls — boolean; when true, render the theory-overlay control
//                       panel (which theory: GN and/or full Hessian; strategy:
//                       top-k overall vs top-k per class; per-class top-k
//                       inputs) into the page-provided host element with id
//                       `${prefix}-theoryControlContainer`. Off by default, so
//                       index.html's widgets show only the default overlay
//                       (GN, pooled top-kEigs, dotted) and are unchanged. The
//                       legacy name showResidualOption is accepted as an alias.
//   predictionConfig — object passed straight to RightChart to set the initial
//                       theory overlay: { theories:['gn'|'full'...],
//                       strategy:'pooled'|'perClass', k, perClassK:{group:k},
//                       show }. Omit for index.html parity (GN pooled top-kEigs).
//   maxPredDatasets  — cap on simultaneous theory curves (forwarded to
//                       RightChart). Raise for dense per-class views.
//   displayEigs      — number of measured eigenvalue curves shown initially on
//                       the sharpness chart (and listed in the legend at init).
//                       Configurable per widget. Live-adjustable later via the
//                       run-controls panel. Default 3. (`kEigs` is accepted as a
//                       legacy alias.) Note this is independent of trackedEigs
//                       (how many Lanczos computes) — displayed must be ≤ tracked.

import { AppState } from './state.js';
import { Simulation } from './simulation.js';
import { LossChart, RightChart } from './visualization.js';
import { buildMComponentsFromSpec } from './matrix.js';

export function initWidget(prefix = '', options = {}) {
  const el = id => document.getElementById(prefix ? `${prefix}-${id}` : id);

  const showLayers = options.showLayers !== false;
  const simple     = options.simple === true;
  // Expose the full theory-overlay control panel (GN/full, pooled/per-class,
  // per-class top-k). Back-compat: accept the old showResidualOption name too.
  const showTheoryControls = options.showTheoryControls === true || options.showResidualOption === true;
  // Expose the run-parameter panel: live displayed-eigenvalue count + the fixed
  // (pre-run) Lanczos tracked-count and iteration controls.
  const showRunControls = options.showRunControls === true;

  // Display count (live): number of measured eigenvalue curves drawn initially.
  // Configurable per widget via `displayEigs` (preferred) or `kEigs` (legacy
  // alias). Default 3. This is the count shown in the legend at init and the
  // number of measured curves plotted until changed live via the run-controls.
  const displayEigs =
    (typeof options.displayEigs === 'number' && options.displayEigs > 0) ? options.displayEigs
    : (typeof options.kEigs === 'number' && options.kEigs > 0) ? options.kEigs
    : 3;
  const kEigs = displayEigs;  // alias kept for the RightChart/legacy call sites

  // Fixed (pre-run) Lanczos parameters. Tracked count defaults to displayed + 3
  // (extra eigenvalues absorb Lanczos noise so the displayed top ones stay
  // smooth — preserves the original behavior). Iterations default 30 / 100.
  const trackedEigs = (typeof options.trackedEigs === 'number' && options.trackedEigs > 0)
    ? options.trackedEigs : displayEigs + 3;
  const hessianNumIters = (typeof options.hessianNumIters === 'number' && options.hessianNumIters > 0)
    ? options.hessianNumIters : 30;
  const hessianMaxIters = (typeof options.hessianMaxIters === 'number' && options.hessianMaxIters > 0)
    ? options.hessianMaxIters : 100;

  const STORAGE_KEY = options.storageKey ||
    (prefix ? `mlp-trainer-state-${prefix}` : 'mlp-trainer-state-train-widget');

  const appState = new AppState(STORAGE_KEY);
  // Intentionally do NOT call appState.load(): each page refresh should start
  // from defaults, not whatever the previous visit left behind. Subsequent
  // appState.save() calls during this session continue to write to
  // localStorage harmlessly — they just aren't read back on next load.
  const hadPersistedState = false;

  if (!hadPersistedState) {
    appState.eta = (options.defaultEta !== undefined) ? options.defaultEta : 0.6;
    if (options.defaultInitScale !== undefined) {
      appState.initScale = options.defaultInitScale;
    }
    if (options.defaultAlignedInit !== undefined) {
      appState.alignedInit = !!options.defaultAlignedInit;
    }
    // defaultDepth: 1, 2, or 3 hidden layers on first load (no persisted state).
    // Defers to AppState's own defaults if omitted.
    if (options.defaultDepth === 1) { appState.useSecondLayer = false; appState.useThirdLayer = false; }
    if (options.defaultDepth === 2) { appState.useSecondLayer = true;  appState.useThirdLayer = false; }
    if (options.defaultDepth === 3) { appState.useSecondLayer = true;  appState.useThirdLayer = true;  }
  }
  if (!showLayers) { appState.useSecondLayer = false; appState.useThirdLayer = false; }

  let currentM = null, currentU = null, currentV = null;

  function rebuildM() {
    const parts = buildMComponentsFromSpec(appState.matrixSpec, appState.inputDim, appState.outputDim);
    currentM = parts.M; currentU = parts.U; currentV = parts.V;
  }

  // ── Init-weight snapshot (full widget only) ────────────────────────────────
  const NUMBER_COL_WIDTH = 9;
  function formatNumber(x) {
    if (!isFinite(x)) return '   NaN  ';
    const sign = x < 0 ? '-' : '+';
    const s = sign + Math.abs(x).toFixed(4);
    return s.length >= NUMBER_COL_WIDTH ? s : s.padStart(NUMBER_COL_WIDTH, ' ');
  }
  function formatMatrix(M) {
    if (!M.length) return '  (empty)';
    return M.map(row => ' ' + row.map(formatNumber).join(' ')).join('\n');
  }
  function snapshotWeights(W) { return W.map(l => l.map(r => r.slice())); }
  function productMatrix(ws) {
    if (!ws.length) return [];
    let P = ws[0].map(r => r.slice());
    for (let l = 1; l < ws.length; l++) {
      const W = ws[l], rows = W.length, inner = W[0].length, cols = P[0].length;
      P = Array.from({length: rows}, (_, i) => {
        const row = new Array(cols).fill(0);
        for (let k = 0; k < inner; k++) for (let j = 0; j < cols; j++) row[j] += W[i][k] * P[k][j];
        return row;
      });
    }
    return P;
  }

  let initWeightSnapshot = null;

  function snapshotInitWeights() {
    if (!simulation.model?.W) { initWeightSnapshot = null; return; }
    initWeightSnapshot = {
      layers: snapshotWeights(simulation.model.W),
      target: currentM?.map(r => r.slice()) ?? null,
      layerSizes: simulation.model.layerSizes.slice(),
      initScale: appState.initScale,
      alignedInit: !!appState.alignedInit
    };
    renderInitWeights();
  }

  function renderInitWeights() {
    const container = el('initWeightsContainer');
    if (!container) return;
    if (!initWeightSnapshot) {
      container.innerHTML = '<span style="color:#999;">Click <em>start</em> to initialize the model and view its weights here.</span>';
      return;
    }
    const { layers, target, layerSizes, initScale, alignedInit } = initWeightSnapshot;
    const lines = [
      `Layer sizes: ${layerSizes.join(' → ')}   |   Init mode: ${alignedInit ? 'Aligned to SVD basis' : 'Random Gaussian (muP)'}   |   ε = ${initScale}`,
      ''
    ];
    layers.forEach((W, l) => { lines.push(`W_${l+1}   (${W.length} × ${W[0].length})`); lines.push(formatMatrix(W)); lines.push(''); });
    const P = productMatrix(layers);
    if (P.length) { lines.push(`Product W_L · ... · W_1   (${P.length} × ${P[0].length})`); lines.push(formatMatrix(P)); lines.push(''); }
    if (target)   { lines.push(`Target W⋆   (${target.length} × ${target[0].length})`);      lines.push(formatMatrix(target)); }
    container.textContent = lines.join('\n');
  }

  // ── Simulation + Charts ────────────────────────────────────────────────────
  // The Simulation requests slightly more eigenvalues from Lanczos than it
  // actually plots, so spurious noise at the bottom of the requested set
  // doesn't contaminate the displayed top-k. The "buffer" of 3 extras
  // matches the original default (Lanczos: 6, displayed: 3).
  // Mutable live display count (number of measured curves shown). Starts at the
  // option default; the run-controls panel can change it mid-run via setDisplayK.
  let currentDisplayEigs = Math.min(displayEigs, trackedEigs);

  // Fixed (pre-run) toggle: compute the EXACT full-Hessian spectrum via dense
  // diagonalization each step, alongside Lanczos, for ground-truth comparison.
  // Only sensible for small models; set via the run-controls panel and locked
  // on run start. Default off.
  let exactDiagEnabled = options.exactDiag === true;

  const simulation = new Simulation({
    stepsPerSecId: prefix ? `${prefix}-stepsPerSec` : 'stepsPerSec',
    // Fixed (pre-run): how many eigenvalues Lanczos tracks, and its iteration
    // budget. Tracked must be ≥ the max we'd ever display.
    kEigs: trackedEigs,
    hessianNumIters,
    hessianMaxIters
  });

  const lossChart  = new LossChart(prefix ? `${prefix}-lossChart`  : 'lossChart',  { showEma: !simple });
  const rightChart = new RightChart(prefix ? `${prefix}-rightChart` : 'rightChart', {
    kEigs: currentDisplayEigs,
    // Pre-allocate a generous ceiling of measured-curve datasets so the live
    // displayed count (and the run-time tracked count) can grow without
    // reconstructing the chart. An explicit maxEigsCeiling option overrides it.
    maxDisplayEigs: (typeof options.maxEigsCeiling === 'number') ? options.maxEigsCeiling : 20,
    clipSharpness:  true,
    clipToEos:      simple ? false : appState.clipToEos,
    // Theory overlay. Default (no predictionConfig passed) = GN-only, pooled
    // top-kEigs, neutral dotted — exactly the original index.html behavior.
    // residuals.html passes a richer config and exposes the control panel.
    predictionConfig: options.predictionConfig || {
      theories: ['gn'], strategy: 'pooled', k: currentDisplayEigs,
      show: simple ? true : appState.showPrediction
    },
    maxPredDatasets: options.maxPredDatasets
  });

  simulation.onFrameUpdate = () => {
    const state = simulation.getState();
    lossChart.update(state.lossHistory, state.eta, state.predictedLossHistory);
    rightChart.update(
      state.eigenvalueHistory,
      state.eta,
      state.sigmaHistory,
      state.sigmaStar,
      state.outputDim,
      state.inputDim,
      state.hiddenWidth,
      state.theoryL,
      { exactEigenvalueHistory: state.exactEigenvalueHistory }
    );
  };

  // Re-draw the sharpness chart from current state with the new theory-aware
  // update signature. Used by the various plot-control handlers below so they
  // all pass σ-history + theory inputs consistently.
  function refreshRightChart(eta) {
    const state = simulation.getState();
    if (!state.eigenvalueHistory || state.eigenvalueHistory.length === 0) return;
    rightChart.update(
      state.eigenvalueHistory,
      eta !== undefined ? eta : state.eta,
      state.sigmaHistory,
      state.sigmaStar,
      state.outputDim,
      state.inputDim,
      state.hiddenWidth,
      state.theoryL,
      { exactEigenvalueHistory: state.exactEigenvalueHistory }
    );
  }

  simulation.onDiverge = (iteration, loss) => {
    el('startPauseButton').textContent = 'start';
    let errEl = el('divergeError');
    if (!errEl) {
      errEl = document.createElement('div');
      errEl.id = prefix ? `${prefix}-divergeError` : 'divergeError';
      errEl.style.cssText = 'margin-top:12px;padding:10px 16px;background:#fff0f0;border:1px solid #e88;border-radius:6px;color:#a33;font-size:14px;text-align:center;';
      const btn = el('startPauseButton');
      btn.parentElement.parentElement.insertBefore(errEl, btn.parentElement.nextSibling);
    }
    errEl.textContent = `⚠ Training stopped: loss diverged to ${isFinite(loss) ? loss.toExponential(2) : 'NaN/Infinity'} at step ${iteration}. Try a smaller learning rate or smaller singular values.`;
    errEl.style.display = 'block';
  };

  simulation.onAutoStop = (iteration, crossStep, loss) => {
    el('startPauseButton').textContent = 'start';
    console.log(`[auto-stop] Training paused at step ${iteration}. Loss minimum reached at step ${crossStep} (loss = ${loss.toExponential(2)}); stopped ${simulation.minLossPatienceSteps} steps later. Click start to resume.`);
  };

  // ── Utilities ──────────────────────────────────────────────────────────────
  function logSliderToValue(v, min, max) {
    return Math.pow(10, Math.log10(min) + (v / 100) * (Math.log10(max) - Math.log10(min)));
  }
  function valueToLogSlider(v, min, max) {
    return ((Math.log10(v) - Math.log10(min)) / (Math.log10(max) - Math.log10(min))) * 100;
  }

  function wireDetailsToggle(buttonId, panelId, showLabel, hideLabel) {
    const btn = el(buttonId), panel = el(panelId);
    if (!btn || !panel) return;
    btn.addEventListener('click', () => {
      const hidden = !panel.style.display || panel.style.display === 'none';
      panel.style.display = hidden ? 'block' : 'none';
      btn.textContent = hidden ? hideLabel : showLabel;
    });
  }

  // ── Data params ────────────────────────────────────────────────────────────
  function renderDataParams() {
    const container = el('dataParamsContainer');
    container.innerHTML = '';
    function addField(labelText, key, attrs) {
      const cell = document.createElement('div'); cell.className = 'inline-field';
      const lbl  = document.createElement('span'); lbl.textContent = labelText;
      const inp  = document.createElement('input'); inp.type = 'number';
      Object.assign(inp, attrs); inp.value = appState[key];
      inp.addEventListener('change', () => {
        let v = parseFloat(inp.value);
        if (!isFinite(v)) v = appState[key];
        if (attrs.min !== undefined) v = Math.max(attrs.min, v);
        if (attrs.max !== undefined) v = Math.min(attrs.max, v);
        if (attrs.step === 1) v = Math.round(v);
        appState[key] = v; inp.value = v; appState.save(); validateInputMode();
      });
      cell.appendChild(lbl); cell.appendChild(inp); return cell;
    }
    container.appendChild(addField('Training Points $N$:', 'nTrain',   { min: 1, max: 10000, step: 1 }));
    container.appendChild(addField('Data Seed:',           'dataSeed', { min: 0, step: 1 }));
    if (window.MathJax?.typesetPromise) MathJax.typesetPromise([container]).catch(console.log);
  }

  // ── Input mode ─────────────────────────────────────────────────────────────
  function initInputModeControls() {
    const whitened = el('inputModeWhitened'), gaussian = el('inputModeGaussian');
    whitened.checked = appState.inputMode === 'whitened';
    gaussian.checked = appState.inputMode === 'gaussian';
    whitened.addEventListener('change', () => { if (whitened.checked) { appState.inputMode = 'whitened'; appState.save(); validateInputMode(); } });
    gaussian.addEventListener('change', () => { if (gaussian.checked) { appState.inputMode = 'gaussian'; appState.save(); validateInputMode(); } });
  }

  function validateInputMode() {
    const warn = el('inputModeWarning');
    const ok   = appState.inputMode !== 'whitened' || appState.nTrain >= appState.inputDim;
    if (!ok) {
      warn.style.display = 'block';
      warn.innerHTML = `⚠ <strong>whitened inputs require N ≥ k.</strong> You have N=${appState.nTrain} training points and k=${appState.inputDim} input dim — increase N or switch to iid Gaussian.`;
    } else {
      warn.style.display = 'none';
    }
    return ok;
  }

  // ── Init mode ──────────────────────────────────────────────────────────────
  function initInitModeControls() {
    const randRadio    = el('initModeRandom');
    const alignRadio   = el('initModeAligned');
    const randomORow   = el('randomORow');
    const randomOCb    = el('randomOCheckbox');
    const initScaleLbl = el('initScaleLabel');
    const weightCap    = el('weightCaption');

    randRadio.checked  = !appState.alignedInit;
    alignRadio.checked =  appState.alignedInit;
    randomOCb.checked  = !!appState.randomO;

    function updateInitModeDependentUI() {
      const aligned = !!appState.alignedInit;
      randomORow.style.display = aligned ? 'flex' : 'none';
      if (initScaleLbl) initScaleLbl.innerHTML = 'init scale $\\varepsilon$';
      if (weightCap) {
        weightCap.innerHTML = aligned
          ? 'Weights aligned in SVD basis: each layer has singular value $\\varepsilon$ per mode, so the product matrix has singular values $\\varepsilon^L$. (Matches the typical scale of $W_{ij} \\sim \\mathcal{N}(0, \\varepsilon^2/n_{\\ell-1})$.)'
          : 'Weights: $W_{ij}^{(\\ell)} \\sim \\mathcal{N}(0,\\, \\varepsilon^2/n_{\\ell-1})$';
      }
      if (window.MathJax?.typesetPromise) {
        MathJax.typesetPromise([initScaleLbl, weightCap].filter(Boolean)).catch(console.log);
      }
    }

    randRadio.addEventListener('change',  () => { if (randRadio.checked)  { appState.alignedInit = false; appState.save(); updateInitModeDependentUI(); } });
    alignRadio.addEventListener('change', () => { if (alignRadio.checked) { appState.alignedInit = true;  appState.save(); updateInitModeDependentUI(); } });
    randomOCb.addEventListener('change',  () => { appState.randomO = randomOCb.checked; appState.save(); });

    const showInitWtCb  = el('showInitWeightsCheckbox');
    const initWtContainer = el('initWeightsContainer');
    if (showInitWtCb && initWtContainer) {
      showInitWtCb.checked = !!appState.showInitWeights;
      initWtContainer.style.display = appState.showInitWeights ? 'block' : 'none';
      if (appState.showInitWeights) renderInitWeights();
      showInitWtCb.addEventListener('change', () => {
        appState.showInitWeights = showInitWtCb.checked;
        initWtContainer.style.display = showInitWtCb.checked ? 'block' : 'none';
        if (showInitWtCb.checked) renderInitWeights();
        appState.save();
      });
    }

    updateInitModeDependentUI();
  }

  // ── Dimension inputs ───────────────────────────────────────────────────────
  function renderDimensionInputs() {
    const container = el('dimInputs');
    container.innerHTML = '';
    function makeDimInput(key, labelHTML, max) {
      const cell = document.createElement('div'); cell.className = 'inline-field';
      const label = document.createElement('span'); label.innerHTML = labelHTML;
      const input = document.createElement('input');
      input.type = 'number'; input.min = 1; input.max = max; input.step = 1; input.value = appState[key];
      input.addEventListener('change', () => {
        let v = Math.max(1, Math.min(max, parseInt(input.value, 10) || appState[key]));
        appState[key] = v; input.value = v;
        ensureSingularValuesLength(); rebuildM(); appState.save();
        renderSingularValueEditor(); validateDimensions(); validateInputMode();
        // Per-class caps (r, single_value, hidden_null, …) depend on n and d.
        rightChart._theoryControlsRefreshCaps?.();
      });
      cell.appendChild(label); cell.appendChild(input); return cell;
    }
    container.appendChild(makeDimInput('inputDim',  'Input Dim <i>k</i>:',  200));
    container.appendChild(makeDimInput('outputDim', 'Output Dim <i>m</i>:', 50));
  }

  // ── Basis options ──────────────────────────────────────────────────────────
  function renderBasisOptions() {
    const container = el('basisOptions');
    container.innerHTML = '';

    const cbCell = document.createElement('label'); cbCell.className = 'inline-field';
    const cb = document.createElement('input'); cb.type = 'checkbox';
    cb.checked = appState.matrixSpec.randomBasis !== false;
    cb.addEventListener('change', () => { appState.matrixSpec.randomBasis = cb.checked; rebuildM(); appState.save(); });
    cbCell.appendChild(cb);
    const cbLabel = document.createElement('span');
    cbLabel.textContent = 'Random Orthogonal $U, V$ (otherwise $W^\\star = \\Sigma$)';
    cbCell.appendChild(cbLabel);
    container.appendChild(cbCell);

    const seedCell = document.createElement('div'); seedCell.className = 'inline-field';
    const seedLabel = document.createElement('span'); seedLabel.textContent = 'Basis Seed:';
    const seedInput = document.createElement('input');
    seedInput.type = 'number'; seedInput.min = 0; seedInput.step = 1;
    seedInput.value = appState.matrixSpec.basisSeed || 0;
    seedInput.addEventListener('change', () => {
      let v = parseInt(seedInput.value, 10); if (!isFinite(v) || v < 0) v = 0;
      appState.matrixSpec.basisSeed = v; seedInput.value = v; rebuildM(); appState.save();
    });
    seedCell.appendChild(seedLabel); seedCell.appendChild(seedInput);
    container.appendChild(seedCell);

    if (window.MathJax?.typesetPromise) MathJax.typesetPromise([container]).catch(console.log);
  }

  // ── Singular value editor ──────────────────────────────────────────────────
  function ensureSingularValuesLength() {
    const spec = appState.matrixSpec, m = appState.outputDim;
    if (!Array.isArray(spec.singularValues)) spec.singularValues = [];
    if (spec.singularValues.length > m) {
      spec.singularValues = spec.singularValues.slice(0, m);
    } else {
      while (spec.singularValues.length < m) {
        const i = spec.singularValues.length;
        const last = i > 0 ? spec.singularValues[i - 1] : 1.0;
        let next = +(last * 0.7).toFixed(4);
        if (next <= 0) next = 1 / (i + 1);
        spec.singularValues.push(next);
      }
    }
  }

  function renderSingularValueEditor() {
    const editor = el('svEditor'); editor.innerHTML = '';
    const spec = appState.matrixSpec;
    for (let i = 0; i < appState.outputDim; i++) {
      const cell = document.createElement('div'); cell.className = 'sv-row';
      const label = document.createElement('div'); label.className = 'sv-label';
      label.innerHTML = `&sigma;<sub>${i + 1}</sub>:`;
      const input = document.createElement('input');
      input.type = 'number'; input.step = 'any'; input.value = spec.singularValues[i];
      input.addEventListener('input', () => {
        const raw = input.value.trim(); if (raw === '') return;
        const v = parseFloat(raw); if (!isFinite(v)) return;
        spec.singularValues[i] = v < 0 ? 0 : v; rebuildM(); appState.save();
      });
      cell.appendChild(label); cell.appendChild(input); editor.appendChild(cell);
    }
  }

  function applySVPreset(name) {
    if (name !== 'powers-of-2') return;
    const spec = appState.matrixSpec, m = appState.outputDim;
    spec.singularValues = Array.from({length: m}, (_, i) => +(Math.pow(0.5, i)).toFixed(4));
    rebuildM(); appState.save(); renderSingularValueEditor();
  }

  function bindSVPresetButtons() {
    const scope = el('functionDetailsPanel');
    if (!scope) return;
    scope.querySelectorAll('[data-sv-preset]').forEach(btn => {
      btn.addEventListener('click', () => applySVPreset(btn.dataset.svPreset));
    });
  }

  function validateDimensions() {
    const warning = el('svWarning');
    if (appState.inputDim < appState.outputDim) {
      warning.style.display = 'block';
      warning.innerHTML = `⚠ <strong>input dim (${appState.inputDim}) &lt; output dim (${appState.outputDim}).</strong> ` +
        `The matrix can have at most ${appState.inputDim} non-zero singular values; the last ${appState.outputDim - appState.inputDim} &sigma; entries will be ignored.`;
    } else {
      warning.style.display = 'none';
    }
  }

  // ── Model controls ─────────────────────────────────────────────────────────
  function initModelControls() {
    const dim1Slider = el('hiddenDim1Slider');
    const dim1Value  = el('hiddenDim1Value');
    const seedInput  = el('modelSeedInput');

    dim1Slider.min = 1; dim1Slider.max = 200;
    appState.hiddenDim1 = Math.max(1, Math.min(200, appState.hiddenDim1));
    dim1Slider.value = appState.hiddenDim1;
    dim1Value.textContent = appState.hiddenDim1;
    seedInput.value = appState.modelSeed;

    dim1Slider.addEventListener('input', () => {
      appState.hiddenDim1 = parseInt(dim1Slider.value);
      dim1Value.textContent = appState.hiddenDim1; appState.save();
      // hidden_null cap = 2·r·(m−r) depends on m = hiddenDim1; keep it current.
      rightChart._theoryControlsRefreshCaps?.();
    });
    seedInput.addEventListener('change', () => {
      appState.modelSeed = parseInt(seedInput.value) || 0;
      seedInput.value = appState.modelSeed; appState.save();
    });

    const layer2Cb   = el('useSecondLayerCheckbox');
    const dim2Row    = el('hiddenDim2Row');
    const dim2Slider = el('hiddenDim2Slider');
    const dim2Value  = el('hiddenDim2Value');
    const layer3Cb   = el('useThirdLayerCheckbox');
    const dim3Row    = el('hiddenDim3Row');
    const dim3Slider = el('hiddenDim3Slider');
    const dim3Value  = el('hiddenDim3Value');

    if (layer2Cb) {
      dim2Slider.min = 1; dim2Slider.max = 200;
      layer2Cb.checked = appState.useSecondLayer;
      dim2Row.style.display = appState.useSecondLayer ? 'flex' : 'none';
      dim2Slider.value = appState.hiddenDim2;
      dim2Value.textContent = appState.hiddenDim2;
      layer2Cb.addEventListener('change', () => {
        appState.useSecondLayer = layer2Cb.checked;
        dim2Row.style.display = appState.useSecondLayer ? 'flex' : 'none';
        if (layer3Cb) {
          layer3Cb.disabled = !appState.useSecondLayer;
          if (!appState.useSecondLayer) {
            appState.useThirdLayer = false; layer3Cb.checked = false; dim3Row.style.display = 'none';
          }
        }
        appState.save();
      });
      dim2Slider.addEventListener('input', () => {
        appState.hiddenDim2 = parseInt(dim2Slider.value);
        dim2Value.textContent = appState.hiddenDim2; appState.save();
      });
    }

    if (layer3Cb) {
      dim3Slider.min = 1; dim3Slider.max = 200;
      layer3Cb.checked  = appState.useThirdLayer;
      layer3Cb.disabled = !appState.useSecondLayer;
      dim3Row.style.display = (appState.useSecondLayer && appState.useThirdLayer) ? 'flex' : 'none';
      dim3Slider.value = appState.hiddenDim3;
      dim3Value.textContent = appState.hiddenDim3;
      layer3Cb.addEventListener('change', () => {
        appState.useThirdLayer = layer3Cb.checked;
        dim3Row.style.display = (appState.useSecondLayer && appState.useThirdLayer) ? 'flex' : 'none';
        appState.save();
      });
      dim3Slider.addEventListener('input', () => {
        appState.hiddenDim3 = parseInt(dim3Slider.value);
        dim3Value.textContent = appState.hiddenDim3; appState.save();
      });
    }
  }

  // ── Training controls ──────────────────────────────────────────────────────
  const INIT_SCALE_SLIDER_MIN = 0.0001, INIT_SCALE_SLIDER_MAX = 1;

  function setInitScaleUI(value) {
    el('initScaleNumber').value = parseFloat(value.toPrecision(6));
    const clamped = Math.max(INIT_SCALE_SLIDER_MIN, Math.min(INIT_SCALE_SLIDER_MAX, value));
    el('initScaleSlider').value = valueToLogSlider(clamped, INIT_SCALE_SLIDER_MIN, INIT_SCALE_SLIDER_MAX);
  }

  function initTrainingControls() {
    el('etaSlider').value = valueToLogSlider(appState.eta, 0.01, 10);
    el('etaValue').textContent = parseFloat(appState.eta.toPrecision(4));
    setInitScaleUI(appState.initScale);

    el('etaSlider').addEventListener('input', () => {
      appState.eta = parseFloat(logSliderToValue(parseFloat(el('etaSlider').value), 0.01, 10).toPrecision(4));
      el('etaValue').textContent = appState.eta; appState.save();
    });
    el('initScaleSlider').addEventListener('input', () => {
      const v = parseFloat(logSliderToValue(parseFloat(el('initScaleSlider').value), INIT_SCALE_SLIDER_MIN, INIT_SCALE_SLIDER_MAX).toPrecision(4));
      appState.initScale = v; el('initScaleNumber').value = parseFloat(v.toPrecision(6)); appState.save();
    });
    el('initScaleNumber').addEventListener('change', () => {
      let v = parseFloat(el('initScaleNumber').value);
      if (!isFinite(v) || v <= 0) { setInitScaleUI(appState.initScale); return; }
      appState.initScale = v; setInitScaleUI(v); appState.save();
    });
  }

  // ── Start / Pause / Reset ──────────────────────────────────────────────────
  function initButtons() {
    const startPauseButton = el('startPauseButton');

    startPauseButton.addEventListener('click', () => {
      if (!simulation.isRunning) {
        if (simulation.model) { simulation.start(); startPauseButton.textContent = 'pause'; return; }
        const errEl = el('divergeError');
        if (errEl) errEl.style.display = 'none';
        if (!validateInputMode()) return;
        if (!currentM) rebuildM();
        simulation.captureParams({
          M: currentM, dataSeed: appState.dataSeed, nTrain: appState.nTrain,
          initScale: appState.initScale, hiddenDims: appState.hiddenDims(),
          eta: appState.eta, modelSeed: appState.modelSeed, inputMode: appState.inputMode,
          sigmaStar: appState.matrixSpec.singularValues,
          alignedInit: appState.alignedInit, randomO: appState.randomO,
          U: currentU, V: currentV,
          hiddenWidth: appState.hiddenDim1,
          exactDiag: exactDiagEnabled
        });
        const r = Math.min(appState.inputDim, appState.outputDim);
        let frobSq = 0;
        for (let i = 0; i < r; i++) { const s = appState.matrixSpec.singularValues[i] || 0; frobSq += s * s; }
        lossChart.setInitialLoss(0.5 * frobSq);
        simulation.start();
        runControlsLock();
        if (!simple) snapshotInitWeights();
        startPauseButton.textContent = 'pause';
      } else {
        simulation.pause(); startPauseButton.textContent = 'start';
      }
    });

    el('resetButton').addEventListener('click', () => {
      simulation.reset(); lossChart.clear(); rightChart.clear();
      startPauseButton.textContent = 'start';
      runControlsUnlock();
      const errEl = el('divergeError');
      if (errEl) errEl.style.display = 'none';
      if (!simple) { initWeightSnapshot = null; renderInitWeights(); }
    });

    el('resetToDefaultsButton')?.addEventListener('click', () => {
      try { localStorage.removeItem(STORAGE_KEY); } catch (e) {}
      if (simple) {
        appState.resetToDefaults();
        if (!showLayers) { appState.useSecondLayer = false; appState.useThirdLayer = false; }
        appState.eta = 0.6;
        simulation.reset(); lossChart.clear(); rightChart.clear();
        startPauseButton.textContent = 'start';
        runControlsUnlock();
        renderDataParams(); initInputModeControls(); initInitModeControls();
        renderDimensionInputs(); renderBasisOptions();
        ensureSingularValuesLength(); rebuildM();
        renderSingularValueEditor(); initModelControls(); initTrainingControls();
        validateDimensions(); validateInputMode();
      } else {
        location.reload();
      }
    });
  }

  // ── Plot controls (full widget only) ──────────────────────────────────────
  function initPlotControls() {
    if (simple) return;

    const logScaleCb      = el('logScaleCheckbox');
    const logScaleXCb     = el('logScaleXCheckbox');
    const clipSharpnessCb = el('clipSharpnessCheckbox');
    const clipToEosCb     = el('clipToEosCheckbox');
    const showPredCb      = el('showPredictionCheckbox');
    const stepLink        = el('step-link');
    const teffLink        = el('teff-link');
    const emaSliderEl     = el('emaSlider');
    const emaValueEl      = el('emaValue');

    logScaleCb.checked = appState.logScale;
    logScaleCb.addEventListener('change', () => {
      appState.logScale = logScaleCb.checked;
      lossChart.setLogScale(appState.logScale); rightChart.setLogScale(appState.logScale); appState.save();
    });

    logScaleXCb.checked = appState.logScaleX;
    logScaleXCb.addEventListener('change', () => {
      appState.logScaleX = logScaleXCb.checked;
      lossChart.setLogScaleX(appState.logScaleX); rightChart.setLogScaleX(appState.logScaleX); appState.save();
    });

    clipSharpnessCb.addEventListener('change', () => {
      rightChart.setClipSharpness(clipSharpnessCb.checked);
      const state = simulation.getState();
      if (state.eigenvalueHistory.length > 0) refreshRightChart(state.eta);
    });

    clipToEosCb.checked = appState.clipToEos;
    rightChart.setClipToEos(appState.clipToEos);
    clipToEosCb.addEventListener('change', () => {
      appState.clipToEos = clipToEosCb.checked;
      rightChart.setClipToEos(appState.clipToEos);
      const state = simulation.getState();
      if (state.eigenvalueHistory.length > 0) refreshRightChart(state.eta);
      appState.save();
    });

    showPredCb.checked = appState.showPrediction;
    rightChart.setShowPrediction(appState.showPrediction);
    lossChart.setShowPrediction(appState.showPrediction);
    showPredCb.addEventListener('change', () => {
      appState.showPrediction = showPredCb.checked;
      rightChart.setShowPrediction(appState.showPrediction);
      lossChart.setShowPrediction(appState.showPrediction);
      appState.save();
    });

    function setXAxisMode(mode) {
      appState.xAxisMode = mode;
      const useEff = mode === 'teff';
      lossChart.setEffectiveTime(useEff, appState.eta); rightChart.setEffectiveTime(useEff, appState.eta);
      stepLink.classList.toggle('active', mode === 'step'); teffLink.classList.toggle('active', mode === 'teff');
      const state = simulation.getState();
      if (state.lossHistory.length > 0) {
        lossChart.update(state.lossHistory, appState.eta, state.predictedLossHistory);
        refreshRightChart(appState.eta);
      }
      appState.save();
    }
    stepLink.addEventListener('click', e => { e.preventDefault(); setXAxisMode('step'); });
    teffLink.addEventListener('click', e => { e.preventDefault(); setXAxisMode('teff'); });

    function emaSliderToWindow(val) { return val === 0 ? 1 : Math.round(Math.pow(10, (val / 100) * 4)); }
    function emaWindowToSlider(w)   { return w <= 1 ? 0 : (Math.log10(w) / 4) * 100; }

    emaSliderEl.value = emaWindowToSlider(appState.emaWindow);
    emaValueEl.textContent = appState.emaWindow <= 1 ? 'off' : appState.emaWindow;
    emaSliderEl.addEventListener('input', () => {
      const w = emaSliderToWindow(parseInt(emaSliderEl.value));
      appState.emaWindow = w; emaValueEl.textContent = w <= 1 ? 'off' : w;
      lossChart.setEmaWindow(w);
      const state = simulation.getState();
      if (state.lossHistory.length > 0) lossChart.update(state.lossHistory, appState.eta, state.predictedLossHistory);
      appState.save();
    });

    if (appState.logScale)  { lossChart.setLogScale(true);  rightChart.setLogScale(true); }
    if (appState.logScaleX) { lossChart.setLogScaleX(true); rightChart.setLogScaleX(true); }
    if (appState.xAxisMode === 'teff') {
      lossChart.setEffectiveTime(true, appState.eta); rightChart.setEffectiveTime(true, appState.eta);
      stepLink.classList.remove('active'); teffLink.classList.add('active');
    }
  }

  // ── Run-parameter controls (only when showRunControls: true) ──────────────
  // Builds a panel into #${prefix}-runControlContainer with:
  //   • Displayed eigenvalues (LIVE): number of measured curves drawn. Clamped
  //     to ≤ tracked. Applied immediately via rightChart.setDisplayK.
  //   • Lanczos eigenvalues tracked (FIXED, pre-run): how many Lanczos returns.
  //   • Lanczos iterations (FIXED, pre-run): numIters (with maxIters scaled).
  //   • Exact diagonalization (FIXED, pre-run): compute the dense full spectrum
  //     each step alongside Lanczos (small models only).
  //   • Plot exact spectrum (LIVE): when exact diag is on, switch the displayed
  //     curves between Lanczos and the exact dense spectrum via
  //     rightChart.setEigenvalueSource — both are computed, this only picks what
  //     is drawn. Stays live during a run; disabled when exact diag is off.
  // The fixed inputs write to simulation.hessianOptions / params and lock once
  // the run has started; reset unlocks them. The two LIVE controls never lock.
  let runControlsLock = () => {};
  let runControlsUnlock = () => {};

  function initRunControls() {
    if (!showRunControls) return;
    const host = el('runControlContainer');
    if (!host) {
      console.warn('[train_widget] showRunControls: true but no host element ' +
                   `#${prefix}-runControlContainer found.`);
      return;
    }
    host.innerHTML = '';
    const wrap = document.createElement('div');
    wrap.style.cssText = 'display:flex; flex-direction:column; gap:10px; font-size:14px;';

    // Displayed eigenvalues (live).
    const dispRow = document.createElement('div');
    dispRow.className = 'inline-field';
    const dispNum = mkNumber(currentDisplayEigs, 0, trackedEigs);
    dispRow.appendChild(document.createTextNode('Displayed eigenvalues '));
    dispRow.appendChild(dispNum);
    const dispNote = document.createElement('span');
    dispNote.style.cssText = 'font-size:12px; color:#999;';
    dispNote.textContent = `(live, max ${trackedEigs})`;
    dispRow.appendChild(dispNote);
    wrap.appendChild(dispRow);

    // Lanczos eigenvalues tracked (fixed).
    const trackRow = document.createElement('div');
    trackRow.className = 'inline-field';
    const trackNum = mkNumber(simulation.hessianOptions.kEigs, 1, 50);
    trackRow.appendChild(document.createTextNode('Lanczos eigenvalues '));
    trackRow.appendChild(trackNum);
    const trackNote = document.createElement('span');
    trackNote.style.cssText = 'font-size:12px; color:#999;';
    trackNote.textContent = '(fixed before run)';
    trackRow.appendChild(trackNote);
    wrap.appendChild(trackRow);

    // Lanczos iterations (fixed).
    const iterRow = document.createElement('div');
    iterRow.className = 'inline-field';
    const iterNum = mkNumber(simulation.hessianOptions.numIters, 1, 500);
    iterRow.appendChild(document.createTextNode('Lanczos iterations '));
    iterRow.appendChild(iterNum);
    const iterNote = document.createElement('span');
    iterNote.style.cssText = 'font-size:12px; color:#999;';
    iterNote.textContent = '(fixed before run)';
    iterRow.appendChild(iterNote);
    wrap.appendChild(iterRow);

    // Exact dense diagonalization (fixed, pre-run). Only sensible for small
    // models; computes the full Hessian + all eigenvalues each step alongside
    // Lanczos, plotted as a distinct overlay for ground-truth comparison.
    const exactRow = document.createElement('div');
    exactRow.className = 'inline-field';
    const exactCb = document.createElement('input');
    exactCb.type = 'checkbox';
    exactCb.checked = exactDiagEnabled;
    const exactLbl = document.createElement('span');
    exactLbl.textContent = ' Exact diagonalization ';
    const exactNote = document.createElement('span');
    exactNote.style.cssText = 'font-size:12px; color:#999;';
    exactNote.textContent = '(fixed; small models only)';
    exactRow.appendChild(exactCb);
    exactRow.appendChild(exactLbl);
    exactRow.appendChild(exactNote);
    wrap.appendChild(exactRow);

    // Plot source (LIVE): when exact diagonalization is enabled, switch the
    // displayed eigenvalue curves between the Lanczos top-k and the exact dense
    // spectrum without resetting — both are computed each step, this only picks
    // which one is drawn (like the displayed-count control). Disabled/greyed
    // when exact diag is off (nothing to switch to).
    const srcRow = document.createElement('div');
    srcRow.className = 'inline-field';
    const srcCb = document.createElement('input');
    srcCb.type = 'checkbox';
    srcCb.checked = false;                 // default: show Lanczos
    const srcLbl = document.createElement('span');
    srcLbl.textContent = ' Plot exact spectrum (vs Lanczos) ';
    const srcNote = document.createElement('span');
    srcNote.style.cssText = 'font-size:12px; color:#999;';
    srcNote.textContent = '(live)';
    srcRow.appendChild(srcCb);
    srcRow.appendChild(srcLbl);
    srcRow.appendChild(srcNote);
    wrap.appendChild(srcRow);

    // Enable the source toggle only when exact diag is on; keep it in sync if
    // the exact-diag checkbox changes before a run.
    const syncSourceRow = () => {
      const on = exactCb.checked;
      srcCb.disabled = !on;
      srcRow.style.opacity = on ? '1' : '0.45';
      if (!on) {
        // Force back to Lanczos when exact diag is disabled.
        srcCb.checked = false;
        rightChart.setEigenvalueSource('lanczos');
      }
    };

    const lockedNote = document.createElement('div');
    lockedNote.style.cssText = 'font-size:12px; color:#888;';
    wrap.appendChild(lockedNote);

    host.appendChild(wrap);

    // Displayed count: live. Clamp ≤ tracked, push to chart immediately.
    dispNum.addEventListener('change', () => {
      let v = Math.max(0, Math.round(parseFloat(dispNum.value) || 0));
      v = Math.min(v, simulation.hessianOptions.kEigs);
      dispNum.value = v;
      currentDisplayEigs = v;
      rightChart.setDisplayK(v);
    });

    // Tracked eigenvalues: fixed. Write to hessianOptions (pre-run only). Also
    // raise the chart's max display ceiling and the dance with maxDisplayEigs:
    // since the chart pre-allocated maxDisplayEigs datasets at construction, the
    // tracked count can only be lowered live, not raised beyond the initial
    // ceiling. So changing it requires a reset to take full effect — we update
    // hessianOptions and clamp the display note, and rely on the lock to enforce
    // that this only happens before a run.
    trackNum.addEventListener('change', () => {
      let v = Math.max(1, Math.round(parseFloat(trackNum.value) || 1));
      trackNum.value = v;
      simulation.hessianOptions.kEigs = v;
      // keep displayed ≤ tracked
      dispNum.max = v;
      dispNote.textContent = `(live, max ${v})`;
      if (currentDisplayEigs > v) { currentDisplayEigs = v; dispNum.value = v; rightChart.setDisplayK(v); }
    });

    iterNum.addEventListener('change', () => {
      let v = Math.max(1, Math.round(parseFloat(iterNum.value) || 1));
      iterNum.value = v;
      simulation.hessianOptions.numIters = v;
      // keep maxIters a sensible ceiling above numIters
      simulation.hessianOptions.maxIters = Math.max(simulation.hessianOptions.maxIters, v + 20, v * 2);
    });

    exactCb.addEventListener('change', () => {
      exactDiagEnabled = exactCb.checked;
      syncSourceRow();
    });

    // Plot-source toggle is LIVE (both spectra are already computed). 'exact'
    // falls back to Lanczos in the chart if no exact history exists yet.
    srcCb.addEventListener('change', () => {
      rightChart.setEigenvalueSource(srcCb.checked ? 'exact' : 'lanczos');
    });

    runControlsLock = () => {
      // Fixed pre-run settings lock during a run; the plot-source toggle (srcCb)
      // stays live, since switching what's displayed is safe mid-run.
      trackNum.disabled = true;  iterNum.disabled = true;  exactCb.disabled = true;
      trackNum.style.opacity = '0.5'; iterNum.style.opacity = '0.5'; exactRow.style.opacity = '0.6';
      lockedNote.textContent = 'Lanczos / exact-diag settings are locked during a run — reset to change them.';
      syncSourceRow();   // keep source toggle enabled iff exact diag is on
    };
    runControlsUnlock = () => {
      trackNum.disabled = false; iterNum.disabled = false; exactCb.disabled = false;
      trackNum.style.opacity = '1'; iterNum.style.opacity = '1'; exactRow.style.opacity = '1';
      lockedNote.textContent = '';
      syncSourceRow();
    };
    // If a run is already underway (e.g. re-init), reflect lock state.
    if (simulation.model) runControlsLock(); else runControlsUnlock();
  }


  // Builds the full control panel into the page-provided host container:
  //   • Which theory: GN and/or full Hessian (checkboxes).
  //   • Strategy: pooled top-k overall, or per-class top-k.
  //   • Pooled-k input (pooled strategy).
  //   • Per-class rows: one per group (aligned, aligned_null, cross,
  //     single_value, hidden_null, idle_null), each a checkbox + k input.
  //     Default k = 3, max clamped to the group's mode count for the current
  //     dims. aligned_null, hidden_null, and idle_null exist only in the full
  //     theory; their rows are disabled when 'full' isn't selected. The two
  //     hidden-null rows come from the surplus hidden units: hidden_null (the ±
  //     nonzero branches, cap 2·r·(m−r)) and idle_null (the flat zero line, cap
  //     (m−r)·(n+d−2r)). Both are empty (max 0) unless m > r.
  // A small notice flags the L ≠ 2 case since the theory is 2-layer-specific.
  const THEORY_GROUPS = ['aligned', 'aligned_null', 'cross', 'single_value', 'hidden_null', 'idle_null'];
  const THEORY_GROUP_LABELS = {
    aligned: 'aligned', aligned_null: 'aligned (null)',
    cross: 'cross', single_value: 'single-value',
    hidden_null: 'hidden null (±)', idle_null: 'hidden null (0)'
  };

  // Mode count per group for the current dims (caps the per-class k inputs).
  // r = min(n,d); n = outputDim, d = inputDim; m = first hidden width
  // (hiddenDim1). Matches theory.js group sizes. hidden_null and idle_null are
  // L = 2 residual classes from the surplus hidden units: hidden_null (the ±
  // nonzero branches) has count 2·r·(m−r), idle_null (the flat zero line) has
  // count (m−r)·(n+d−2r). Both are empty when m ≤ r.
  function groupModeCounts() {
    const n = appState.outputDim, d = appState.inputDim;
    const m = appState.hiddenDim1;
    const r = Math.min(n, d);
    const surplus = Math.max(m - r, 0);
    return {
      aligned:      r,
      aligned_null: r,
      cross:        2 * r * (r - 1),
      single_value: r * (n - r) + r * (d - r),
      hidden_null:  2 * r * surplus,
      idle_null:    surplus * (Math.max(n - r, 0) + Math.max(d - r, 0))
    };
  }

  function initTheoryControls() {
    if (!showTheoryControls) return;
    const host = el('theoryControlContainer');
    if (!host) {
      console.warn('[train_widget] showTheoryControls: true but no host element ' +
                   `#${prefix}-theoryControlContainer found.`);
      return;
    }
    host.innerHTML = '';

    // Local mutable mirror of the chart's predictionConfig.
    const cfg = {
      theories: new Set(rightChart.predictionConfig.theories),
      strategy: rightChart.predictionConfig.strategy,
      k: rightChart.predictionConfig.k,
      perClassK: { ...rightChart.predictionConfig.perClassK }
    };
    // Seed per-class defaults (k=3, clamped) for any group not already set.
    const caps = groupModeCounts();
    for (const g of THEORY_GROUPS) {
      if (cfg.perClassK[g] === undefined) cfg.perClassK[g] = Math.min(3, caps[g]);
    }

    const wrap = document.createElement('div');
    wrap.style.cssText = 'display:flex; flex-direction:column; gap:10px; font-size:14px;';

    // ── Which theory ──
    const thRow = document.createElement('div');
    thRow.className = 'inline-row';
    const gnCb = mkCheckbox('GN', cfg.theories.has('gn'));
    const fullCb = mkCheckbox('full Hessian', cfg.theories.has('full'));
    thRow.appendChild(labelWrap('Theory:', null, true));
    thRow.appendChild(gnCb.label);
    thRow.appendChild(fullCb.label);
    wrap.appendChild(thRow);

    // ── Strategy ──
    const stRow = document.createElement('div');
    stRow.className = 'inline-row';
    const pooledRadio = mkRadio(`${prefix}-thStrategy`, 'top-k overall', cfg.strategy === 'pooled');
    const perClassRadio = mkRadio(`${prefix}-thStrategy`, 'top-k per class', cfg.strategy === 'perClass');
    stRow.appendChild(labelWrap('Show:', null, true));
    stRow.appendChild(pooledRadio.label);
    stRow.appendChild(perClassRadio.label);
    wrap.appendChild(stRow);

    // ── Pooled-k input ──
    const pooledRow = document.createElement('div');
    pooledRow.className = 'inline-field';
    const pooledK = mkNumber(cfg.k, 0, 9999);
    pooledRow.appendChild(document.createTextNode('top '));
    pooledRow.appendChild(pooledK);
    pooledRow.appendChild(document.createTextNode(' overall (per theory)'));
    wrap.appendChild(pooledRow);

    // ── Per-class rows ──
    const perClassBox = document.createElement('div');
    perClassBox.style.cssText = 'display:flex; flex-direction:column; gap:4px; padding-left:4px;';
    const classInputs = {};
    for (const g of THEORY_GROUPS) {
      const row = document.createElement('div');
      row.className = 'inline-field';
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.checked = (cfg.perClassK[g] || 0) > 0;
      const num = mkNumber(cfg.perClassK[g] || 0, 0, caps[g]);
      const lbl = document.createElement('span');
      lbl.textContent = ` ${THEORY_GROUP_LABELS[g]} `;
      const capNote = document.createElement('span');
      capNote.style.cssText = 'font-size:12px; color:#999;';
      capNote.textContent = `(max ${caps[g]})`;
      row.appendChild(cb); row.appendChild(lbl); row.appendChild(num); row.appendChild(capNote);
      perClassBox.appendChild(row);
      classInputs[g] = { cb, num, row };
    }
    wrap.appendChild(perClassBox);

    // ── L ≠ 2 notice ──
    const notice = document.createElement('div');
    notice.style.cssText = 'font-size:12px; color:#888;';
    wrap.appendChild(notice);

    host.appendChild(wrap);

    // ── Apply + visibility logic ──
    function currentL() {
      return appState.hiddenDims().length + 1;
    }

    function syncStrategyVisibility() {
      const pooled = cfg.strategy === 'pooled';
      pooledRow.style.display = pooled ? 'flex' : 'none';
      perClassBox.style.display = pooled ? 'none' : 'flex';

      // Full Hessian theory is 2-layer-only. When L != 2, force it off and
      // disable the checkbox so the user can't select it.
      const isL2 = currentL() === 2;
      if (!isL2 && cfg.theories.has('full')) cfg.theories.delete('full');
      fullCb.input.checked = cfg.theories.has('full');
      fullCb.input.disabled = !isL2;
      fullCb.label.style.opacity = isL2 ? '1' : '0.45';

      // The residual-only classes (aligned_null, hidden_null, idle_null) are
      // meaningful only when the full theory is selected (hence only at L = 2).
      // Their rows are disabled/greyed otherwise.
      const fullOn = cfg.theories.has('full');
      for (const g of ['aligned_null', 'hidden_null', 'idle_null']) {
        const ci = classInputs[g];
        if (!ci) continue;
        ci.cb.disabled = !fullOn;
        ci.num.disabled = !fullOn;
        ci.row.style.opacity = fullOn ? '1' : '0.45';
      }
    }

    function pushConfig() {
      rightChart.setTheories([...cfg.theories]);
      rightChart.setPredictionStrategy(cfg.strategy);
      rightChart.setPooledK(cfg.k);
      for (const g of THEORY_GROUPS) rightChart.setClassK(g, cfg.perClassK[g] || 0);
      // A single repaint is enough; the setters each repaint but that's cheap
      // and keeps the code simple.
    }

    function updateNotice() {
      const L = currentL();
      notice.textContent = (L !== 2)
        ? `Note: you currently have L = ${L}. The full-Hessian theory is derived ` +
          `for L = 2 (single hidden layer) only, so it is unavailable; the ` +
          `Gauss-Newton theory shown is valid at this depth.`
        : '';
    }

    gnCb.input.addEventListener('change', () => {
      if (gnCb.input.checked) cfg.theories.add('gn'); else cfg.theories.delete('gn');
      syncStrategyVisibility(); pushConfig();
    });
    fullCb.input.addEventListener('change', () => {
      if (fullCb.input.checked) cfg.theories.add('full'); else cfg.theories.delete('full');
      syncStrategyVisibility(); pushConfig();
    });
    pooledRadio.input.addEventListener('change', () => {
      if (pooledRadio.input.checked) { cfg.strategy = 'pooled'; syncStrategyVisibility(); pushConfig(); }
    });
    perClassRadio.input.addEventListener('change', () => {
      if (perClassRadio.input.checked) { cfg.strategy = 'perClass'; syncStrategyVisibility(); pushConfig(); }
    });
    pooledK.addEventListener('change', () => {
      cfg.k = Math.max(0, Math.round(parseFloat(pooledK.value) || 0));
      pooledK.value = cfg.k; pushConfig();
    });
    for (const g of THEORY_GROUPS) {
      const { cb, num } = classInputs[g];
      cb.addEventListener('change', () => {
        // Toggling the checkbox sets k to default-3 (clamped) on, or 0 off.
        cfg.perClassK[g] = cb.checked ? Math.min(3, caps[g]) : 0;
        num.value = cfg.perClassK[g]; pushConfig();
      });
      num.addEventListener('change', () => {
        let v = Math.max(0, Math.round(parseFloat(num.value) || 0));
        v = Math.min(v, caps[g]);
        cfg.perClassK[g] = v; num.value = v; cb.checked = v > 0; pushConfig();
      });
    }

    // Recompute caps + notice when dims or depth change.
    function refreshCaps() {
      const c = groupModeCounts();
      for (const g of THEORY_GROUPS) {
        classInputs[g].num.max = c[g];
        // re-clamp current value
        let v = Math.min(parseFloat(classInputs[g].num.value) || 0, c[g]);
        classInputs[g].num.value = v; cfg.perClassK[g] = v;
        classInputs[g].row.querySelector('span:last-child').textContent = `(max ${c[g]})`;
      }
      updateNotice(); pushConfig();
    }
    el('useSecondLayerCheckbox')?.addEventListener('change', () => { syncStrategyVisibility(); updateNotice(); pushConfig(); });
    el('useThirdLayerCheckbox')?.addEventListener('change', () => { syncStrategyVisibility(); updateNotice(); pushConfig(); });

    syncStrategyVisibility();
    updateNotice();
    pushConfig();

    // Expose a refresh hook so dimension-input handlers can recompute caps.
    rightChart._theoryControlsRefreshCaps = refreshCaps;
  }

  // Small DOM helpers for the theory panel.
  function mkCheckbox(text, checked) {
    const label = document.createElement('label');
    label.className = 'inline-field';
    const input = document.createElement('input');
    input.type = 'checkbox'; input.checked = !!checked;
    const span = document.createElement('span'); span.textContent = ' ' + text;
    label.appendChild(input); label.appendChild(span);
    return { label, input };
  }
  function mkRadio(name, text, checked) {
    const label = document.createElement('label');
    label.className = 'inline-field';
    const input = document.createElement('input');
    input.type = 'radio'; input.name = name; input.checked = !!checked;
    const span = document.createElement('span'); span.textContent = ' ' + text;
    label.appendChild(input); label.appendChild(span);
    return { label, input };
  }
  function mkNumber(value, min, max) {
    const input = document.createElement('input');
    input.type = 'number'; input.value = value;
    if (min !== undefined) input.min = min;
    if (max !== undefined) input.max = max;
    input.style.cssText = 'width:64px; font-size:13px; padding:3px 6px; text-align:center; border:1px solid #ccc; border-radius:3px;';
    return input;
  }
  function labelWrap(text, _unused, bold) {
    const span = document.createElement('span');
    span.textContent = text;
    if (bold) span.style.fontWeight = '600';
    span.style.color = '#555';
    return span;
  }

  // ── Initial render ─────────────────────────────────────────────────────────
  function initialRender() {
    ensureSingularValuesLength(); rebuildM();
    renderDataParams(); initInputModeControls(); initInitModeControls();
    renderDimensionInputs(); renderBasisOptions(); renderSingularValueEditor();
    if (!simple) bindSVPresetButtons();
    initModelControls(); initTrainingControls(); initPlotControls(); initButtons();
    initTheoryControls();
    initRunControls();
    validateDimensions(); validateInputMode();
    wireDetailsToggle('toggleModelDetailsButton',   'modelDetailsPanel',   'show model details',   'hide model details');
    wireDetailsToggle('toggleFunctionDetailsButton','functionDetailsPanel','show function details','hide function details');
  }

  function waitForMathJax(attempts = 0) {
    if (window.MathJax?.typesetPromise && window.MathJax?.startup?.promise) {
      window.MathJax.startup.promise.then(() => initialRender()).catch(err => console.error('initialRender error:', err));
    } else if (attempts < 50) {
      setTimeout(() => waitForMathJax(attempts + 1), 50);
    } else {
      initialRender();
    }
  }

  waitForMathJax();
}
