// ============================================================================
// TRAIN WIDGET — streamlined deep-linear playground
// ============================================================================
// Exported as initWidget(prefix, options) so the same file drives:
//
//   • train_widget.html  — standalone page, calls initWidget('', {})
//   • index.html widget1 — calls initWidget('w1', { showLayers: false, simple: true })
//   • index.html widget2 — calls initWidget('w2', { showLayers: true,  simple: true })
//
// `prefix` is prepended to every element ID with a '-' separator, e.g.
// prefix 'w1' turns 'lossChart' into 'w1-lossChart'. An empty prefix
// leaves IDs unchanged (standalone page behaviour).
//
// `options`:
//   storageKey   — localStorage key (defaults to prefix-scoped key)
//   showLayers   — show second/third layer toggles (default true)
//   simple       — when true: hides EMA slider, logscale checkboxes,
//                  step/teff toggle, init-weights panel, and preset button;
//                  locks theory prediction + clip-sharpness on; resets to
//                  defaults in-place instead of reloading the page.
//                  (default false)

import { AppState } from './state.js';
import { Simulation } from './simulation.js';
import { LossChart, RightChart } from './visualization.js';
import { buildMComponentsFromSpec } from './matrix.js';

export function initWidget(prefix = '', options = {}) {
  const el = id => document.getElementById(prefix ? `${prefix}-${id}` : id);

  const showLayers = options.showLayers !== false;
  const simple     = options.simple === true;

  const STORAGE_KEY = options.storageKey ||
    (prefix ? `mlp-trainer-state-${prefix}` : 'mlp-trainer-state-train-widget');

  const appState = new AppState(STORAGE_KEY);
  const hadPersistedState = appState.load();

  if (!hadPersistedState) appState.eta = 0.6;
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
  const simulation = new Simulation({ stepsPerSecId: prefix ? `${prefix}-stepsPerSec` : 'stepsPerSec' });

  const lossChart  = new LossChart(prefix ? `${prefix}-lossChart`  : 'lossChart',  { showEma: !simple });
  const rightChart = new RightChart(prefix ? `${prefix}-rightChart` : 'rightChart', {
    clipSharpness:  true,
    clipToEos:      simple ? true : appState.clipToEos,
    showPrediction: simple ? true : appState.showPrediction
  });

  simulation.onFrameUpdate = () => {
    const state = simulation.getState();
    lossChart.update(state.lossHistory, state.eta, state.predictedLossHistory);
    rightChart.update(state.eigenvalueHistory, state.eta, state.predictedEigenvalueHistory);
  };

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
          U: currentU, V: currentV
        });
        const r = Math.min(appState.inputDim, appState.outputDim);
        let frobSq = 0;
        for (let i = 0; i < r; i++) { const s = appState.matrixSpec.singularValues[i] || 0; frobSq += s * s; }
        lossChart.setInitialLoss(0.5 * frobSq);
        simulation.start();
        if (!simple) snapshotInitWeights();
        startPauseButton.textContent = 'pause';
      } else {
        simulation.pause(); startPauseButton.textContent = 'start';
      }
    });

    el('resetButton').addEventListener('click', () => {
      simulation.reset(); lossChart.clear(); rightChart.clear();
      startPauseButton.textContent = 'start';
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
      if (state.eigenvalueHistory.length > 0) rightChart.update(state.eigenvalueHistory, state.eta, state.predictedEigenvalueHistory);
    });

    clipToEosCb.checked = appState.clipToEos;
    rightChart.setClipToEos(appState.clipToEos);
    clipToEosCb.addEventListener('change', () => {
      appState.clipToEos = clipToEosCb.checked;
      rightChart.setClipToEos(appState.clipToEos);
      const state = simulation.getState();
      if (state.eigenvalueHistory.length > 0) rightChart.update(state.eigenvalueHistory, state.eta, state.predictedEigenvalueHistory);
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
        rightChart.update(state.eigenvalueHistory, appState.eta, state.predictedEigenvalueHistory);
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

  // ── Initial render ─────────────────────────────────────────────────────────
  function initialRender() {
    ensureSingularValuesLength(); rebuildM();
    renderDataParams(); initInputModeControls(); initInitModeControls();
    renderDimensionInputs(); renderBasisOptions(); renderSingularValueEditor();
    if (!simple) bindSVPresetButtons();
    initModelControls(); initTrainingControls(); initPlotControls(); initButtons();
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
