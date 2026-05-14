// ============================================================================
// TRAIN WIDGET — streamlined deep-linear playground
// ============================================================================
// This is a slimmed-down sibling of app.js. The dataset is fixed (linear
// regression on Y = W*X) so the descriptive panels are gone, and most of the
// model + ground-truth knobs are tucked behind two collapsible panels:
//
//   - "Show model details": hidden dims, second-layer toggle, model seed,
//     random-O toggle (only meaningful when alignedInit is on), and the
//     weight-distribution caption (text adapts to alignedInit mode).
//   - "Show function details": input/output dims, σ editor + preset, then
//     random U/V toggle and basis seed.
//
// The always-visible task controls are: input distribution (whitened /
// Gaussian), training points N, data seed, and the Initialization radios
// (Random Gaussian vs Aligned to SVD basis). Training keeps η + ε.
//
// State is persisted under its own localStorage key so this widget can be
// edited independently of other widgets in the project.

import { AppState } from './state.js';
import { Simulation } from './simulation.js';
import { LossChart, RightChart } from './visualization.js';
import { buildMComponentsFromSpec } from './matrix.js';

// ============================================================================
// STATE & SIMULATION
// ============================================================================
// Custom storage key keeps this widget's state separate from index.html and
// any other widget pages. Each widget gets its own sandbox so users can mess
// with settings independently.

const STORAGE_KEY = 'mlp-trainer-state-train-widget';

const appState = new AppState(STORAGE_KEY);
const hadPersistedState = appState.load();

// Widget-specific default override: η = 0.6 on first ever load (no persisted
// state). Once the user touches a control and the widget calls save(), the
// persisted value wins on subsequent loads, and "reset to defaults" wipes
// localStorage + reloads, which lands back on this override.
if (!hadPersistedState) {
  appState.eta = 0.6;
}

// In-memory target matrix M plus its SVD components U, V. We always pull both
// here (cheap) so the start-button handler can pass U, V into the simulation
// when aligned init is on, without re-running the basis sampler. Rebuilt from
// appState.matrixSpec via rebuildM().
let currentM = null;
let currentU = null;
let currentV = null;

// Snapshot of the model's weights captured immediately after
// Simulation.initialize() builds the MLP. Stays frozen during training so
// the "show init weights" panel always shows the t=0 weights, not whatever
// the model has drifted to. Set by snapshotInitWeights() (below) and read by
// renderInitWeights().
let initWeightSnapshot = null;

function rebuildM() {
  const parts = buildMComponentsFromSpec(appState.matrixSpec, appState.inputDim, appState.outputDim);
  currentM = parts.M;
  currentU = parts.U;
  currentV = parts.V;
}

// ============================================================================
// INIT WEIGHT DISPLAY — frozen t=0 snapshot for the "show init weights" toggle
// ============================================================================
// Captures the model's per-layer weight matrices, the product W_L···W_1, and
// the target W⋆ at the moment Simulation.initialize() runs. The display
// stays static during training so users can verify the chosen init mode
// (aligned vs random) actually does what its label says.

const NUMBER_COL_WIDTH = 9;  // chars per cell, includes leading space

function formatNumber(x) {
  // Compact signed fixed-point: e.g. "+0.0234", "-1.0000", " 0.0000".
  // 4 decimals is enough to spot e.g. ε^L = 1e-6 vs an unexpected 0.5.
  if (!isFinite(x)) return '   NaN  ';
  const sign = x < 0 ? '-' : '+';
  const abs = Math.abs(x);
  const s = sign + abs.toFixed(4);
  // Pad/truncate so each column is the same width.
  return s.length >= NUMBER_COL_WIDTH ? s : s.padStart(NUMBER_COL_WIDTH, ' ');
}

function formatMatrix(M) {
  const rows = M.length;
  if (rows === 0) return '  (empty)';
  const cols = M[0].length;
  const lines = [];
  for (let i = 0; i < rows; i++) {
    let row = ' ';
    for (let j = 0; j < cols; j++) row += formatNumber(M[i][j]) + ' ';
    lines.push(row);
  }
  return lines.join('\n');
}

// Plain-data deep copy of all weight matrices, so later in-place updates by
// the trainer can't mutate the snapshot.
function snapshotWeights(modelW) {
  return modelW.map(layer => layer.map(row => row.slice()));
}

// W_L · ... · W_1, returned as a plain 2D array. Same convention as model.js.
function productMatrix(weightsSnapshot) {
  if (weightsSnapshot.length === 0) return [];
  // Start with W_1, then left-multiply by W_2, W_3, ...
  let P = weightsSnapshot[0].map(row => row.slice());
  for (let l = 1; l < weightsSnapshot.length; l++) {
    const W = weightsSnapshot[l];
    const rows = W.length;
    const inner = W[0].length;
    const outerCols = P[0].length;
    const newP = new Array(rows);
    for (let i = 0; i < rows; i++) {
      const Wi = W[i];
      const newRow = new Array(outerCols).fill(0);
      for (let k = 0; k < inner; k++) {
        const wik = Wi[k];
        const Pk = P[k];
        for (let j = 0; j < outerCols; j++) newRow[j] += wik * Pk[j];
      }
      newP[i] = newRow;
    }
    P = newP;
  }
  return P;
}

// Take a snapshot of the freshly-initialized model's weights and store
// metadata needed to render them. Called from the start handler immediately
// after simulation.captureParams + simulation.start() built the MLP. The
// snapshot is plain data — safe from later in-place mutation by the trainer.
function snapshotInitWeights() {
  if (!simulation.model || !simulation.model.W) {
    initWeightSnapshot = null;
    return;
  }
  initWeightSnapshot = {
    layers: snapshotWeights(simulation.model.W),
    target: currentM ? currentM.map(row => row.slice()) : null,
    layerSizes: simulation.model.layerSizes.slice(),
    initScale: appState.initScale,
    alignedInit: !!appState.alignedInit
  };
  renderInitWeights();
}

function renderInitWeights() {
  const container = document.getElementById('initWeightsContainer');
  if (!container) return;
  if (!initWeightSnapshot) {
    container.innerHTML =
      '<span style="color: #999;">Click <em>start</em> to initialize the model and view its weights here.</span>';
    return;
  }
  const snap = initWeightSnapshot;
  const L = snap.layers.length;
  const lines = [];

  // Header summary so the reader knows what they're looking at.
  const sizesStr = snap.layerSizes.join(' → ');
  const modeStr = snap.alignedInit ? 'Aligned to SVD basis' : 'Random Gaussian (muP)';
  lines.push(`Layer sizes: ${sizesStr}   |   Init mode: ${modeStr}   |   ε = ${snap.initScale}`);
  lines.push('');

  // Per-layer W_ℓ.
  for (let l = 0; l < L; l++) {
    const W = snap.layers[l];
    lines.push(`W_${l + 1}   (${W.length} × ${W[0].length})`);
    lines.push(formatMatrix(W));
    lines.push('');
  }

  // Product matrix W_L · ... · W_1 — the most direct sanity check for
  // aligned-init claims. Under aligned init with randomO=false this should
  // be U · diag(ε^L) · V^⊤, i.e. exactly the same shape as W⋆ but scaled
  // by ε^L.
  const P = productMatrix(snap.layers);
  if (P.length > 0) {
    lines.push(`Product W_L · ... · W_1   (${P.length} × ${P[0].length})`);
    lines.push(formatMatrix(P));
    lines.push('');
  }

  // Target matrix W⋆ for direct comparison with the product.
  if (snap.target) {
    lines.push(`Target W⋆   (${snap.target.length} × ${snap.target[0].length})`);
    lines.push(formatMatrix(snap.target));
  }

  container.textContent = lines.join('\n');
}

const simulation = new Simulation();
const lossChart = new LossChart('lossChart');
const rightChart = new RightChart('rightChart');

simulation.onFrameUpdate = () => {
  const state = simulation.getState();
  const eta = state.eta;
  lossChart.update(state.lossHistory, eta, state.predictedLossHistory);
  rightChart.update(state.eigenvalueHistory, eta, state.predictedEigenvalueHistory);
};

simulation.onDiverge = (iteration, loss) => {
  startPauseButton.textContent = 'start';
  let errorEl = document.getElementById('divergeError');
  if (!errorEl) {
    errorEl = document.createElement('div');
    errorEl.id = 'divergeError';
    errorEl.style.cssText = 'margin-top: 12px; padding: 10px 16px; background: #fff0f0; border: 1px solid #e88; border-radius: 6px; color: #a33; font-size: 14px; text-align: center;';
    const buttonRow = startPauseButton.parentElement;
    buttonRow.parentElement.insertBefore(errorEl, buttonRow.nextSibling);
  }
  const lossStr = isFinite(loss) ? loss.toExponential(2) : 'NaN/Infinity';
  errorEl.textContent = `⚠ Training stopped: loss diverged to ${lossStr} at step ${iteration}. Try a smaller learning rate or smaller singular values.`;
  errorEl.style.display = 'block';
};

// Auto-stop after convergence: simulation paused itself after the cooldown
// elapsed. Just flip the button label so the user can resume by clicking
// start again. (See simulation.js convergenceLossThreshold /
// convergenceCooldownSteps.)
simulation.onAutoStop = (iteration, crossStep, loss) => {
  startPauseButton.textContent = 'start';
  console.log(
    `[auto-stop] Training paused at step ${iteration}. ` +
    `Loss first dropped below ${simulation.convergenceLossThreshold} at step ${crossStep}; ` +
    `${simulation.convergenceCooldownSteps} steps later, current loss = ${loss.toExponential(2)}. ` +
    `Click start to resume.`
  );
};

// ============================================================================
// LOG-SCALE SLIDER MAPPING
// ============================================================================

function logSliderToValue(sliderVal, min, max) {
  const logMin = Math.log10(min);
  const logMax = Math.log10(max);
  return Math.pow(10, logMin + (sliderVal / 100) * (logMax - logMin));
}

function valueToLogSlider(value, min, max) {
  const logMin = Math.log10(min);
  const logMax = Math.log10(max);
  return ((Math.log10(value) - logMin) / (logMax - logMin)) * 100;
}

// ============================================================================
// COLLAPSIBLE DETAILS TOGGLE
// ============================================================================
// Pattern lifted from ntk-app.js — toggles a details panel and updates the
// button text. Panels are transient: closed on every page load (no
// persistence), per the simpler-by-default design goal.

function wireDetailsToggle(buttonId, panelId, showLabel, hideLabel) {
  const btn = document.getElementById(buttonId);
  const panel = document.getElementById(panelId);
  if (!btn || !panel) return;
  btn.addEventListener('click', () => {
    const hidden = panel.style.display === 'none' || panel.style.display === '';
    panel.style.display = hidden ? 'block' : 'none';
    btn.textContent = hidden ? hideLabel : showLabel;
  });
}

// ============================================================================
// DATA-PARAMS EDITOR (nTrain, dataSeed)
// ============================================================================

function renderDataParams() {
  const container = document.getElementById('dataParamsContainer');
  container.innerHTML = '';

  function addField(labelText, key, attrs) {
    const cell = document.createElement('div');
    cell.className = 'inline-field';
    const lbl = document.createElement('span');
    lbl.textContent = labelText;
    const inp = document.createElement('input');
    inp.type = 'number';
    Object.assign(inp, attrs);
    inp.value = appState[key];
    inp.addEventListener('change', () => {
      let v = parseFloat(inp.value);
      if (!isFinite(v)) v = appState[key];
      if (attrs.min !== undefined) v = Math.max(attrs.min, v);
      if (attrs.max !== undefined) v = Math.min(attrs.max, v);
      if (attrs.step === 1) v = Math.round(v);
      appState[key] = v;
      inp.value = v;
      appState.save();
      validateInputMode();
    });
    cell.appendChild(lbl);
    cell.appendChild(inp);
    return cell;
  }

  container.appendChild(addField('Training Points $N$:', 'nTrain', { min: 1, max: 10000, step: 1 }));
  container.appendChild(addField('Data Seed:',           'dataSeed', { min: 0, step: 1 }));

  if (window.MathJax && window.MathJax.typesetPromise) {
    MathJax.typesetPromise([container]).catch(err => console.log(err));
  }
}

// ============================================================================
// INPUT-MODE CONTROLS (whitened vs iid Gaussian)
// ============================================================================

const inputModeWhitened = document.getElementById('inputModeWhitened');
const inputModeGaussian = document.getElementById('inputModeGaussian');
const inputModeWarning = document.getElementById('inputModeWarning');

function initInputModeControls() {
  inputModeWhitened.checked = appState.inputMode === 'whitened';
  inputModeGaussian.checked = appState.inputMode === 'gaussian';

  inputModeWhitened.addEventListener('change', () => {
    if (inputModeWhitened.checked) {
      appState.inputMode = 'whitened';
      appState.save();
      validateInputMode();
    }
  });
  inputModeGaussian.addEventListener('change', () => {
    if (inputModeGaussian.checked) {
      appState.inputMode = 'gaussian';
      appState.save();
      validateInputMode();
    }
  });
}

function validateInputMode() {
  const ok = appState.inputMode !== 'whitened' || appState.nTrain >= appState.inputDim;
  if (!ok) {
    inputModeWarning.style.display = 'block';
    inputModeWarning.innerHTML =
      `⚠ <strong>whitened inputs require N ≥ k.</strong> ` +
      `You have N=${appState.nTrain} training points and k=${appState.inputDim} input dim — ` +
      `increase N or switch to iid Gaussian.`;
  } else {
    inputModeWarning.style.display = 'none';
  }
  return ok;
}

// ============================================================================
// INIT-MODE CONTROLS (Random Gaussian vs Aligned to SVD basis)
// ============================================================================
// alignedInit toggles which path model.js takes when building the weights.
//
// Convention: ε (the user-entered init scale) is the *per-layer* singular
// value in both modes. Under random Gaussian, that's the muP std; under
// aligned, each layer's per-mode singular value equals ε directly. The
// product matrix has typical singular values ε^L in either mode, so flipping
// the toggle at fixed ε preserves the overall init scale and the staircase-
// vs-plateau comparison is meaningful.
//
// In aligned mode a "Random orthogonal matrices O" checkbox becomes visible
// inside model details. When unchecked, every junction rotation is I; when
// checked, each is a fresh random orthogonal seeded by modelSeed.
//
// The actual MLP construction happens in simulation.captureParams /
// initialize; here we only manage UI state and persistence.

const initModeRandomRadio  = document.getElementById('initModeRandom');
const initModeAlignedRadio = document.getElementById('initModeAligned');
const randomORow           = document.getElementById('randomORow');
const randomOCheckbox      = document.getElementById('randomOCheckbox');
const initScaleLabel       = document.getElementById('initScaleLabel');
const weightCaption        = document.getElementById('weightCaption');

function initInitModeControls() {
  // Mirror persisted state into the radios.
  initModeRandomRadio.checked  = !appState.alignedInit;
  initModeAlignedRadio.checked =  appState.alignedInit;
  randomOCheckbox.checked      = !!appState.randomO;

  initModeRandomRadio.addEventListener('change', () => {
    if (initModeRandomRadio.checked) {
      appState.alignedInit = false;
      appState.save();
      updateInitModeDependentUI();
    }
  });
  initModeAlignedRadio.addEventListener('change', () => {
    if (initModeAlignedRadio.checked) {
      appState.alignedInit = true;
      appState.save();
      updateInitModeDependentUI();
    }
  });
  randomOCheckbox.addEventListener('change', () => {
    appState.randomO = randomOCheckbox.checked;
    appState.save();
  });

  // "Show init weights" checkbox toggles visibility of the numerical-weight
  // panel. The actual content is populated/refreshed by snapshotInitWeights()
  // on start, and cleared on reset. We render on toggle too so the panel
  // shows the placeholder text immediately when first opened.
  const showInitWeightsCheckbox = document.getElementById('showInitWeightsCheckbox');
  const initWeightsContainer = document.getElementById('initWeightsContainer');
  showInitWeightsCheckbox.checked = !!appState.showInitWeights;
  initWeightsContainer.style.display = appState.showInitWeights ? 'block' : 'none';
  if (appState.showInitWeights) renderInitWeights();
  showInitWeightsCheckbox.addEventListener('change', () => {
    appState.showInitWeights = showInitWeightsCheckbox.checked;
    initWeightsContainer.style.display = showInitWeightsCheckbox.checked ? 'block' : 'none';
    if (showInitWeightsCheckbox.checked) renderInitWeights();
    appState.save();
  });
}

/**
 * Show/hide the random-O row, swap the ε label and the weight-distribution
 * caption based on alignedInit. Called whenever alignedInit changes, and
 * once at startup. Safe to call before MathJax has finished loading — the
 * typeset call is guarded.
 */
function updateInitModeDependentUI() {
  const aligned = !!appState.alignedInit;

  // Random-O row is only meaningful in aligned mode.
  randomORow.style.display = aligned ? 'flex' : 'none';

  // Swap ε's label and the weight caption.
  if (aligned) {
    initScaleLabel.innerHTML = 'init scale $\\varepsilon$';
    weightCaption.innerHTML =
      'Weights aligned in SVD basis: each layer has singular value $\\varepsilon$ per mode, so the product matrix has singular values $\\varepsilon^L$. (Matches the typical scale of $W_{ij} \\sim \\mathcal{N}(0, \\varepsilon^2/n_{\\ell-1})$.)';
  } else {
    initScaleLabel.innerHTML = 'init scale $\\varepsilon$';
    weightCaption.innerHTML =
      'Weights: $W_{ij}^{(\\ell)} \\sim \\mathcal{N}(0,\\, \\varepsilon^2/n_{\\ell-1})$';
  }

  // Re-typeset the two regions whose math changed.
  if (window.MathJax && window.MathJax.typesetPromise) {
    MathJax.typesetPromise([initScaleLabel, weightCaption]).catch(err => console.log(err));
  }
}

// ============================================================================
// DIMENSION INPUTS (input dim k, output dim m) — now live in model details
// ============================================================================

const DIM_MIN = 1;
const DIM_MAX_INPUT = 200;
const DIM_MAX_OUTPUT = 50;   // capped to keep the σ editor manageable

function renderDimensionInputs() {
  const container = document.getElementById('dimInputs');
  container.innerHTML = '';

  function makeDimInput(key, labelHTML, max) {
    const cell = document.createElement('div');
    cell.className = 'inline-field';
    const label = document.createElement('span');
    label.innerHTML = labelHTML;

    const input = document.createElement('input');
    input.type = 'number';
    input.min = DIM_MIN;
    input.max = max;
    input.step = 1;
    input.value = appState[key];

    function commit() {
      let v = parseInt(input.value, 10);
      if (!isFinite(v)) v = appState[key];
      v = Math.max(DIM_MIN, Math.min(max, v));
      appState[key] = v;
      input.value = v;
      ensureSingularValuesLength();
      rebuildM();
      appState.save();
      renderSingularValueEditor();
      validateDimensions();
      validateInputMode();
    }
    input.addEventListener('change', commit);

    cell.appendChild(label);
    cell.appendChild(input);
    return cell;
  }

  container.appendChild(makeDimInput('inputDim',  'Input Dim <i>k</i>:',  DIM_MAX_INPUT));
  container.appendChild(makeDimInput('outputDim', 'Output Dim <i>m</i>:', DIM_MAX_OUTPUT));
}

// ============================================================================
// BASIS OPTIONS (randomBasis checkbox + basisSeed) — now live in function details
// ============================================================================

function renderBasisOptions() {
  const container = document.getElementById('basisOptions');
  container.innerHTML = '';

  // Random U,V toggle
  const checkboxCell = document.createElement('label');
  checkboxCell.className = 'inline-field';
  const cb = document.createElement('input');
  cb.type = 'checkbox';
  cb.checked = appState.matrixSpec.randomBasis !== false;
  cb.addEventListener('change', () => {
    appState.matrixSpec.randomBasis = cb.checked;
    rebuildM();
    appState.save();
  });
  checkboxCell.appendChild(cb);
  const cbLabel = document.createElement('span');
  cbLabel.textContent = 'Random Orthogonal $U, V$ (otherwise $W^\\star = \\Sigma$)';
  checkboxCell.appendChild(cbLabel);
  container.appendChild(checkboxCell);

  // Basis seed
  const seedCell = document.createElement('div');
  seedCell.className = 'inline-field';
  const seedLabel = document.createElement('span');
  seedLabel.textContent = 'Basis Seed:';
  seedCell.appendChild(seedLabel);
  const seedInput = document.createElement('input');
  seedInput.type = 'number';
  seedInput.min = 0;
  seedInput.step = 1;
  seedInput.value = appState.matrixSpec.basisSeed || 0;
  seedInput.addEventListener('change', () => {
    let v = parseInt(seedInput.value, 10);
    if (!isFinite(v) || v < 0) v = 0;
    appState.matrixSpec.basisSeed = v;
    seedInput.value = v;
    rebuildM();
    appState.save();
  });
  seedCell.appendChild(seedInput);
  container.appendChild(seedCell);

  if (window.MathJax && window.MathJax.typesetPromise) {
    MathJax.typesetPromise([container]).catch(err => console.log(err));
  }
}

// ============================================================================
// SINGULAR-VALUE EDITOR
// ============================================================================

function ensureSingularValuesLength() {
  const spec = appState.matrixSpec;
  const m = appState.outputDim;
  if (!Array.isArray(spec.singularValues)) spec.singularValues = [];
  if (spec.singularValues.length > m) {
    spec.singularValues = spec.singularValues.slice(0, m);
  } else {
    while (spec.singularValues.length < m) {
      // New entries continue the geometric trend, or fall back to 1/(i+1).
      const i = spec.singularValues.length;
      const last = i > 0 ? spec.singularValues[i - 1] : 1.0;
      let next = +(last * 0.7).toFixed(4);
      if (next <= 0) next = 1 / (i + 1);
      spec.singularValues.push(next);
    }
  }
}

function renderSingularValueEditor() {
  const editor = document.getElementById('svEditor');
  editor.innerHTML = '';

  const spec = appState.matrixSpec;
  const m = appState.outputDim;

  for (let i = 0; i < m; i++) {
    const cell = document.createElement('div');
    cell.className = 'sv-row';

    const label = document.createElement('div');
    label.className = 'sv-label';
    label.innerHTML = `&sigma;<sub>${i + 1}</sub>:`;
    cell.appendChild(label);

    const input = document.createElement('input');
    input.type = 'number';
    input.step = 'any';
    input.value = spec.singularValues[i];

    input.addEventListener('input', () => {
      const raw = input.value.trim();
      if (raw === '') return;
      const v = parseFloat(raw);
      if (!isFinite(v)) return;
      spec.singularValues[i] = v < 0 ? 0 : v;
      rebuildM();
      appState.save();
    });

    cell.appendChild(input);
    editor.appendChild(cell);
  }
}

// Only powers-of-2 remains. All other presets have been removed from the UI.
function applySVPreset(name) {
  if (name !== 'powers-of-2') return;
  const spec = appState.matrixSpec;
  const m = appState.outputDim;
  const svs = new Array(m);
  for (let i = 0; i < m; i++) svs[i] = +(Math.pow(0.5, i)).toFixed(4);
  spec.singularValues = svs;
  rebuildM();
  appState.save();
  renderSingularValueEditor();
}

function bindSVPresetButtons() {
  document.querySelectorAll('[data-sv-preset]').forEach(btn => {
    btn.addEventListener('click', () => applySVPreset(btn.dataset.svPreset));
  });
}

function validateDimensions() {
  const warning = document.getElementById('svWarning');
  if (appState.inputDim < appState.outputDim) {
    warning.style.display = 'block';
    warning.innerHTML = `⚠ <strong>input dim (${appState.inputDim}) &lt; output dim (${appState.outputDim}).</strong> ` +
      `The matrix can have at most ${appState.inputDim} non-zero singular values; the last ${appState.outputDim - appState.inputDim} ` +
      `&sigma; entries will be ignored.`;
  } else {
    warning.style.display = 'none';
  }
}

// ============================================================================
// MODEL CONTROLS
// ============================================================================

const hiddenDim1Slider = document.getElementById('hiddenDim1Slider');
const hiddenDim1Value = document.getElementById('hiddenDim1Value');
const useSecondLayerCheckbox = document.getElementById('useSecondLayerCheckbox');
const hiddenDim2Row = document.getElementById('hiddenDim2Row');
const hiddenDim2Slider = document.getElementById('hiddenDim2Slider');
const hiddenDim2Value = document.getElementById('hiddenDim2Value');
const modelSeedInput = document.getElementById('modelSeedInput');

const HIDDEN_DIM_RANGE = { min: 1, max: 200 };

function initModelControls() {
  hiddenDim1Slider.min = HIDDEN_DIM_RANGE.min;
  hiddenDim1Slider.max = HIDDEN_DIM_RANGE.max;
  appState.hiddenDim1 = Math.max(HIDDEN_DIM_RANGE.min, Math.min(HIDDEN_DIM_RANGE.max, appState.hiddenDim1));
  hiddenDim1Slider.value = appState.hiddenDim1;
  hiddenDim1Value.textContent = appState.hiddenDim1;

  useSecondLayerCheckbox.checked = appState.useSecondLayer;
  hiddenDim2Row.style.display = appState.useSecondLayer ? 'flex' : 'none';

  hiddenDim2Slider.min = HIDDEN_DIM_RANGE.min;
  hiddenDim2Slider.max = HIDDEN_DIM_RANGE.max;
  hiddenDim2Slider.value = appState.hiddenDim2;
  hiddenDim2Value.textContent = appState.hiddenDim2;

  modelSeedInput.value = appState.modelSeed;
}

modelSeedInput.addEventListener('change', () => {
  appState.modelSeed = parseInt(modelSeedInput.value) || 0;
  modelSeedInput.value = appState.modelSeed;
  appState.save();
});

hiddenDim1Slider.addEventListener('input', () => {
  appState.hiddenDim1 = parseInt(hiddenDim1Slider.value);
  hiddenDim1Value.textContent = appState.hiddenDim1;
  appState.save();
});

useSecondLayerCheckbox.addEventListener('change', () => {
  appState.useSecondLayer = useSecondLayerCheckbox.checked;
  hiddenDim2Row.style.display = appState.useSecondLayer ? 'flex' : 'none';
  appState.save();
});

hiddenDim2Slider.addEventListener('input', () => {
  appState.hiddenDim2 = parseInt(hiddenDim2Slider.value);
  hiddenDim2Value.textContent = appState.hiddenDim2;
  appState.save();
});

// ============================================================================
// TRAINING CONTROLS
// ============================================================================
// η lives in the TRAINING panel; init scale ε has moved to model details.
// Both are still managed here in one place.

const etaSlider = document.getElementById('etaSlider');
const etaValue = document.getElementById('etaValue');
const initScaleSlider = document.getElementById('initScaleSlider');
const initScaleNumber = document.getElementById('initScaleNumber');

const INIT_SCALE_SLIDER_MIN = 0.0001;
const INIT_SCALE_SLIDER_MAX = 1;

function setInitScaleUI(value) {
  initScaleNumber.value = parseFloat(value.toPrecision(6));
  const clamped = Math.max(INIT_SCALE_SLIDER_MIN, Math.min(INIT_SCALE_SLIDER_MAX, value));
  initScaleSlider.value = valueToLogSlider(clamped, INIT_SCALE_SLIDER_MIN, INIT_SCALE_SLIDER_MAX);
}

function initTrainingControls() {
  etaSlider.value = valueToLogSlider(appState.eta, 0.01, 10);
  etaValue.textContent = parseFloat(appState.eta.toPrecision(4));

  setInitScaleUI(appState.initScale);
}

etaSlider.addEventListener('input', () => {
  appState.eta = parseFloat(logSliderToValue(parseFloat(etaSlider.value), 0.01, 10).toPrecision(4));
  etaValue.textContent = appState.eta;
  appState.save();
});

initScaleSlider.addEventListener('input', () => {
  const v = parseFloat(logSliderToValue(parseFloat(initScaleSlider.value), INIT_SCALE_SLIDER_MIN, INIT_SCALE_SLIDER_MAX).toPrecision(4));
  appState.initScale = v;
  initScaleNumber.value = parseFloat(v.toPrecision(6));
  appState.save();
});

initScaleNumber.addEventListener('change', () => {
  let v = parseFloat(initScaleNumber.value);
  if (!isFinite(v) || v <= 0) {
    setInitScaleUI(appState.initScale);
    return;
  }
  appState.initScale = v;
  setInitScaleUI(v);
  appState.save();
});

// ============================================================================
// START / PAUSE / RESET
// ============================================================================

const startPauseButton = document.getElementById('startPauseButton');
startPauseButton.addEventListener('click', () => {
  if (!simulation.isRunning) {
    if (simulation.model) {
      // Already initialized — just resume.
      simulation.start();
      startPauseButton.textContent = 'pause';
      return;
    }

    const errorEl = document.getElementById('divergeError');
    if (errorEl) errorEl.style.display = 'none';

    if (!validateInputMode()) {
      return;
    }

    if (!currentM) rebuildM();

    const hiddenDims = appState.hiddenDims();

    simulation.captureParams({
      M: currentM,
      dataSeed: appState.dataSeed,
      nTrain: appState.nTrain,
      initScale: appState.initScale,
      hiddenDims,
      eta: appState.eta,
      modelSeed: appState.modelSeed,
      inputMode: appState.inputMode,
      // Target singular values feed the Saxe theory predictor in
      // simulation.js. Same array the σ editor maintains — Simulation
      // truncates/pads internally to min(inputDim, outputDim).
      sigmaStar:   appState.matrixSpec.singularValues,
      // Aligned-init params. U, V come from rebuildM(); see top of file.
      alignedInit: appState.alignedInit,
      randomO:     appState.randomO,
      U:           currentU,
      V:           currentV
    });

    // Seed the loss-chart EMA baseline. See app.js for the derivation;
    // logic is identical here.
    const r = Math.min(appState.inputDim, appState.outputDim);
    let frobSq = 0;
    for (let i = 0; i < r; i++) {
      const s = appState.matrixSpec.singularValues[i] || 0;
      frobSq += s * s;
    }
    lossChart.setInitialLoss(0.5 * frobSq);

    simulation.start();
    // simulation.start() invoked initialize() (because !simulation.model on the
    // path above), which built a fresh MLP. Snapshot its t=0 weights now so
    // the "show init weights" panel can render them without races against
    // training updates.
    snapshotInitWeights();
    startPauseButton.textContent = 'pause';
  } else {
    simulation.pause();
    startPauseButton.textContent = 'start';
  }
});

const resetButton = document.getElementById('resetButton');
resetButton.addEventListener('click', () => {
  simulation.reset();
  lossChart.clear();
  rightChart.clear();
  startPauseButton.textContent = 'start';
  const errorEl = document.getElementById('divergeError');
  if (errorEl) errorEl.style.display = 'none';
  // The init-weight snapshot is tied to a specific MLP build. Reset destroys
  // that model; clear the snapshot so the panel shows the placeholder text
  // instead of stale weights from before the reset.
  initWeightSnapshot = null;
  renderInitWeights();
});

const resetToDefaultsButton = document.getElementById('resetToDefaultsButton');
resetToDefaultsButton.addEventListener('click', () => {
  try { localStorage.removeItem(STORAGE_KEY); } catch (e) {}
  location.reload();
});

// ============================================================================
// PLOT CONTROLS
// ============================================================================

const logScaleCheckbox = document.getElementById('logScaleCheckbox');
logScaleCheckbox.checked = appState.logScale;
logScaleCheckbox.addEventListener('change', () => {
  appState.logScale = logScaleCheckbox.checked;
  lossChart.setLogScale(appState.logScale);
  rightChart.setLogScale(appState.logScale);
  appState.save();
});

const logScaleXCheckbox = document.getElementById('logScaleXCheckbox');
logScaleXCheckbox.checked = appState.logScaleX;
logScaleXCheckbox.addEventListener('change', () => {
  appState.logScaleX = logScaleXCheckbox.checked;
  lossChart.setLogScaleX(appState.logScaleX);
  rightChart.setLogScaleX(appState.logScaleX);
  appState.save();
});

const clipSharpnessCheckbox = document.getElementById('clipSharpnessCheckbox');
clipSharpnessCheckbox.addEventListener('change', () => {
  rightChart.setClipSharpness(clipSharpnessCheckbox.checked);
  const state = simulation.getState();
  if (state.eigenvalueHistory.length > 0) {
    rightChart.update(state.eigenvalueHistory, state.eta, state.predictedEigenvalueHistory);
  }
});

// "clip to EoS" checkbox — when on, the sharpness y-axis is forced to include
// the 2/η threshold (current behavior). When off, the axis auto-scales based
// purely on the observed eigenvalues, which is what you want when η is small
// enough that 2/η sits far above the curves and would compress them into the
// bottom of the plot. Persisted via appState.clipToEos.
const clipToEosCheckbox = document.getElementById('clipToEosCheckbox');
clipToEosCheckbox.checked = appState.clipToEos;
rightChart.setClipToEos(appState.clipToEos);
clipToEosCheckbox.addEventListener('change', () => {
  appState.clipToEos = clipToEosCheckbox.checked;
  rightChart.setClipToEos(appState.clipToEos);
  const state = simulation.getState();
  if (state.eigenvalueHistory.length > 0) {
    rightChart.update(state.eigenvalueHistory, state.eta, state.predictedEigenvalueHistory);
  }
  appState.save();
});

// "show theory prediction" checkbox — overlays the Saxe analytic prediction
// (dotted black curves) on BOTH the sharpness plot and the loss plot. On by
// default; persisted to AppState so it survives reload. One checkbox drives
// both overlays because they're two views of the same underlying theory,
// and toggling them together keeps the visual comparison consistent.
const showPredictionCheckbox = document.getElementById('showPredictionCheckbox');
showPredictionCheckbox.checked = appState.showPrediction;
rightChart.setShowPrediction(appState.showPrediction);
lossChart.setShowPrediction(appState.showPrediction);
showPredictionCheckbox.addEventListener('change', () => {
  appState.showPrediction = showPredictionCheckbox.checked;
  rightChart.setShowPrediction(appState.showPrediction);
  lossChart.setShowPrediction(appState.showPrediction);
  appState.save();
});

function setXAxisMode(mode) {
  appState.xAxisMode = mode;
  const useEff = mode === 'teff';
  lossChart.setEffectiveTime(useEff, appState.eta);
  rightChart.setEffectiveTime(useEff, appState.eta);

  document.getElementById('step-link').classList.toggle('active', mode === 'step');
  document.getElementById('teff-link').classList.toggle('active', mode === 'teff');

  const state = simulation.getState();
  if (state.lossHistory.length > 0) {
    lossChart.update(state.lossHistory, appState.eta, state.predictedLossHistory);
    rightChart.update(state.eigenvalueHistory, appState.eta, state.predictedEigenvalueHistory);
  }
  appState.save();
}

document.getElementById('step-link').addEventListener('click', (e) => {
  e.preventDefault();
  setXAxisMode('step');
});
document.getElementById('teff-link').addEventListener('click', (e) => {
  e.preventDefault();
  setXAxisMode('teff');
});

const emaSlider = document.getElementById('emaSlider');
const emaValue = document.getElementById('emaValue');

function emaSliderToWindow(val) {
  if (val === 0) return 1;
  return Math.round(Math.pow(10, (val / 100) * 4));
}

function emaWindowToSlider(window) {
  if (window <= 1) return 0;
  return (Math.log10(window) / 4) * 100;
}

emaSlider.value = emaWindowToSlider(appState.emaWindow);
emaValue.textContent = appState.emaWindow <= 1 ? 'off' : appState.emaWindow;

emaSlider.addEventListener('input', () => {
  const window = emaSliderToWindow(parseInt(emaSlider.value));
  appState.emaWindow = window;
  emaValue.textContent = window <= 1 ? 'off' : window;
  lossChart.setEmaWindow(window);
  const state = simulation.getState();
  if (state.lossHistory.length > 0) {
    lossChart.update(state.lossHistory, appState.eta, state.predictedLossHistory);
  }
  appState.save();
});

// ============================================================================
// INITIAL RENDER
// ============================================================================

function initialRender() {
  ensureSingularValuesLength();
  rebuildM();

  renderDataParams();
  initInputModeControls();
  initInitModeControls();
  renderDimensionInputs();
  renderBasisOptions();
  renderSingularValueEditor();
  bindSVPresetButtons();
  initModelControls();
  initTrainingControls();
  validateDimensions();
  validateInputMode();
  updateInitModeDependentUI();

  // Collapsible details panels — closed on every load. Toggle button text
  // mirrors the show/hide state.
  wireDetailsToggle(
    'toggleModelDetailsButton', 'modelDetailsPanel',
    'show model details', 'hide model details'
  );
  wireDetailsToggle(
    'toggleFunctionDetailsButton', 'functionDetailsPanel',
    'show function details', 'hide function details'
  );

  if (appState.logScale) {
    lossChart.setLogScale(true);
    rightChart.setLogScale(true);
  }
  if (appState.logScaleX) {
    lossChart.setLogScaleX(true);
    rightChart.setLogScaleX(true);
  }
  if (appState.xAxisMode === 'teff') {
    lossChart.setEffectiveTime(true, appState.eta);
    rightChart.setEffectiveTime(true, appState.eta);
    document.getElementById('step-link').classList.remove('active');
    document.getElementById('teff-link').classList.add('active');
  }
}

function waitForMathJax(attempts = 0) {
  if (window.MathJax && window.MathJax.typesetPromise && window.MathJax.startup && window.MathJax.startup.promise) {
    window.MathJax.startup.promise.then(() => {
      initialRender();
    }).catch(err => console.error('initialRender error:', err));
  } else if (attempts < 50) {
    setTimeout(() => waitForMathJax(attempts + 1), 50);
  } else {
    initialRender();
  }
}

waitForMathJax();
