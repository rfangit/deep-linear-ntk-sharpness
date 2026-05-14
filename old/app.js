// ============================================================================
// DEEP LINEAR NETWORK PAGE - Main Application
// ============================================================================
// Wires up:
//   - Network controls (hidden dims, model seed)
//   - Target-matrix editor: dimensions, σ values, basis options (randomBasis +
//     basisSeed). The matrix M is rebuilt from the matrixSpec on every change
//     and stored on AppState. Anything downstream of this page consumes M
//     directly — see data-generator.js.
//   - Data sampling controls (nTrain, dataSeed, inputMode)
//   - Training controls (η, init scale). Batch size is always full-batch
//     (= nTrain) and is not user-controllable.
//   - Plot controls (log scale, η·step toggle, EMA, sharpness clip)

import { AppState } from './state.js';
import { Simulation } from './simulation.js';
import { LossChart, RightChart } from './visualization.js';
import { buildMFromSpec } from './matrix.js';

// ============================================================================
// STATE & SIMULATION
// ============================================================================

const appState = new AppState();
appState.load();

// In-memory target matrix M. Rebuilt from appState.matrixSpec via rebuildM().
// Kept off AppState's persisted JSON because it would be wasteful for large
// dimensions and is fully determined by (matrixSpec, inputDim, outputDim).
let currentM = null;

function rebuildM() {
  currentM = buildMFromSpec(appState.matrixSpec, appState.inputDim, appState.outputDim);
}

const simulation = new Simulation();
const lossChart = new LossChart('lossChart');
const rightChart = new RightChart('rightChart');

simulation.onFrameUpdate = () => {
  const state = simulation.getState();
  const eta = state.eta;
  lossChart.update(state.lossHistory, eta);
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

// ============================================================================
// LOG-SCALE SLIDER MAPPING
// ============================================================================
// η and the init-scale slider are 0-100 range inputs mapped exponentially.
// Update both directions when changing the range.

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
// TASK FORMULA
// ============================================================================

function renderTaskFormula() {
  const formulaContainer = document.getElementById('taskFormulaContainer');
  formulaContainer.innerHTML = '';
  const formulaLabel = document.createElement('div');
  formulaLabel.style.cssText = 'font-size: 15px; margin-bottom: 4px; color: #666;';
  formulaLabel.textContent = 'target function';
  formulaContainer.appendChild(formulaLabel);

  const formulaDiv = document.createElement('div');
  formulaDiv.innerHTML = '$$\\mathbf{y} = W^\\star \\mathbf{x},\\quad W^\\star = U\\,\\mathrm{diag}(\\sigma_1,\\ldots,\\sigma_m)\\,V^\\top$$' +
    '<div style="font-size: 12px; color: #999; margin-top: 4px;">$U, V$ random orthogonal (or identity, when basis is disabled)</div>';
  formulaContainer.appendChild(formulaDiv);

  if (window.MathJax && window.MathJax.typesetPromise) {
    MathJax.typesetPromise([formulaContainer]).catch(err => console.log(err));
  }
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
      // nTrain may have changed; whitened mode needs N ≥ k. Cheap to call
      // unconditionally — it's a single comparison plus DOM toggle.
      validateInputMode();
    });
    cell.appendChild(lbl);
    cell.appendChild(inp);
    return cell;
  }

  container.appendChild(addField('training points $N$:', 'nTrain', { min: 1, max: 10000, step: 1 }));
  container.appendChild(addField('data seed:',           'dataSeed', { min: 0, step: 1 }));

  if (window.MathJax && window.MathJax.typesetPromise) {
    MathJax.typesetPromise([container]).catch(err => console.log(err));
  }
}

// ============================================================================
// INPUT-MODE CONTROLS (whitened vs iid Gaussian)
// ============================================================================
// Whitened mode requires nTrain ≥ inputDim — k orthonormal vectors in fewer
// than k dimensions don't exist. validateInputMode() surfaces this inline
// and the simulation start handler refuses to run when it fails.

const inputModeWhitened = document.getElementById('inputModeWhitened');
const inputModeGaussian = document.getElementById('inputModeGaussian');
const inputModeWarning = document.getElementById('inputModeWarning');

function initInputModeControls() {
  // Reflect persisted state into the radios
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

/**
 * Surface a warning when whitened mode is selected with nTrain < inputDim.
 * Returns true when the current configuration is runnable, false otherwise.
 */
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
// DIMENSION INPUTS (input dim k, output dim m)
// ============================================================================
// These get their own row at the top of the σ editor section because they
// directly drive how many σ values the user has to specify.

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
      renderNetworkViz();
      validateDimensions();
      validateInputMode();
    }
    input.addEventListener('change', commit);

    cell.appendChild(label);
    cell.appendChild(input);
    return cell;
  }

  container.appendChild(makeDimInput('inputDim',  'input dim <i>k</i>:',  DIM_MAX_INPUT));
  container.appendChild(makeDimInput('outputDim', 'output dim <i>m</i>:', DIM_MAX_OUTPUT));
}

// ============================================================================
// BASIS OPTIONS (randomBasis checkbox + basisSeed)
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
  cbLabel.textContent = 'random orthogonal $U, V$ (otherwise $W^\\star = \\Sigma$)';
  checkboxCell.appendChild(cbLabel);
  container.appendChild(checkboxCell);

  // Basis seed
  const seedCell = document.createElement('div');
  seedCell.className = 'inline-field';
  const seedLabel = document.createElement('span');
  seedLabel.textContent = 'basis seed:';
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
    label.innerHTML = `&sigma;<sub>${i + 1}</sub>`;
    cell.appendChild(label);

    const input = document.createElement('input');
    input.type = 'number';
    input.step = 'any';
    input.value = spec.singularValues[i];

    // Update on `input` (every keystroke) so the value is always current,
    // but don't sanitize/clobber what the user typed — they may be mid-edit.
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

function applySVPreset(name) {
  const spec = appState.matrixSpec;
  const m = appState.outputDim;
  const svs = new Array(m);
  switch (name) {
    case 'powers-of-2':
      for (let i = 0; i < m; i++) svs[i] = +(Math.pow(0.5, i)).toFixed(4);
      break;
    case 'linear-decay':
      for (let i = 0; i < m; i++) svs[i] = +((m - i) / m).toFixed(4);
      break;
    case 'all-equal':
      for (let i = 0; i < m; i++) svs[i] = 1.0;
      break;
    case 'one-large':
      for (let i = 0; i < m; i++) svs[i] = i === 0 ? 1.0 : 0.05;
      break;
    default:
      return;
  }
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

// Warn the user when inputDim < outputDim — the matrix can have at most
// min(k,m) non-zero singular values, so any extra σ entries are ignored.
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
  renderNetworkViz();
});

useSecondLayerCheckbox.addEventListener('change', () => {
  appState.useSecondLayer = useSecondLayerCheckbox.checked;
  hiddenDim2Row.style.display = appState.useSecondLayer ? 'flex' : 'none';
  appState.save();
  renderNetworkViz();
});

hiddenDim2Slider.addEventListener('input', () => {
  appState.hiddenDim2 = parseInt(hiddenDim2Slider.value);
  hiddenDim2Value.textContent = appState.hiddenDim2;
  appState.save();
  renderNetworkViz();
});

// ============================================================================
// TRAINING CONTROLS
// ============================================================================
// η is a log-scale slider only.
// init scale ε is a linked pair: a log-scale slider for quick scrubbing
// (range 0.0001 to 1) AND a number input for typing arbitrary values
// (any positive value is accepted; if the typed value falls outside the
// slider range, the slider visually clamps to its nearest end but
// appState.initScale and the number field hold the true value).
// Batch size is not user-controllable — it is always set to nTrain
// (full-batch) at simulation start.

const etaSlider = document.getElementById('etaSlider');
const etaValue = document.getElementById('etaValue');
const initScaleSlider = document.getElementById('initScaleSlider');
const initScaleNumber = document.getElementById('initScaleNumber');

const INIT_SCALE_SLIDER_MIN = 0.0001;
const INIT_SCALE_SLIDER_MAX = 1;

function setInitScaleUI(value) {
  // Number field shows the true value with reasonable precision.
  initScaleNumber.value = parseFloat(value.toPrecision(6));
  // Slider clamps to its range; the user can still see / adjust the true
  // value in the number field.
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

// 'change' (not 'input') so we don't fight the user mid-typing
initScaleNumber.addEventListener('change', () => {
  let v = parseFloat(initScaleNumber.value);
  if (!isFinite(v) || v <= 0) {
    // Reject invalid input — restore previous value
    setInitScaleUI(appState.initScale);
    return;
  }
  appState.initScale = v;
  setInitScaleUI(v);
  appState.save();
});

// ============================================================================
// NETWORK VISUALIZATION
// ============================================================================

function renderNetworkViz() {
  const svg = document.getElementById('networkViz');
  svg.innerHTML = '';

  const inputDim = appState.inputDim;
  const outputDim = appState.outputDim;
  const h1 = appState.hiddenDim1;
  const h2 = appState.useSecondLayer ? appState.hiddenDim2 : null;

  const dims = h2 ? [inputDim, h1, h2, outputDim] : [inputDim, h1, outputDim];
  const numLayers = dims.length;

  const WIDTH = 400;
  const PADDING = 10;
  const GAP = 6;
  const BASE_HEIGHT = 20;
  const LABEL_SPACE = 50;

  const calcHeight = (dim) => BASE_HEIGHT * (Math.log2(Math.max(dim, 1)) + 1);
  const heights = dims.map(calcHeight);
  const maxHeight = Math.max(...heights);
  const height = maxHeight + 2 * PADDING + LABEL_SPACE;

  svg.setAttribute('height', height);
  svg.setAttribute('width', WIDTH);

  const spacing = (WIDTH - 2 * PADDING) / (numLayers + 1);
  const xPositions = [];
  for (let i = 1; i <= numLayers; i++) xPositions.push(PADDING + spacing * i);

  const colors = ['#dddddd', '#bbbbbb', '#dddddd'];
  for (let i = 0; i < numLayers - 1; i++) {
    const x1 = xPositions[i] + GAP;
    const x2 = xPositions[i + 1] - GAP;
    const ha = heights[i];
    const hb = heights[i + 1];
    const centerY = height / 2;
    const points = [
      [x1, centerY - ha / 2], [x2, centerY - hb / 2],
      [x2, centerY + hb / 2], [x1, centerY + ha / 2]
    ];
    const poly = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
    poly.setAttribute('points', points.map(p => p.join(',')).join(' '));
    poly.setAttribute('fill', colors[i % colors.length]);
    poly.setAttribute('opacity', '0.5');
    svg.appendChild(poly);
  }

  for (let i = 0; i < numLayers; i++) {
    const x = xPositions[i];
    const h = heights[i];
    const centerY = height / 2;
    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    line.setAttribute('x1', x);
    line.setAttribute('y1', centerY - h / 2);
    line.setAttribute('x2', x);
    line.setAttribute('y2', centerY + h / 2);
    line.setAttribute('stroke', '#333');
    line.setAttribute('stroke-width', '3');
    svg.appendChild(line);
  }

  for (let i = 0; i < numLayers; i++) {
    const x = xPositions[i];
    const h = heights[i];
    const centerY = height / 2;
    const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    text.setAttribute('x', x);
    text.setAttribute('y', centerY + h / 2 + 18);
    text.setAttribute('text-anchor', 'middle');
    text.setAttribute('font-size', '13');
    text.setAttribute('fill', '#666');
    text.textContent = dims[i];
    svg.appendChild(text);
  }

  const weightLabels = h2 ? ['W₁', 'W₂', 'W₃'] : ['W₁', 'W₂'];
  for (let i = 0; i < numLayers - 1; i++) {
    const cx = (xPositions[i] + xPositions[i + 1]) / 2;
    const centerY = height / 2;
    const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    text.setAttribute('x', cx);
    text.setAttribute('y', centerY + 5);
    text.setAttribute('text-anchor', 'middle');
    text.setAttribute('font-size', '15');
    text.setAttribute('fill', '#333');
    text.textContent = weightLabels[i];
    svg.appendChild(text);
  }
}

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
      // Warning is already visible inline; don't start the simulation.
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
      sigmaStar: appState.matrixSpec.singularValues
    });

    // Initial loss baseline at zero output: ½ · (1/N) Σᵢ ||yᵢ||².
    // For a freshly initialized network with init scale ε ≪ 1, ŷ ≈ 0, so
    // L ≈ (½/N) Σᵢ ||yᵢ||² ≈ ½·E[||y||²] = ½·||M||_F² = ½ Σₗ σₗ².
    // (Σᵢ σᵢ² is a tight expectation for the random-basis case; for the
    // identity-basis case it's exact for E[‖y‖²] given x ~ N(0, I).)
    // The live loss snaps to truth on the first step regardless — this only
    // seeds the EMA so the smoothed curve doesn't ramp from zero.
    const r = Math.min(appState.inputDim, appState.outputDim);
    let frobSq = 0;
    for (let i = 0; i < r; i++) {
      const s = appState.matrixSpec.singularValues[i] || 0;
      frobSq += s * s;
    }
    lossChart.setInitialLoss(0.5 * frobSq);

    simulation.start();
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
});

const resetToDefaultsButton = document.getElementById('resetToDefaultsButton');
resetToDefaultsButton.addEventListener('click', () => {
  try { localStorage.removeItem('mlp-trainer-state-deep-linear'); } catch (e) {}
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

// "show theory prediction" checkbox — overlays the Saxe analytic prediction
// (dotted black curves) on the sharpness plot. On by default; persisted to
// AppState so it survives reload.
const showPredictionCheckbox = document.getElementById('showPredictionCheckbox');
showPredictionCheckbox.checked = appState.showPrediction;
rightChart.setShowPrediction(appState.showPrediction);
showPredictionCheckbox.addEventListener('change', () => {
  appState.showPrediction = showPredictionCheckbox.checked;
  rightChart.setShowPrediction(appState.showPrediction);
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
    lossChart.update(state.lossHistory, appState.eta);
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
    lossChart.update(state.lossHistory, appState.eta);
  }
  appState.save();
});

// ============================================================================
// INITIAL RENDER
// ============================================================================

function initialRender() {
  ensureSingularValuesLength();
  rebuildM();

  renderTaskFormula();
  renderDataParams();
  initInputModeControls();
  renderDimensionInputs();
  renderBasisOptions();
  renderSingularValueEditor();
  bindSVPresetButtons();
  initModelControls();
  initTrainingControls();
  renderNetworkViz();
  validateDimensions();
  validateInputMode();

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
