// ============================================================================
// APPLICATION STATE
// ============================================================================
// Flat state for the deep-linear playground. Persists to localStorage.
//
// matrixSpec is the bookkeeping that lets the σ editor (or future M-editing
// modes) re-render with the right values after a reload. The actual matrix M
// is *not* stored in localStorage — it would be wasteful for large dims and
// is deterministic from matrixSpec. The UI is responsible for calling
// rebuildM() (in app.js) whenever spec fields change so the in-memory M
// stays in sync.

const STORAGE_KEY = 'mlp-trainer-state-deep-linear';

// Inlined defaults — the old defaults.js has been removed; per-widget
// overrides happen by writing directly to AppState fields after construction.
const DEFAULTS = {
  // Network
  hiddenDim1: 30,
  useSecondLayer: true,
  hiddenDim2: 20,
  useThirdLayer: false,
  hiddenDim3: 15,
  modelSeed: 0,

  // Target function dimensions (also implicit in matrixSpec / M, but kept
  // here so the UI can read them without a built M)
  inputDim: 5,
  outputDim: 3,

  // How M is constructed. The UI reads `mode` to pick which editor to show;
  // mode-specific fields are siblings.
  matrixSpec: {
    mode: 'singular-values',
    singularValues: [1.0, 0.5, 0.25],
    basisSeed: 0,
    randomBasis: true
  },

  // Data
  nTrain: 100,
  dataSeed: 0,
  inputMode: 'whitened',  // 'whitened' (exact Σ_x = I) or 'gaussian' (iid N(0,I), approximate)

  // Training
  eta: 1.4,
  initScale: 0.01,

  // Initialization mode. When alignedInit is true, the model is initialized
  // *in the target's SVD basis* (per Saxe et al). In that case initScale ε is
  // re-interpreted: it becomes the product-matrix singular value at t=0, so
  // each layer carries ε^(1/L) per mode. randomO controls whether the
  // hidden-space orthogonal rotations O are random (seeded by modelSeed) or
  // identity. See model.js for the construction.
  alignedInit: false,
  randomO: false,

  // Plot
  logScale: false,
  logScaleX: false,
  xAxisMode: 'step',     // 'step' | 'teff' (η·step axis toggle)
  emaWindow: 1,
  showPrediction: true,  // overlay Saxe analytic prediction on sharpness plot
  clipToEos: true,       // force sharpness y-axis to include the 2/η threshold

  // Diagnostics
  showInitWeights: false // show numerical t=0 weights in the model details panel
};

export class AppState {
  // storageKey: optional override for localStorage key. The default is the
  // deep-linear page's key.
  constructor(storageKey = STORAGE_KEY) {
    this.storageKey = storageKey;
    this.resetToDefaults();
  }

  toJSON() {
    return {
      hiddenDim1: this.hiddenDim1,
      useSecondLayer: this.useSecondLayer,
      hiddenDim2: this.hiddenDim2,
      useThirdLayer: this.useThirdLayer,
      hiddenDim3: this.hiddenDim3,
      modelSeed: this.modelSeed,

      inputDim: this.inputDim,
      outputDim: this.outputDim,
      matrixSpec: this.matrixSpec,

      nTrain: this.nTrain,
      dataSeed: this.dataSeed,
      inputMode: this.inputMode,

      eta: this.eta,
      initScale: this.initScale,

      alignedInit: this.alignedInit,
      randomO: this.randomO,

      logScale: this.logScale,
      logScaleX: this.logScaleX,
      xAxisMode: this.xAxisMode,
      emaWindow: this.emaWindow,
      showPrediction: this.showPrediction,
      clipToEos: this.clipToEos,
      showInitWeights: this.showInitWeights
    };
  }

  fromJSON(json) {
    if (!json) return;
    if (json.hiddenDim1 !== undefined) this.hiddenDim1 = json.hiddenDim1;
    if (json.useSecondLayer !== undefined) this.useSecondLayer = json.useSecondLayer;
    if (json.hiddenDim2 !== undefined) this.hiddenDim2 = json.hiddenDim2;
    if (json.useThirdLayer !== undefined) this.useThirdLayer = json.useThirdLayer;
    if (json.hiddenDim3 !== undefined) this.hiddenDim3 = json.hiddenDim3;
    if (json.modelSeed !== undefined) this.modelSeed = json.modelSeed;

    if (json.inputDim !== undefined) this.inputDim = json.inputDim;
    if (json.outputDim !== undefined) this.outputDim = json.outputDim;
    // Merge matrixSpec field-by-field so adding new mode-specific fields later
    // doesn't break older saved state.
    if (json.matrixSpec) {
      this.matrixSpec = { ...DEFAULTS.matrixSpec, ...json.matrixSpec };
    }

    if (json.nTrain !== undefined) this.nTrain = json.nTrain;
    if (json.dataSeed !== undefined) this.dataSeed = json.dataSeed;
    if (json.inputMode !== undefined) this.inputMode = json.inputMode;

    if (json.eta !== undefined) this.eta = json.eta;
    if (json.initScale !== undefined) this.initScale = json.initScale;

    if (json.alignedInit !== undefined) this.alignedInit = json.alignedInit;
    if (json.randomO !== undefined) this.randomO = json.randomO;

    if (json.logScale !== undefined) this.logScale = json.logScale;
    if (json.logScaleX !== undefined) this.logScaleX = json.logScaleX;
    if (json.xAxisMode !== undefined) this.xAxisMode = json.xAxisMode;
    if (json.emaWindow !== undefined) this.emaWindow = json.emaWindow;
    if (json.showPrediction !== undefined) this.showPrediction = json.showPrediction;
    if (json.clipToEos !== undefined) this.clipToEos = json.clipToEos;
    if (json.showInitWeights !== undefined) this.showInitWeights = json.showInitWeights;
  }

  save() {
    try { localStorage.setItem(this.storageKey, JSON.stringify(this.toJSON())); }
    catch (e) { console.warn('Failed to save state:', e); }
  }

  load() {
    try {
      const saved = localStorage.getItem(this.storageKey);
      if (saved) { this.fromJSON(JSON.parse(saved)); return true; }
    } catch (e) { console.warn('Failed to load state:', e); }
    return false;
  }

  resetToDefaults() {
    this.hiddenDim1 = DEFAULTS.hiddenDim1;
    this.useSecondLayer = DEFAULTS.useSecondLayer;
    this.hiddenDim2 = DEFAULTS.hiddenDim2;
    this.useThirdLayer = DEFAULTS.useThirdLayer;
    this.hiddenDim3 = DEFAULTS.hiddenDim3;
    this.modelSeed = DEFAULTS.modelSeed;

    this.inputDim = DEFAULTS.inputDim;
    this.outputDim = DEFAULTS.outputDim;
    this.matrixSpec = JSON.parse(JSON.stringify(DEFAULTS.matrixSpec));

    this.nTrain = DEFAULTS.nTrain;
    this.dataSeed = DEFAULTS.dataSeed;
    this.inputMode = DEFAULTS.inputMode;

    this.eta = DEFAULTS.eta;
    this.initScale = DEFAULTS.initScale;

    this.alignedInit = DEFAULTS.alignedInit;
    this.randomO = DEFAULTS.randomO;

    this.logScale = DEFAULTS.logScale;
    this.logScaleX = DEFAULTS.logScaleX;
    this.xAxisMode = DEFAULTS.xAxisMode;
    this.emaWindow = DEFAULTS.emaWindow;
    this.showPrediction = DEFAULTS.showPrediction;
    this.clipToEos = DEFAULTS.clipToEos;
    this.showInitWeights = DEFAULTS.showInitWeights;
  }

  /** Convenience: layer widths as a plain array. */
  hiddenDims() {
    const dims = [this.hiddenDim1];
    if (this.useSecondLayer) dims.push(this.hiddenDim2);
    if (this.useSecondLayer && this.useThirdLayer) dims.push(this.hiddenDim3);
    return dims;
  }
}
