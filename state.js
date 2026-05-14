// ============================================================================
// APPLICATION STATE
// ============================================================================
// Flat state for the deep-linear playground. Persists to localStorage.
//
// matrixSpec is the bookkeeping that lets the σ editor (or future M-editing
// modes) re-render with the right values after a reload. The actual matrix M
// is *not* stored in localStorage — it would be wasteful for large dims and
// is deterministic from matrixSpec. The UI is responsible for rebuilding M
// (in train_widget.js) whenever spec fields change so the in-memory M
// stays in sync.

const STORAGE_KEY = 'mlp-trainer-state-deep-linear';

// Single source of truth. Adding a new field here automatically extends
// toJSON / fromJSON / resetToDefaults (they iterate Object.keys(DEFAULTS)).
// `matrixSpec` is special-cased on load so adding new mode-specific subfields
// later doesn't break older saved state — see fromJSON below.
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

const FIELD_NAMES = Object.keys(DEFAULTS);

function cloneDefault(name) {
  const v = DEFAULTS[name];
  // Only matrixSpec is an object; everything else is a primitive.
  return (v !== null && typeof v === 'object') ? JSON.parse(JSON.stringify(v)) : v;
}

export class AppState {
  // storageKey: optional override for localStorage key. The default is the
  // deep-linear page's key.
  constructor(storageKey = STORAGE_KEY) {
    this.storageKey = storageKey;
    this.resetToDefaults();
  }

  toJSON() {
    const out = {};
    for (const name of FIELD_NAMES) out[name] = this[name];
    return out;
  }

  fromJSON(json) {
    if (!json) return;
    for (const name of FIELD_NAMES) {
      if (json[name] === undefined) continue;
      // Special case: merge matrixSpec onto its defaults so adding new
      // mode-specific subfields later doesn't break older saved state.
      if (name === 'matrixSpec') {
        this.matrixSpec = { ...DEFAULTS.matrixSpec, ...json.matrixSpec };
      } else {
        this[name] = json[name];
      }
    }
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
    for (const name of FIELD_NAMES) this[name] = cloneDefault(name);
  }

  /** Convenience: layer widths as a plain array. */
  hiddenDims() {
    const dims = [this.hiddenDim1];
    if (this.useSecondLayer) dims.push(this.hiddenDim2);
    if (this.useSecondLayer && this.useThirdLayer) dims.push(this.hiddenDim3);
    return dims;
  }
}
