# `train_widget` Module Dependencies

This document describes the file structure and dependency graph of the
`train_widget` deep-linear playground, so the codebase can be navigated
without re-reading every file, and so trimmed-down variants (single-purpose
demo widgets with fewer knobs) can be carved out with minimal cargo-culting.

The widget trains a deep linear MLP on linear regression with a user-specified
target matrix $W^\star$, plots loss and top Hessian eigenvalues, overlays the
Saxe analytic prediction, and tracks edge-of-stability diagnostics.

## Entry points

| HTML | Loads | Purpose |
|------|-------|---------|
| `index.html` | `train_widget.js` (twice, prefixed `w1` / `w2`) | Blog page embedding two widgets side by side |
| `train_widget.html` | `train_widget.js` (no prefix) | Standalone single-widget dev sandbox |
| `analytic_test.html` | `analytic-test-app.js` | NTK Jacobian-vs-analytic side-by-side check |
| `ntk-widgets.html` | `ntk-app.js` | Self-contained NTK exploration widgets |

`train_widget.js` exports `initWidget(prefix, options)`. With a non-empty
prefix it looks up DOM ids as `${prefix}-${id}`, so the same JS drives both
embedded copies in `index.html` and the standalone page.

## File list (train_widget chain)

| Layer | File | Purpose |
|-------|------|---------|
| Entry | `index.html` / `train_widget.html` | DOM, controls, script tags |
| Entry | `train_widget.js` | Wires DOM ↔ AppState ↔ Simulation, builds dynamic controls (σ editor, dim/basis inputs), draws SVG network cartoon |
| UI state | `state.js` | `AppState`: holds all persisted settings; `save()` / `load()` via localStorage |
| Sim orchestrator | `simulation.js` | `Simulation`: full training loop, Hessian eigenvalues per step, edge-of-stability eigenvector capture, Saxe theory prediction |
| Domain — model | `model.js` | `MLP`: deep linear stack (no biases, no nonlinearities), muP or aligned init |
| Domain — training | `training.js` | `Trainer.step` (full-batch GD), `Trainer.computeGradientFlat` (for Hessian-vector products) |
| Domain — Hessian | `hessian.js` | `lanczosTopEigenvalues`: top-k eigs/eigenvectors via Lanczos with finite-difference HvP |
| Domain — theory | `theory.js` | `SaxePredictor`: integrates the Saxe ODE alongside training, predicts loss and Hessian eigenvalues |
| Data — matrix | `matrix.js` | `buildMComponentsFromSpec` and `randomOrthogonal`: construct $W^\star = U\Sigma V^\top$ from user σ; expose the SVD basis |
| Data — inputs | `inputs.js` | `generateInputs`: builds X in `'gaussian'` or `'whitened'` mode |
| Data — labels | `data-generator.js` | `generateLinearData`: $Y = M \cdot X$ |
| Data — rng | `rng.js` | `mulberry32`, `seededRandn` |
| Plot — charts | `visualization.js` | `LossChart`, `RightChart`. Both expose `update()`, `clear()`, `setShowPrediction()`, etc. |
| Plot — chart utils | `chart-utils.js` | `baseChartOptions`, `formatTickLabel` |
| Plot — caching | `incremental-cache.js` | `IncrementalCache`: incremental downsampling + EMA |
| Style | `styles.css` | Global styles |

## Files outside the train_widget chain

These are loaded only by `analytic_test.html` / `ntk-widgets.html` and don't
participate in the training playground:

| File | Purpose |
|------|---------|
| `analytic-test-app.js` | Driver for `analytic_test.html` |
| `ntk-app.js` | Driver for `ntk-widgets.html` (self-contained — uses its own inlined PRNG, Jacobi eig, Saxe ODE, bar charts, and time-evolution plots) |
| `ntk.js` | NTK matrix via per-output Jacobians; analytic 2-layer formula |
| `ntkModels.js` | Sources of $(W_1, W_2, \dots)$ for NTK widgets — `randomModel`, `analyticModel`, `randomBasis` |
| `ntkDisplay.js` | Render an NTK matrix as a colored heatmap |

`ntkModels.js` re-uses `MLP` from `model.js` and `randomOrthogonal` from
`matrix.js`. `ntk-app.js` is deliberately self-contained (no imports) — see
note in *Open cleanup items* below.

## Dependency graph (train_widget)

```
index.html / train_widget.html
  └── train_widget.js  ──────────────────┐ (entry point)
       │  imports                        │
       ├── state.js                      │  AppState (settings + persistence)
       ├── matrix.js                     │  buildMComponentsFromSpec → M, U, V
       ├── visualization.js              │  LossChart, RightChart
       │     ├── chart-utils.js          │
       │     └── incremental-cache.js    │
       └── simulation.js                 │  Simulation (the orchestrator)
              imports                    │
              ├── model.js               MLP
              ├── training.js            Trainer
              │     (Trainer.step / computeGradientFlat)
              ├── inputs.js              generateInputs
              │     └── rng.js
              ├── data-generator.js      generateLinearData (Y = M·X)
              ├── hessian.js             lanczosTopEigenvalues
              │     (calls Trainer.computeGradientFlat for HvP)
              └── theory.js              SaxePredictor

rng.js is also imported by inputs.js, matrix.js, model.js (seeded PRNG primitives)
```

Nothing imports from `train_widget.js` — it's a leaf entry point.

## What each "layer" does

**Entry (`*.html` + `train_widget.js`).** The UI surface. All knobs the user
can turn live here: dim inputs, σ editor, basis options, data params, inputs
mode, learning rate, init scale, plot toggles, start/pause/reset buttons.
`train_widget.js` reads/writes `AppState`, builds the dynamic controls, calls
`buildMComponentsFromSpec` to keep `currentM`, `currentU`, `currentV` fresh,
and wires the simulation lifecycle. **This is the file you edit to change UI;
everything else stays put.**

**UI state (`state.js`).** A flat object of every persisted setting, driven
off a single `DEFAULTS` table. `toJSON`, `fromJSON`, and `resetToDefaults`
iterate `Object.keys(DEFAULTS)`, so adding a new control means adding one
field to `DEFAULTS` and nothing else (unless it's a nested object — see the
`matrixSpec` special case in `fromJSON`). The matrix M is *not* stored — it's
rebuilt from `matrixSpec` each time the page loads.

**Simulation (`simulation.js`).** The training loop. `captureParams` →
`initialize` → `start`/`pause`/`reset`. `runLoop` calls `Trainer.step` +
`lanczosTopEigenvalues` + `SaxePredictor.step` each iteration, captures top
eigenvector at edge-of-stability, computes gradient projection. Exposes a
`getState()` snapshot for the UI to read.

**Domain (`model.js`, `training.js`, `hessian.js`, `theory.js`).** Pure
numerics, no DOM. Each file is independently testable in Node. `theory.js` is
self-contained and only touches `simulation.js`.

**Data (`matrix.js`, `inputs.js`, `data-generator.js`, `rng.js`).** Build the
target matrix M, the design matrix X, and Y = M·X. Cleanly separated: M
decides "what is the true function?", X decides "what statistics does the
data have?", Y just plugs them together.

**Plot (`visualization.js`, `chart-utils.js`, `incremental-cache.js`).**
Chart.js wrappers. Both charts accept option objects in their constructors to
toggle which series get rendered (so a simpler widget can declare e.g. only
loss + theory overlay, no sharpness plot).

## Typical request → which files to touch

| Request | Files |
|---------|-------|
| Add a new control / change a label / re-style a panel | `train_widget.html` (or `index.html`), `styles.css`, `train_widget.js`. Add a field to `DEFAULTS` in `state.js` if it should persist. |
| Change a default value | `state.js` (the `DEFAULTS` object) |
| Add a new way to specify $W^\star$ | `matrix.js` (new mode in `buildMComponentsFromSpec`), then `train_widget.js` (editor UI) and `state.js` (matrixSpec defaults) |
| Add a new input distribution | `inputs.js` (new mode in `generateInputs`), then `train_widget.js` (toggle) |
| Add a new plot or curve | `visualization.js` (new chart class or new dataset), the HTML entry (canvas), `train_widget.js` (wiring) |
| Add a new training-loop diagnostic | Compute it in `simulation.js` (`runLoop`), expose via `getState()`, plot via `visualization.js` |
| Change optimizer / loss | `training.js` |
| Change Hessian computation | `hessian.js` (call site is `simulation.js`) |
| Change the model (e.g. add a nonlinearity) | `model.js` forward + `training.js` backprop |
| Change/extend the analytic prediction | `theory.js`. `Simulation` already advances and records it; `visualization.js` already plots it. |

## Building trimmed-down variants

The codebase is designed so a simpler widget is a smaller widget — same files,
fewer fields surfaced, with the irrelevant code paths left dormant via
constructor options. There are three main axes you can cut along.

### Axis 1: Trim the UI

`train_widget.js`'s `initWidget(prefix, options)` already supports two trim
flags used by `index.html`: `showLayers` (hide depth controls) and `simple`
(hide EMA / log toggles etc.). For deeper cuts, write a smaller `*.html` +
`*.js` pair. The new entry point can:

- Construct `AppState` with a different `storageKey` (so it doesn't conflict).
- After construction, **overwrite specific fields directly** to set fixed
  values that the user can't change — e.g. `appState.inputMode = 'whitened'`
  if you want to hide the iid/whitened toggle. The UI for those fields simply
  isn't built.
- Skip dynamic controls it doesn't need (e.g. the σ editor — hardcode a
  `matrixSpec` instead).
- Construct `LossChart`/`RightChart` with options that suppress unwanted
  series:
    - `new LossChart('lossChart', { showEma: false, showPrediction: false })`
    - `new RightChart('rightChart', { kEigs: 1, showThreshold: false })`
- Pass a subset of params to `simulation.captureParams`. Anything omitted
  falls back to the simulation's own defaults (e.g. omitting `sigmaStar`
  disables the theory predictor without further changes).

### Axis 2: Drop whole features

Each feature is contained well enough that you can omit it cleanly:

- **No theory prediction**: don't pass `sigmaStar` to `captureParams`. The
  predictor never spins up; `predictedEigenvalueHistory` and
  `predictedLossHistory` stay empty. Chart overlays auto-hide.
- **No sharpness plot**: skip `RightChart` entirely. Lanczos still runs in
  `Simulation` (since it's part of the EoS capture logic). If you also want
  to skip that, set `simulation.HESSIAN_INTERVAL = Infinity` after construction.
- **No EoS / gradient projection**: ignore those fields in `getState()`. They
  cost basically nothing if you don't read them.
- **Fixed depth**: hardcode `appState.useSecondLayer` and `hiddenDims`.
- **Fixed M**: hardcode `state.matrixSpec` to a constant value; skip the
  whole σ editor.

### Axis 3: Replace components

The clean module boundaries mean you can swap a piece without touching the
rest:

- Replace `Trainer` with SGD or Adam? Edit `training.js` (and call site is in
  `simulation.js`).
- Replace `lanczosTopEigenvalues` with a different sharpness estimator? Edit
  `hessian.js` and update the call in `simulation.js`'s
  `computeHessianEigenvalues`.
- Replace `theory.js`? It's only used in `simulation.js` — three call sites
  (`initialize`, `runLoop`, and the histories in `getState`).

## A minimal "loss-only demo" sketch

For the smallest possible variant — show the training loss and nothing else —
the dependency footprint shrinks to:

```
minimal_widget.html
  └── minimal_widget.js
       ├── state.js               (or skip; hardcode everything)
       ├── matrix.js              (still need this to build M)
       ├── visualization.js       (only LossChart)
       │     ├── chart-utils.js
       │     └── incremental-cache.js
       └── simulation.js
              ├── model.js
              ├── training.js
              ├── inputs.js → rng.js
              └── data-generator.js
              (skip hessian.js + theory.js? no — they're imported
               unconditionally by simulation.js. See note below.)
```

Note that `simulation.js` imports `hessian.js` and `theory.js`
unconditionally, so even a "loss only" variant pulls them in. They don't cost
anything at runtime if you set `HESSIAN_INTERVAL = Infinity` and omit
`sigmaStar`, but they're on disk. If that's bad enough to matter, split
`Simulation` into a base class + a sharpness-extension subclass — but that's
a non-trivial refactor and probably not worth it for an in-browser widget.

## Persistence and storage keys

`AppState` uses localStorage to persist settings. Each widget should pass its
own `storageKey` to the constructor so widgets don't trample each other's
state. `train_widget.js` already does this automatically based on the
`prefix` it receives:

- `initWidget('')`     → key `'mlp-trainer-state-train-widget'`
- `initWidget('w1')`   → key `'mlp-trainer-state-w1'`
- `initWidget('w2')`   → key `'mlp-trainer-state-w2'`

The fallback in `state.js`'s `STORAGE_KEY` (`'mlp-trainer-state-deep-linear'`)
applies only if an `AppState` is constructed with no argument — which the
current entry points never do.

When carving out a variant, pass an explicit key:

```js
const appState = new AppState('my-simple-widget-state');
```

## Open cleanup items

- **`ntk-app.js` is self-contained** by design (no imports). It re-implements
  things that exist as proper modules in the train_widget chain (PRNG, Jacobi
  eigensolver, 2-layer Saxe σ(t), simple bar / line plotting). Deduping would
  save several hundred lines but risks breaking the NTK widgets — left as-is
  for now.

## Where to find things at a glance

- **Defaults**: `state.js` → `DEFAULTS`
- **Adaptive frame-rate stepping**: `simulation.js` → `runLoop`, look for
  `TARGET_FRAME_TIME`
- **Hessian frequency**: `simulation.js` → `HESSIAN_INTERVAL` (default 1 =
  every step)
- **Edge-of-stability proximity threshold**: `simulation.js` →
  `sharpnessProximityThreshold` (default 0.05 = 5% of 2/η)
- **Saxe ODE integrator step size**: `theory.js` → `stepSaxeODE` (currently
  two RK4 sub-steps of `dt/2` per call)
- **Chart styling**: `chart-utils.js` and constructors in `visualization.js`
- **Maximum plot points (downsampling)**: `visualization.js` →
  `MAX_PLOT_POINTS`
