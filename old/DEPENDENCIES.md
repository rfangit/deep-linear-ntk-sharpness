# `train_widget` Module Dependencies

This document describes the file structure and dependency graph of the
`train_widget` deep-linear playground, so the codebase can be navigated
without re-reading every file, and so trimmed-down variants (single-purpose
demo widgets with fewer knobs) can be carved out with minimal cargo-culting.

The widget trains a deep linear MLP on linear regression with a user-specified
target matrix $W^\star$, plots loss and top Hessian eigenvalues, overlays the
Saxe analytic prediction, and tracks edge-of-stability diagnostics.

## File list

| Layer | File | Purpose |
|-------|------|---------|
| Entry | `train_widget.html` | DOM, controls, script tags |
| Entry | `train_widget.js` | Wires DOM ↔ AppState ↔ Simulation, builds dynamic controls (σ editor, dim/basis inputs), draws SVG network cartoon |
| UI state | `state.js` | `AppState`: holds all persisted settings; `save()` / `load()` via localStorage |
| Sim orchestrator | `simulation.js` | `Simulation`: full training loop, Hessian eigenvalues per step, edge-of-stability eigenvector capture, Saxe theory prediction |
| Domain — model | `model.js` | `MLP`: deep linear stack (no biases, no nonlinearities), muP or aligned init |
| Domain — training | `training.js` | `Trainer.step` (full-batch GD), `Trainer.computeGradientFlat` (for Hessian-vector products) |
| Domain — Hessian | `hessian.js` | `lanczosTopEigenvalues`: top-k eigs/eigenvectors via Lanczos with finite-difference HvP |
| Domain — theory | `theory.js` | `SaxePredictor`: integrates the Saxe ODE alongside training, predicts loss and Hessian eigenvalues |
| Data — matrix | `matrix.js` | `buildMFromSpec` / `buildMComponentsFromSpec`: construct $W^\star = U\Sigma V^\top$ from user σ |
| Data — inputs | `inputs.js` | `generateInputs`: builds X in `'gaussian'` or `'whitened'` mode |
| Data — labels | `data-generator.js` | `generateLinearData`: $Y = M \cdot X$ |
| Data — rng | `rng.js` | `mulberry32`, `seededRandn` |
| Plot — charts | `visualization.js` | `LossChart`, `RightChart`. Both expose `update()`, `clear()`, `setShowPrediction()`, etc. |
| Plot — chart utils | `chart-utils.js` | `baseChartOptions`, `formatTickLabel` |
| Plot — caching | `incremental-cache.js` | `IncrementalCache`: incremental downsampling + EMA |
| Style | `styles.css` | Global styles |

## Dependency graph

```
train_widget.html
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

Nothing imports from `train_widget.js` — it's a leaf entry point. Removing or
renaming it leaves every other file untouched.

## What each "layer" does

**Entry (`train_widget.html` + `train_widget.js`).** The UI surface. All knobs
the user can turn live here: dim inputs, σ editor, basis options, data params,
inputs mode, learning rate, init scale, plot toggles, start/pause/reset
buttons. `train_widget.js` reads/writes `AppState`, builds the dynamic
controls, calls `buildMComponentsFromSpec` to keep `currentM`, `currentU`,
`currentV` fresh, and wires the simulation lifecycle. **This is the file you
edit to change UI; everything else stays put.**

**UI state (`state.js`).** A flat object of every persisted setting. New
controls add a field here, in `toJSON`, in `fromJSON`, and in
`resetToDefaults`. The matrix M is *not* stored — it's rebuilt from
`matrixSpec` each time the page loads.

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
| Add a new control / change a label / re-style a panel | `train_widget.html`, `styles.css`, `train_widget.js`. Add a field to `state.js` if it should persist. |
| Change a default value | `state.js` (the `DEFAULTS` object) |
| Add a new way to specify $W^\star$ | `matrix.js` (new mode in `buildMFromSpec` and `buildMComponentsFromSpec`), then `train_widget.js` (editor UI) and `state.js` (matrixSpec defaults) |
| Add a new input distribution | `inputs.js` (new mode in `generateInputs`), then `train_widget.js` (toggle) |
| Add a new plot or curve | `visualization.js` (new chart class or new dataset), `train_widget.html` (canvas), `train_widget.js` (wiring) |
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

Most "less knobs" requests are answered by writing a smaller `*.html` +
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
state:

```js
const appState = new AppState('my-simple-widget-state');
```

The current `train_widget` uses `'mlp-trainer-state-train-widget'` (set as a
default in `state.js`'s `STORAGE_KEY`). When carving out a variant, pick a
new key like `'mlp-trainer-state-{variant-name}'`.

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
