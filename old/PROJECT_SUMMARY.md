# Deep Linear Network Playground — File Map

Browser playground: trains a deep linear MLP on linear regression with a user-specified $W^\star$, plots loss + top Hessian eigenvalues, tracks edge-of-stability.

## Dependency chain (who imports / calls whom)

```
index.html
  └── app.js  ──────────────────┐ (entry point)
       │  imports               │
       ├── state.js             │  AppState (read/write settings, persist)
       ├── matrix.js            │  buildMFromSpec → builds W*
       ├── visualization.js     │  LossChart, RightChart
       │     ├── chart-utils.js
       │     └── incremental-cache.js
       └── simulation.js        │  Simulation (the orchestrator)
              imports           │
              ├── model.js              MLP
              ├── training.js           Trainer
              │     (Trainer.step / computeGradientFlat)
              ├── inputs.js             generateInputs
              │     └── rng.js
              ├── data-generator.js     generateLinearData (Y = M·X)
              └── hessian.js            lanczosTopEigenvalues
                    (calls Trainer.computeGradientFlat for HvP)

rng.js is also imported by inputs.js, matrix.js, model.js (seeded PRNG primitives)
```

## Typical request → which files to touch

- **"Add a new control / change a label / re-style a panel"** → `index.html`, `styles.css`, `app.js` (wire it). Add a field to `state.js` if it should persist.
- **"Change a default value"** → `state.js` (the `DEFAULTS` object).
- **"Add a new way to specify $W^\star$"** → `matrix.js` (new mode in `buildMFromSpec`), then `app.js` (editor UI) and `state.js` (matrixSpec defaults).
- **"Add a new input distribution"** → `inputs.js` (new mode in `generateInputs`), then `app.js` (toggle).
- **"Add a new plot or curve"** → `visualization.js` (new chart class or new dataset), `index.html` (canvas), `app.js` (wiring).
- **"Add a new training-loop diagnostic"** → compute it in `simulation.js` (`runLoop`), expose via `getState()`, plot via `visualization.js`.
- **"Change optimizer / loss"** → `training.js`.
- **"Change Hessian computation"** → `hessian.js` (call site is `simulation.js`).
- **"Change the model (e.g. add nonlinearity)"** → `model.js` forward + `training.js` backprop.

## File reference

### UI
- **`index.html`** — page layout, static controls, script tags.
- **`styles.css`** — global styles.
- **`app.js`** — entry point. Wires DOM ↔ `AppState` ↔ `Simulation`. Renders dynamic controls (σ editor, dim/basis/data inputs), draws SVG network cartoon, handles start/pause/reset, calls `buildMFromSpec` to keep `currentM` fresh.

### State
- **`state.js`** — `AppState`: holds all persisted settings; `save()` / `load()` use localStorage.

### Simulation
- **`simulation.js`** — `Simulation`: `captureParams` → `initialize` → `start`/`pause`/`reset`. `runLoop` calls `Trainer.step` + `lanczosTopEigenvalues` each iteration, captures top eigenvector at edge of stability, computes gradient projection.
- **`training.js`** — `Trainer.step` (one full-batch GD update, returns loss); `Trainer.computeGradientFlat` (used by Hessian HvP).
- **`hessian.js`** — `lanczosTopEigenvalues`: top-k eigs/eigenvectors via Lanczos with finite-difference HvP (calls `Trainer.computeGradientFlat`). Includes a tridiagonal eigensolver.

### Domain
- **`model.js`** — `MLP`: layer stack, muP init, `forward`.
- **`matrix.js`** — `buildMFromSpec` / `buildMFromSingularValues`: construct $W^\star = U\Sigma V^T$.
- **`inputs.js`** — `generateInputs`: builds $X$ in `'gaussian'` or `'whitened'` (exact $\Sigma_x = I$) mode.
- **`data-generator.js`** — `generateLinearData`: $Y = M \cdot X$.
- **`rng.js`** — `mulberry32` (seeded uniform), `seededRandn` (Box-Muller).

### Plotting
- **`visualization.js`** — `LossChart` (loss + EMA), `RightChart` (top eigenvalues + $2/\eta$ threshold). Both expose `update(history, eta)` and `clear()`.
- **`chart-utils.js`** — `baseChartOptions`, `formatTickLabel`.
- **`incremental-cache.js`** — `IncrementalCache`: incremental downsampling + EMA. Used by `LossChart`.
