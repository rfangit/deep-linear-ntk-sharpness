// ============================================================================
// SIMULATION - Main training loop controller
// ============================================================================
// Drives the deep-linear playground: builds an MLP, runs full-batch GD,
// periodically computes top-k Hessian eigenvalues, and (once λ_max enters a
// band near 2/η) captures the top eigenvector and tracks per-step gradient
// projection onto it for edge-of-stability diagnostics.
//
// captureParams takes the explicit target matrix M; inputDim and outputDim
// are read off M.shape. Inputs are sampled (or constructed-whitened) via
// inputs.js, then Y = M·X via data-generator.js.
//
// In addition to the measured Lanczos eigenvalues, each step also advances
// the Saxe-analytic ODE (see theory.js) and stores the predicted GN-Hessian
// eigenvalues, so the chart can overlay theory vs. measurement.

import { MLP } from './model.js';
import { Trainer } from './training.js';
import { generateInputs } from './inputs.js';
import { generateLinearData } from './data-generator.js';
import { lanczosTopEigenvalues } from './hessian.js';
import { SaxePredictor } from './theory.js';

export class Simulation {
  constructor(options = {}) {
    this.isRunning = false;
    this.iteration = 0;
    this.lossHistory = [];          // { iteration, loss }
    this.eigenvalueHistory = [];    // { iteration, eigs: number[] }
    this.predictedEigenvalueHistory = []; // { iteration, eigs: number[] } (Saxe theory)
    this.predictedLossHistory = [];       // { iteration, loss: number }   (Saxe theory)
    this.params = null;
    this.model = null;
    this.trainer = null;
    this.dataset = null;
    this.dataYArrays = null;        // cached array-wrapped train targets (for Hessian)
    this.predictor = null;          // SaxePredictor (theory.js) — null until initialize()
    this.animationFrameId = null;

    // Optional: DOM id for steps/sec display (null = skip display)
    this.stepsPerSecId = options.stepsPerSecId !== undefined ? options.stepsPerSecId : 'stepsPerSec';

    // Callbacks
    this.onFrameUpdate = null;
    this.onDiverge = null;
    this.onAutoStop = null;     // fired once when the auto-stop cooldown elapses

    // Steps per second tracking
    this.stepCounts = [];
    this.totalSteps = 0;
    this.STEPS_PER_SEC_WINDOW = 60;
    this.lastStepsPerSecUpdate = 0;
    this.STEPS_PER_SEC_UPDATE_INTERVAL = 250;

    // Adaptive stepping
    this.TARGET_FRAME_TIME = 25;
    this.avgStepTime = 0.8;
    this.STEP_TIME_ALPHA = 0.15;

    // Hessian eigenvalue computation frequency (every step is expensive but
    // gives smooth curves; bump for speed if needed).
    this.HESSIAN_INTERVAL = 1;

    // Hessian computation defaults.
    //
    // The chart only displays its own top-kEigs (visualization.js, default 3),
    // so requesting more eigenvalues here is cheap. The extras absorb the
    // numerical noise that Lanczos puts at the bottom of its requested set,
    // leaving the displayed top eigenvalues smoother — this is the main lever
    // for suppressing small spurious spikes from loss-of-orthogonality.
    this.hessianOptions = {
      kEigs: options.kEigs || 6,
      numIters: options.hessianNumIters || 30,
      maxIters: options.hessianMaxIters || 100,
      tolRatio: 1e-4
    };

    // ---- Auto-stop on convergence ----
    //
    // Tracks the running minimum loss. Each time a new minimum is found, the
    // "last improvement step" is updated. Training stops automatically once
    // minLossPatienceSteps steps have elapsed since the last improvement —
    // i.e. 20 steps after the loss bottoms out.
    //
    // Set minLossPatienceSteps to 0 (or negative) to disable.
    this.convergenceLossThreshold = options.convergenceLossThreshold !== undefined
      ? options.convergenceLossThreshold
      : 0.01;   // kept for the onAutoStop message in train_widget.js
    this.convergenceCooldownSteps = options.convergenceCooldownSteps !== undefined
      ? options.convergenceCooldownSteps
      : 200;    // kept for the onAutoStop message in train_widget.js
    this.minLossPatienceSteps = options.minLossPatienceSteps !== undefined
      ? options.minLossPatienceSteps
      : 20;
    this.minLoss = Infinity;            // running minimum loss seen so far
    this.minLossStep = null;            // step at which minLoss was last improved
    this.convergenceCrossStep = null;   // (legacy) step at which loss first dipped below threshold
    this.autoStopFired = false;         // set once patience elapsed; cleared on reset

    // ---- Edge-of-Stability eigenvector tracking ----
    //
    // When the top Hessian eigenvalue first enters the proximity band
    //   λ_max >= (1 - sharpnessProximityThreshold) * (2/η)
    // Lanczos is re-run with returnEigenvectors:true and the resulting top
    // eigenvector is stored permanently in topEigenvector (a flat array of
    // length P, same parameter order as hessian.js / computeGradientFlat).
    //
    // After capture, every training step computes the dot product of the raw
    // gradient with topEigenvector and appends it to gradProjectionHistory,
    // giving a step-by-step record of how much gradient descent moves along
    // the sharpest direction of the loss landscape.
    //
    // sharpnessProximityThreshold: fraction of 2/eta (default 0.05 = 5%).
    //   Override via: new Simulation({ sharpnessProximityThreshold: 0.10 })
    this.sharpnessProximityThreshold = options.sharpnessProximityThreshold !== undefined
      ? options.sharpnessProximityThreshold
      : 0.05;

    this.topEigenvector = null;
    this.eigenvectorCaptureStep = null;
    this.eigenvectorCaptureValue = null;

    // Per-step gradient projections onto topEigenvector.
    // Each entry: { iteration, projection, gradNorm }
    // projection = cos(g, v̂) ∈ [-1, 1].
    this.gradProjectionHistory = [];
  }

  /**
   * Capture parameters for the next run. Call before start().
   *
   * @param {object}     p
   * @param {number[][]} p.M           Target matrix, shape [outputDim][inputDim].
   * @param {number}     p.dataSeed    Seed for input construction.
   * @param {number}     p.nTrain
   * @param {number}     p.initScale   Weight init scale ε.
   * @param {number[]}   p.hiddenDims  One or more hidden widths.
   * @param {number}     p.eta         GD learning rate.
   * @param {number}     p.modelSeed   Seed for weight initialization.
   * @param {'gaussian'|'whitened'} [p.inputMode='whitened']
   *                                   How X is constructed. 'whitened' makes
   *                                   (1/N) X^T X = I exactly.
   * @param {number[]}  [p.sigmaStar]  Target singular values of M (length
   *                                   min(outputDim, inputDim)). Used for
   *                                   the Saxe theory prediction. Pad with
   *                                   zeros for modes above rank(M). If
   *                                   omitted, the prediction is disabled.
   */
  captureParams(p) {
    if (!Array.isArray(p.M) || p.M.length === 0 || !Array.isArray(p.M[0])) {
      throw new Error('Simulation.captureParams: M must be a non-empty 2D array');
    }
    this.params = {
      M: p.M,
      dataSeed: p.dataSeed !== undefined ? p.dataSeed : 0,
      nTrain: p.nTrain,
      initScale: p.initScale !== undefined ? p.initScale : 1.0,
      hiddenDims: p.hiddenDims,
      eta: p.eta,
      modelSeed: p.modelSeed,
      inputMode: p.inputMode !== undefined ? p.inputMode : 'whitened',

      // Target singular values for the theory prediction. Caller is
      // responsible for passing the same sigma values used to construct M
      // (see matrix.js buildMComponentsFromSpec); we don't re-SVD M here.
      sigmaStar: Array.isArray(p.sigmaStar) ? p.sigmaStar.slice() : null,

      // Aligned-init parameters. When alignedInit is true the model is built
      // in the target's SVD basis (see model.js _alignedInit). U and V are
      // the target's bases — pass them via buildMComponentsFromSpec at the
      // call site. randomO controls whether hidden-layer rotations are
      // random (seeded by modelSeed) or identity.
      alignedInit: p.alignedInit === true,
      randomO:     p.randomO === true,
      U:           p.U !== undefined ? p.U : null,
      V:           p.V !== undefined ? p.V : null
    };
    this.model = null;
    this.trainer = null;
    this.predictor = null;
  }

  initialize() {
    if (!this.params) throw new Error('No parameters captured. Call captureParams first.');

    const p = this.params;
    const inputDim = p.M[0].length;
    const outputDim = p.M.length;

    // Build X (iid Gaussian or exact-whitened), then Y = M·X.
    const X = generateInputs({
      inputDim,
      nTrain: p.nTrain,
      dataSeed: p.dataSeed,
      mode: p.inputMode
    });
    this.dataset = generateLinearData({ M: p.M, X });

    this.dataYArrays = this.dataset.y.map(y => Array.isArray(y) ? y : [y]);

    // Build model and trainer. When aligned-init is requested, U and V must
    // have been passed (the caller computes them via
    // buildMComponentsFromSpec). Fall back to muP random init otherwise.
    const layerSizes = [inputDim, ...p.hiddenDims, outputDim];
    let alignedOpts = null;
    if (p.alignedInit && p.U && p.V) {
      alignedOpts = { U: p.U, V: p.V, randomO: !!p.randomO };
    }
    this.model = new MLP(layerSizes, p.modelSeed, p.initScale, alignedOpts);
    this.trainer = new Trainer(this.model, p.eta, this.dataset);

    // Build the Saxe theory predictor (if σ⋆ is available). L counts only
    // the *weight matrices* (= layerSizes.length - 1), which is the same L
    // that appears in f(x) = W_L · … · W_1 · x in the writeup.
    if (p.sigmaStar) {
      const L = layerSizes.length - 1;
      // Pad/truncate sigmaStar to length r = min(inputDim, outputDim) so the
      // predictor matches the σ_i pairing in the GN eigenvalue formula. Modes
      // beyond the supplied σ⋆ entries are zeros (low-rank-M directions).
      const r = Math.min(inputDim, outputDim);
      const sigmaStarPadded = new Array(r);
      for (let i = 0; i < r; i++) {
        sigmaStarPadded[i] = (p.sigmaStar[i] !== undefined) ? p.sigmaStar[i] : 0;
      }
      this.predictor = new SaxePredictor({
        sigmaStar: sigmaStarPadded,
        L,
        epsilon: p.initScale
      });
    } else {
      this.predictor = null;
    }

    // Reset histories
    this.iteration = 0;
    this.lossHistory = [];
    this.eigenvalueHistory = [];
    this.predictedEigenvalueHistory = [];
    this.predictedLossHistory = [];
    this.gradProjectionHistory = [];
    this.topEigenvector = null;
    this.eigenvectorCaptureStep = null;
    this.eigenvectorCaptureValue = null;

    // Seed the predicted history with the iteration-0 prediction so the
    // overlay has a point to start from even before the first training step.
    // (The measured history doesn't get a seed because Lanczos hasn't run
    // yet at iter 0; the chart handles that asymmetry fine.)
    if (this.predictor) {
      this.predictedEigenvalueHistory.push({
        iteration: 0,
        eigs: this.predictor.currentEigenvalues()
      });
      this.predictedLossHistory.push({
        iteration: 0,
        loss: this.predictor.currentLoss()
      });
    }
  }

  /**
   * Compute top-k Hessian eigenvalues (and eigenvectors) using Lanczos.
   * Eigenvectors are always requested so that the top one is available for
   * the capture check below without a second Lanczos pass.
   *
   * If no eigenvector has been stored yet and the top eigenvalue is within
   * sharpnessProximityThreshold of the critical threshold 2/eta, the top
   * eigenvector from this run is stored permanently in this.topEigenvector.
   *
   * Returns the eigenvalue array (sorted ascending), or null on failure.
   */
  computeHessianEigenvalues() {
    if (!this.trainer || !this.dataset) return null;

    const result = lanczosTopEigenvalues(
      this.trainer,
      this.dataset.x,
      this.dataYArrays,
      { ...this.hessianOptions, returnEigenvectors: true }
    );

    const eigs = result.eigenvalues;

    // ---- Eigenvector capture check (store-once, outside Lanczos) ----
    if (!this.topEigenvector && eigs && eigs.length > 0 && this.params) {
      const lambdaMax = eigs[eigs.length - 1];
      const threshold = 2 / this.params.eta;
      const proximityFraction = lambdaMax / threshold;

      if (proximityFraction >= (1 - this.sharpnessProximityThreshold)) {
        if (result.eigenvectors && result.eigenvectors.length > 0) {
          // eigenvectors sorted ascending — last one corresponds to top eigenvalue
          this.topEigenvector = result.eigenvectors[result.eigenvectors.length - 1];
          this.eigenvectorCaptureStep = this.iteration;
          this.eigenvectorCaptureValue = lambdaMax;
          console.log(
            `[EoS] Top eigenvector captured at step ${this.iteration}.` +
            ` λ_max=${lambdaMax.toFixed(4)}, threshold=${threshold.toFixed(4)}` +
            ` (${(proximityFraction * 100).toFixed(1)}% of 2/η)`
          );
        }
      }
    }

    return eigs;
  }

  /**
   * Project the most recent gradient onto the stored top eigenvector.
   * Returns null if the eigenvector hasn't been captured yet or if no
   * gradient is available.
   *
   * Returns:
   *   projection — cosine similarity dot(g/||g||, v̂) ∈ [-1, 1]
   *                independent of learning rate or gradient magnitude.
   *   gradNorm   — ||g||, the Euclidean norm of the raw gradient.
   *
   * @returns {{ projection: number, gradNorm: number }|null}
   */
  computeGradientProjection() {
    if (!this.topEigenvector || !this.trainer || !this.trainer.lastGradFlat) return null;

    const g = this.trainer.lastGradFlat;
    const v = this.topEigenvector;

    if (g.length !== v.length) {
      console.warn('[EoS] Gradient and eigenvector length mismatch:', g.length, v.length);
      return null;
    }

    let normSq = 0;
    for (let i = 0; i < g.length; i++) normSq += g[i] * g[i];
    const gradNorm = Math.sqrt(normSq);

    if (gradNorm < 1e-12) return { projection: 0, gradNorm: 0 };

    let dot = 0;
    for (let i = 0; i < g.length; i++) dot += g[i] * v[i];
    return { projection: dot / gradNorm, gradNorm };
  }

  start() {
    if (this.isRunning) return;
    if (!this.model) this.initialize();

    this.stepCounts = [];
    this.totalSteps = 0;
    this.lastStepsPerSecUpdate = 0;
    if (this.stepsPerSecId) {
      const spsEl = document.getElementById(this.stepsPerSecId);
      if (spsEl) spsEl.textContent = '—';
    }

    this.isRunning = true;
    this.runLoop();
  }

  pause() {
    this.isRunning = false;
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
  }

  reset() {
    this.pause();
    this.model = null;
    this.trainer = null;
    this.dataset = null;
    this.dataYArrays = null;
    this.predictor = null;
    this.iteration = 0;
    this.lossHistory = [];
    this.eigenvalueHistory = [];
    this.predictedEigenvalueHistory = [];
    this.predictedLossHistory = [];
    this.gradProjectionHistory = [];
    this.topEigenvector = null;
    this.eigenvectorCaptureStep = null;
    this.eigenvectorCaptureValue = null;
    this.convergenceCrossStep = null;
    this.autoStopFired = false;
    this.minLoss = Infinity;
    this.minLossStep = null;
    this.stepCounts = [];
    this.totalSteps = 0;
    this.lastStepsPerSecUpdate = 0;
    if (this.stepsPerSecId) {
      const spsEl = document.getElementById(this.stepsPerSecId);
      if (spsEl) spsEl.textContent = '—';
    }
  }

  runLoop() {
    if (!this.isRunning) return;

    const frameStart = performance.now();
    let stepsThisFrame = 0;
    const timeBudget = this.TARGET_FRAME_TIME - 1.5;

    while (true) {
      const elapsed = performance.now() - frameStart;
      if (elapsed + this.avgStepTime > timeBudget && stepsThisFrame > 0) break;

      const stepStart = performance.now();
      const loss = this.trainer.step();
      const stepTime = performance.now() - stepStart;

      this.avgStepTime = this.STEP_TIME_ALPHA * stepTime + (1 - this.STEP_TIME_ALPHA) * this.avgStepTime;

      this.iteration++;
      this.lossHistory.push({ iteration: this.iteration, loss: loss });

      // Numerical instability check
      if (!isFinite(loss) || loss > 100000) {
        this.isRunning = false;
        if (this.onFrameUpdate) this.onFrameUpdate();
        if (this.onDiverge) this.onDiverge(this.iteration, loss);
        return;
      }

      // Auto-stop: stop minLossPatienceSteps steps after loss first drops
      // below convergenceLossThreshold.
      if (this.minLossPatienceSteps > 0 && !this.autoStopFired) {
        if (this.convergenceCrossStep === null && loss < this.convergenceLossThreshold) {
          this.convergenceCrossStep = this.iteration;
        }
        if (
          this.convergenceCrossStep !== null &&
          this.iteration >= this.convergenceCrossStep + this.minLossPatienceSteps
        ) {
          this.autoStopFired = true;
          this.isRunning = false;
          if (this.onFrameUpdate) this.onFrameUpdate();
          if (this.onAutoStop) this.onAutoStop(this.iteration, this.convergenceCrossStep, loss);
          return;
        }
      }

      // Hessian eigenvalues (expensive — bump HESSIAN_INTERVAL for speed)
      if (this.iteration % this.HESSIAN_INTERVAL === 0) {
        const eigs = this.computeHessianEigenvalues();
        if (eigs) {
          this.eigenvalueHistory.push({ iteration: this.iteration, eigs });
        }
      }

      // Theory prediction: advance the Saxe ODE by dt = η per step so that
      // theory time tracks η·step exactly. Cheap compared to Lanczos.
      // Record both predicted eigenvalues (for the sharpness plot) and
      // predicted loss (for the loss plot) from the same advanced σ vector.
      if (this.predictor) {
        const predEigs = this.predictor.step(this.params.eta);
        this.predictedEigenvalueHistory.push({
          iteration: this.iteration,
          eigs: predEigs
        });
        this.predictedLossHistory.push({
          iteration: this.iteration,
          loss: this.predictor.currentLoss()
        });
      }

      // Gradient projection (cheap; no-op until eigenvector is captured)
      const projResult = this.computeGradientProjection();
      if (projResult !== null) {
        this.gradProjectionHistory.push({
          iteration: this.iteration,
          projection: projResult.projection,
          gradNorm: projResult.gradNorm
        });
      }

      stepsThisFrame++;
      if (stepsThisFrame >= 1000) break;
    }

    this.updateStepsPerSec(stepsThisFrame);
    if (this.onFrameUpdate) this.onFrameUpdate();

    this.animationFrameId = requestAnimationFrame(() => this.runLoop());
  }

  updateStepsPerSec(stepsThisFrame) {
    const now = performance.now();
    this.totalSteps += stepsThisFrame;
    this.stepCounts.push([now, this.totalSteps]);

    if (this.stepCounts.length > this.STEPS_PER_SEC_WINDOW) this.stepCounts.shift();
    if (now - this.lastStepsPerSecUpdate < this.STEPS_PER_SEC_UPDATE_INTERVAL) return;
    this.lastStepsPerSecUpdate = now;

    if (this.stepCounts.length < 2) return;

    const [oldestTime, oldestSteps] = this.stepCounts[0];
    const [newestTime, newestSteps] = this.stepCounts[this.stepCounts.length - 1];
    const stepsPerSec = (newestSteps - oldestSteps) / ((newestTime - oldestTime) / 1000);

    if (this.stepsPerSecId) {
      const spsEl = document.getElementById(this.stepsPerSecId);
      if (spsEl) spsEl.textContent = Math.round(stepsPerSec).toString();
    }
  }

  getState() {
    return {
      iteration: this.iteration,
      lossHistory: this.lossHistory,
      eigenvalueHistory: this.eigenvalueHistory,
      predictedEigenvalueHistory: this.predictedEigenvalueHistory,
      predictedLossHistory: this.predictedLossHistory,
      gradProjectionHistory: this.gradProjectionHistory,
      topEigenvector: this.topEigenvector,
      eigenvectorCaptureStep: this.eigenvectorCaptureStep,
      eigenvectorCaptureValue: this.eigenvectorCaptureValue,
      eta: this.params ? this.params.eta : 0.01,
      isRunning: this.isRunning
    };
  }
}
