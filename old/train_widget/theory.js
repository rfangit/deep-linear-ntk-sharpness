// ============================================================================
// THEORY - Saxe analytic prediction for deep linear networks
// ============================================================================
// Given the user's target singular values {σ_i⋆}, depth L, init scale ε, and
// learning rate η, integrate the Saxe gradient-flow ODE
//
//     dσ_i/dt = L · σ_i^(2 − 2/L) · (σ_i⋆ − σ_i)
//
// alongside training, with the standard correspondence t ↔ η · step. Use the
// integrated singular values to compute the (Gauss-Newton) Hessian eigenvalue
// predictions from the blog post:
//
//   Aligned modes  (k = i,    i ≤ r):       L · s_i^(2(L−1)/L)
//   Cross modes    (k ≠ i, both ≤ r):       Σ_{ℓ=1..L} s_k^(2(ℓ−1)/L) · s_i^(2(L−ℓ)/L)
//   Low-rank-M     (i ≤ r, k > r,  σ_k⋆=0): cross formula with s_k = 0
//   Extra data     (eigenvalue 0):          skipped — uninteresting
//
// All r·n nonzero eigenvalues are computed each step; the caller decides how
// many to plot.
//
// Initial conditions: σ_i(0) = ε^L for every mode. This is the crude product-
// matrix scale under both aligned and muP-Gaussian init — the latter is not
// exact (different modes start at different values under random init) but
// avoids SVD-extraction artifacts that can spuriously inflate the largest
// initial singular value when random vectors happen to align.
//
// Integration: one RK4 step of size dt = η per training step, so the theory
// time t = (current step) · η exactly matches the η·step axis.

/**
 * Right-hand side of the Saxe ODE for a single mode.
 * @param {number} sigma  — current singular value
 * @param {number} sStar  — target singular value
 * @param {number} L      — number of layers
 * @returns {number} dσ/dt
 */
function saxeRHS(sigma, sStar, L) {
  // L · σ^(2 − 2/L) · (σ⋆ − σ).
  // For L = 1 this collapses to σ⋆ − σ. For L ≥ 2 the power is in (0, 2),
  // and we always evaluate at non-negative σ (clamped below), so Math.pow
  // is well-defined.
  if (sigma <= 0) return 0;          // floor: ODE has σ=0 as a fixed point
  const exp = 2 - 2 / L;
  return L * Math.pow(sigma, exp) * (sStar - sigma);
}

/**
 * Advance one mode's singular value by dt using classical RK4.
 * Independent of other modes (decoupled under the analytic assumption).
 *
 * Note on stability: the Saxe ODE linearized at the attractor σ = σ⋆ has
 * Jacobian -L · σ⋆^(2-2/L), so RK4 with dt = η is stable only when
 *     η · L · σ⋆_max^(2-2/L) ≲ 2.78
 * (the real-axis bound of the RK4 stability region). For very large η this
 * bound can be violated and the prediction will blow up near convergence —
 * out of scope for now; the simple one-step integrator is fine in the
 * stable regime.
 */
function rk4StepMode(sigma, sStar, L, dt) {
  const k1 = saxeRHS(sigma,             sStar, L);
  const k2 = saxeRHS(sigma + 0.5*dt*k1, sStar, L);
  const k3 = saxeRHS(sigma + 0.5*dt*k2, sStar, L);
  const k4 = saxeRHS(sigma +     dt*k3, sStar, L);
  let next = sigma + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4);
  if (next < 0) next = 0;            // numerical safety; σ should stay ≥ 0
  return next;
}

/**
 * Advance the full singular-value vector by one RK4 step of size dt, taken
 * as two sub-steps of dt/2. The substepping (introduced because users
 * routinely run at η ≳ 1 where a single RK4 step at dt = η can sit just
 * outside the RK4 stability region) is internal — callers still pass dt = η
 * and the theory time advances by exactly η per call, so the t ↔ η·step
 * correspondence is unchanged.
 *
 * @param {number[]} sigmas      — current σ_i, length r = min(m, k)
 * @param {number[]} sigmaStar   — target σ_i⋆, same length (zeros for modes
 *                                 above the rank of M)
 * @param {number}   L
 * @param {number}   dt          — total advance in theory time (== η)
 * @returns {number[]} new singular value array (fresh allocation)
 */
export function stepSaxeODE(sigmas, sigmaStar, L, dt) {
  const half = dt / 2;
  const tmp = new Array(sigmas.length);
  for (let i = 0; i < sigmas.length; i++) {
    tmp[i] = rk4StepMode(sigmas[i], sigmaStar[i] || 0, L, half);
  }
  const out = new Array(sigmas.length);
  for (let i = 0; i < sigmas.length; i++) {
    out[i] = rk4StepMode(tmp[i], sigmaStar[i] || 0, L, half);
  }
  return out;
}

/**
 * Initial singular values for the Saxe ODE. We use σ_i(0) = ε^L uniformly
 * across all modes. This is the crude product-matrix scale and is exact
 * under aligned init; under muP-Gaussian it's an order-of-magnitude
 * approximation that avoids SVD-extraction artifacts.
 *
 * @param {number} epsilon
 * @param {number} L
 * @param {number} r       — number of singular-value modes to track
 * @returns {number[]}
 */
export function initialSingularValues(epsilon, L, r) {
  const s0 = Math.pow(Math.max(epsilon, 0), L);
  const out = new Array(r);
  for (let i = 0; i < r; i++) out[i] = s0;
  return out;
}

/**
 * Compute all nonzero predicted GN-Hessian eigenvalues from the current
 * singular-value vector. Returns them sorted *descending* (so eigs[0] is the
 * largest, matching what the chart compares against the top measured eig).
 *
 * Modes enumerated:
 *   Aligned  (k = i, σ_i⋆ > 0 and σ_k⋆ > 0):     r entries
 *   Cross    (k ≠ i, both σ > 0):                r(r−1) entries
 *   Low-rank (one of σ_k⋆, σ_i⋆ is 0, the other ≠ 0): collapses cross formula
 *
 * Modes with both σ_k⋆ = 0 and σ_i⋆ = 0 stay at σ=0 forever and contribute
 * zero eigenvalues — skipped.
 *
 * @param {number[]} sigmas      — current singular values, length r_total
 * @param {number[]} sigmaStar   — target σ⋆ (same length); modes with σ⋆ = 0
 *                                 are "low-rank-M" directions
 * @param {number}   L
 * @returns {number[]} sorted descending
 */
export function predictedEigenvalues(sigmas, sigmaStar, L) {
  const r = sigmas.length;
  const eigs = [];

  // Helper: c_{ki} = Σ_{ℓ=1..L} s_k^(2(ℓ−1)/L) · s_i^(2(L−ℓ)/L)
  // For aligned k = i this collapses to L · s_i^(2(L−1)/L).
  const crossEig = (sk, si) => {
    let sum = 0;
    for (let ell = 1; ell <= L; ell++) {
      const pk = (ell - 1) * 2 / L;
      const pi = (L - ell) * 2 / L;
      // pow(0, 0) = 1 in JS; that's correct here (both factors of a degenerate
      // term contribute 1 when their exponent is 0).
      sum += Math.pow(sk, pk) * Math.pow(si, pi);
    }
    return sum;
  };

  for (let i = 0; i < r; i++) {
    for (let k = 0; k < r; k++) {
      const si = sigmas[i];
      const sk = sigmas[k];
      // Skip strictly-zero modes (both target zero AND current zero); the
      // eigenvalue would be 0 and doesn't belong in this list.
      if (si === 0 && sk === 0) continue;
      eigs.push(crossEig(sk, si));
    }
  }

  eigs.sort((a, b) => b - a);
  return eigs;
}

/**
 * Population loss under the analytic deep-linear solution with whitened
 * inputs:
 *     L(t) = (1/2) ‖W(t) − W⋆‖_F² = (1/2) Σ_i (σ_i(t) − σ_i⋆)².
 *
 * Justification: under the analytic assumption the product matrix is
 * W(t) = U Σ(t) V^⊤ and W⋆ = U Σ⋆ V^⊤ (same SVD basis), so
 * W(t) − W⋆ = U (Σ(t) − Σ⋆) V^⊤ has singular values |σ_i(t) − σ_i⋆| and the
 * Frobenius norm collapses to the diagonal sum. With Σ_x = I (whitened
 * inputs) the population MSE equals (1/2) ‖W − W⋆‖_F² exactly.
 *
 * Modes with σ_i⋆ = 0 contribute (1/2)σ_i(t)² — fine; those are the
 * low-rank-M directions, and the ODE drives them toward 0.
 *
 * Under iid Gaussian inputs the empirical loss differs from this by
 * O(1/√N) finite-sample noise; under non-aligned init the analytic
 * assumption is violated and the dashed line diverges from the measured
 * curve early on. Both gaps are pedagogically informative.
 *
 * @param {number[]} sigmas
 * @param {number[]} sigmaStar
 * @returns {number}
 */
export function predictedLoss(sigmas, sigmaStar) {
  let s = 0;
  const n = sigmas.length;
  for (let i = 0; i < n; i++) {
    const d = sigmas[i] - (sigmaStar[i] || 0);
    s += d * d;
  }
  return 0.5 * s;
}

/**
 * Stateful wrapper used by Simulation: holds the current σ vector and
 * advances it one RK4 step per call, returning the predicted eigenvalues
 * (sorted descending) after the step.
 *
 * Usage:
 *   const pred = new SaxePredictor({ sigmaStar, L, epsilon });
 *   // each training step:
 *   const topEigs = pred.step(eta);   // advances σ by dt = eta
 */
export class SaxePredictor {
  /**
   * @param {object} opts
   * @param {number[]} opts.sigmaStar  — target singular values, length r =
   *                                     min(inputDim, outputDim). Pad with
   *                                     zeros for modes above rank(M).
   * @param {number}   opts.L
   * @param {number}   opts.epsilon
   */
  constructor({ sigmaStar, L, epsilon }) {
    this.sigmaStar = sigmaStar.slice();          // defensive copy
    this.L = L;
    this.epsilon = epsilon;
    this.sigmas = initialSingularValues(epsilon, L, sigmaStar.length);
  }

  /** Eigenvalues at the current σ, without advancing. */
  currentEigenvalues() {
    return predictedEigenvalues(this.sigmas, this.sigmaStar, this.L);
  }

  /** Predicted loss at the current σ, without advancing. */
  currentLoss() {
    return predictedLoss(this.sigmas, this.sigmaStar);
  }

  /**
   * Advance σ by one RK4 step of size dt and return the new eigenvalues.
   * @param {number} dt   typically = η so theory time tracks η·step
   */
  step(dt) {
    this.sigmas = stepSaxeODE(this.sigmas, this.sigmaStar, this.L, dt);
    return this.currentEigenvalues();
  }

  /** Snapshot of the current singular-value vector (for diagnostics). */
  currentSigmas() {
    return this.sigmas.slice();
  }
}
