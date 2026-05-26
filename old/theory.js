// ============================================================================
// THEORY - Saxe analytic prediction for deep linear networks
// ============================================================================
// Given the user's target singular values {σ_i⋆}, depth L, init scale ε, and
// learning rate η, integrate the Saxe gradient-flow ODE
//
//     dσ_i/dt = L · σ_i^(2 − 2/L) · (σ_i⋆ − σ_i)
//
// alongside training, with the standard correspondence t ↔ η · step. Use the
// integrated singular values to compute Hessian eigenvalue predictions.
//
// Two prediction flavors are exposed:
//
//   1. Gauss-Newton only — predictedEigenvalues / SaxePredictor. Uses
//      H_GN = J^T J and yields the formulas from the blog post:
//        Aligned modes (k = i, i ≤ r):     L · s_i^(2(L−1)/L)
//        Cross modes   (k ≠ i, both ≤ r):  Σ_{ℓ=1..L} s_k^(2(ℓ−1)/L) · s_i^(2(L−ℓ)/L)
//        Low-rank-M    (one of σ⋆ is 0):   cross formula with that s = 0
//
//   2. Gauss-Newton + Residual — predictedResidualEigenvalues /
//      SaxeResidualPredictor. Adds the full Hessian residual term R, derived
//      for the 2-layer (L = 2) case:
//        Cross modes   (i < j):  (s_i + s_j) ± ρ_ij,
//                                ρ_ij = (2 s_i s_j − s_i s_j⋆ − s_j s_i⋆)
//                                        / (s_i + s_j)
//      The aligned-mode contribution is currently under revision and is
//      intentionally omitted; the predictor returns only the cross-mode
//      eigenvalues sorted descending so the toggle can be used to sanity-
//      check the cross-mode formula in isolation against measurements.
//      Both branches λ_± are reported; negative branches are kept as-is.
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


// ============================================================================
// RESIDUAL-CORRECTED PREDICTION (H_GN + R)
// ============================================================================
// Same singular-value dynamics as the GN-only predictor (same ODE, same RK4,
// same ε^L initial conditions) — only the eigenvalue formula differs. The
// residual derivation in the writeup is L = 2 specific; we apply it for any
// L and leave the UX to flag the mismatch.
//
// Aligned-mode eigenvalues carry multiplicity m (the hidden width); we
// expand the multiplicities here so the descending-sorted top-k matches what
// the empirical Lanczos spectrum reports.

const RESIDUAL_NUMERICAL_EPS = 1e-10;

/**
 * Compute Hessian eigenvalues from the residual term R using only the
 * cross-mode contribution. The aligned-mode formula is currently under
 * revision and intentionally omitted; once it's resolved it will be added
 * back here.
 *
 * Cross modes (i < j), both branches of the 2D {h_ij, h_ji} block:
 *     λ_± = (s_i + s_j) ± ρ_ij
 *     ρ_ij = (2 s_i s_j − s_i s_j⋆ − s_j s_i⋆) / (s_i + s_j)
 *
 * Modes with s_i + s_j ≈ 0 (e.g. at initialization with very small ε)
 * collapse to ρ = 0 and λ_± = 0, matching the degenerate limit in the
 * pseudocode.
 *
 * Returns the full list sorted descending. Negative branches are kept as-is;
 * the caller slices the top k.
 *
 * @param {number[]} sigmas      — current σ_i, length r
 * @param {number[]} sigmaStar   — target σ_i⋆, same length
 * @param {number}   m           — aligned-mode multiplicity (= hidden width).
 *                                 Currently unused since aligned modes are
 *                                 omitted; kept in the signature so callers
 *                                 don't need to change when the formula is
 *                                 reinstated.
 * @returns {number[]} sorted descending
 */
export function predictedResidualEigenvalues(sigmas, sigmaStar, m) {
  const r = sigmas.length;
  const eigs = [];

  // Cross modes — both ± branches per unordered pair.
  for (let i = 0; i < r; i++) {
    for (let j = i + 1; j < r; j++) {
      const s_i  = sigmas[i];
      const s_j  = sigmas[j];
      const s_i0 = sigmaStar[i] || 0;
      const s_j0 = sigmaStar[j] || 0;
      const gn   = s_i + s_j;
      let rho;
      if (gn > RESIDUAL_NUMERICAL_EPS) {
        rho = (2 * s_i * s_j - s_i * s_j0 - s_j * s_i0) / gn;
      } else {
        rho = 0;
      }
      eigs.push(gn + rho);
      eigs.push(gn - rho);
    }
  }

  eigs.sort((a, b) => b - a);
  return eigs;
}

/**
 * Stateful predictor for the residual-corrected (H_GN + R) eigenvalues.
 * Mirrors SaxePredictor exactly in lifecycle, but reports the residual
 * formula instead of the GN-only formula. The dynamics are identical: same
 * stepSaxeODE call with the same dt = η, so a paired (SaxePredictor,
 * SaxeResidualPredictor) advance in lock-step.
 *
 * Loss prediction is left unchanged from SaxePredictor — the residual
 * correction enters the Hessian curvature, not the loss value.
 *
 * Usage:
 *   const res = new SaxeResidualPredictor({ sigmaStar, L, epsilon, hiddenWidth });
 *   res.step(eta);
 */
export class SaxeResidualPredictor {
  /**
   * @param {object} opts
   * @param {number[]} opts.sigmaStar    — target σ⋆, length r
   * @param {number}   opts.L
   * @param {number}   opts.epsilon
   * @param {number}   opts.hiddenWidth  — m (aligned-mode multiplicity)
   */
  constructor({ sigmaStar, L, epsilon, hiddenWidth }) {
    this.sigmaStar   = sigmaStar.slice();
    this.L           = L;
    this.epsilon     = epsilon;
    this.hiddenWidth = hiddenWidth;
    this.sigmas      = initialSingularValues(epsilon, L, sigmaStar.length);
  }

  currentEigenvalues() {
    return predictedResidualEigenvalues(this.sigmas, this.sigmaStar, this.hiddenWidth);
  }

  currentLoss() {
    return predictedLoss(this.sigmas, this.sigmaStar);
  }

  step(dt) {
    this.sigmas = stepSaxeODE(this.sigmas, this.sigmaStar, this.L, dt);
    return this.currentEigenvalues();
  }

  currentSigmas() {
    return this.sigmas.slice();
  }
}


// ============================================================================
// GROUPED THEORY (2-LAYER) — stateless eigenvalue functions over σ
// ============================================================================
// Architecture: the simulation stores ONLY the singular-value trajectory σ(t).
// At plot time the visualization layer calls these stateless functions on a
// stored σ vector, choosing GN vs full (and which groups to draw) via user
// knobs. Nothing here holds state or advances dynamics — that's stepSaxeODE.
//
// Strictly L = 2 (single hidden layer): the closed forms are 2-layer-specific.
//
// Two entry points:
//
//   theory_GN(sigmas, sigmaStar)
//       → { aligned, cross, single_value }
//     Gauss-Newton (H_GN = JᵀJ) eigenvalues.
//       aligned       value = 2 s_i                         (one per nonzero mode)
//       cross         GN branches per unordered pair {i,k}: σ=s_i+s_k (×2), 0 (×2)
//       single_value  value = s_i                           (nonzero mode ⊗ zero mode)
//
//   theory_Hessian_full(sigmas, sigmaStar, n, d)
//       → { aligned, aligned_null, cross, single_value }
//     Full Hessian (H_GN + residual R) eigenvalues.
//       aligned       value = 3 s_i − s_i0
//       aligned_null  value = s_i0 − s_i      (null partner of aligned; gn=0)
//       cross         closed form, computeCrossEigenvalues (full branch values)
//       single_value  value = s_i             (residual trivial — identical to GN)
//
// single_value is shared and numerically identical between the two functions.
// aligned_null is a residual/full construct only (it has no GN counterpart).
// hidden_null (always 0) is intentionally omitted from both.
//
// Record schema: { value, indices, [branch] }. `indices` is a 2-tuple; a real
// mode index where a singular value exists, null for a zero-mode partner slot.
// `branch` (1..4) appears on cross records for grouping only — see the note in
// computeCrossEigenvalues about per-branch vs set-level behavior.
//
// Counts (per call): aligned r, aligned_null r, cross 2·r(r−1) [full] or the
// same shape for GN, single_value r·(n−r)+r·(d−r). r = sigmas.length.

/**
 * Closed-form eigenvalues of the 2-layer Hessian for one unordered nonzero
 * pair {i,k}. Returns 4 records, each with BOTH the full and GN value (the
 * caller keeps whichever it needs). Inputs guaranteed mid-training (σ>0).
 *
 * As verified previously, the per-branch (full,gn) pairing is not invariant in
 * the s→s0 limit: at s==s0 the full set is {σ,0,σ,0} and the GN set is
 * {σ,σ,0,0} — equal as SETS but not branch-by-branch. Downstream plotting sorts
 * within a group per timepoint rather than tracking a fixed branch.
 *
 * @returns {Array<{value_full:number, value_gn:number, branch:number}>}
 */
export function computeCrossEigenvalues(s_i, s_k, s_i0, s_k0) {
  const sigma = s_i + s_k;
  const delta = (s_i - s_i0) - (s_k - s_k0);
  const crossTerm = 4 * s_i * s_k * delta * delta / (sigma * sigma);

  const D1 = Math.sqrt((s_i0 + s_k0) ** 2 + crossTerm);
  const D2 = Math.sqrt((2 * sigma - s_i0 - s_k0) ** 2 + crossTerm);
  const cp = sigma + (s_i - s_k) * delta / sigma;
  const cm = sigma - (s_i - s_k) * delta / sigma;

  return [
    { value_full: 0.5 * (cp + D1), value_gn: sigma, branch: 1 }, // h_ik
    { value_full: 0.5 * (cp - D1), value_gn: sigma, branch: 2 }, // h_ki
    { value_full: 0.5 * (cm + D2), value_gn: 0,     branch: 3 }, // g_ik
    { value_full: 0.5 * (cm - D2), value_gn: 0,     branch: 4 }  // g_ki
  ];
}

/** Aligned mode indices/values are shared structure; tiny helpers keep the two
 *  entry points readable and guarantee single_value matches between them. */
function buildSingleValue(sigmas, n, d, r) {
  const out = [];
  for (let i = 0; i < r; i++) {
    for (let t = 0; t < n - r; t++) out.push({ value: sigmas[i], indices: [i, null] });
    for (let t = 0; t < d - r; t++) out.push({ value: sigmas[i], indices: [null, i] });
  }
  return out;
}

/**
 * General-L Gauss-Newton cross/aligned eigenvalue for an ordered mode pair
 * (k, i):  c_{ki} = Σ_{ℓ=1..L} s_k^(2(ℓ−1)/L) · s_i^(2(L−ℓ)/L).
 * On the diagonal (k = i) this collapses to L · s_i^(2(L−1)/L) (the aligned
 * value); off-diagonal it is the cross value. At L = 2 it gives s_i + s_k.
 * This is the depth-general GN formula carried over from the original
 * predictedEigenvalues; multiplicities for L > 2 are not modeled (they don't
 * affect the GN eigenvalue *values*, only how many times each would appear).
 */
function gnCrossEig(sk, si, L) {
  let sum = 0;
  for (let ell = 1; ell <= L; ell++) {
    // pow(0,0) === 1 in JS, which is the intended limit for degenerate terms.
    sum += Math.pow(sk, (ell - 1) * 2 / L) * Math.pow(si, (L - ell) * 2 / L);
  }
  return sum;
}

/**
 * Gauss-Newton eigenvalues, grouped. Stateless. Works for any depth L.
 *
 * Grouping (identical structure at all L):
 *   aligned       diagonal pair (k = i): value = L · s_i^(2(L−1)/L)
 *                 (= 2 s_i at L = 2). One per nonzero mode.
 *   cross         off-diagonal ordered pairs (k ≠ i): value = c_{ki}
 *                 (= s_i + s_k at L = 2).
 *   single_value  one per nonzero mode: value = s_i (depth-independent; the GN
 *                 single-value eigenvalue is the lone singular value).
 *
 * Two code paths, kept separate by design:
 *   L === 2 → routes cross through computeCrossEigenvalues (the 2-layer closed
 *             form), preserving the validated 4-branch structure and the
 *             index.html top-k parity. aligned uses 2 s_i directly.
 *   L !== 2 → uses the depth-general c_{ki} enumeration: diagonal → aligned,
 *             each off-diagonal ORDERED pair → one cross record. (No zero
 *             branches — those are specific to the 2-layer closed form.)
 *
 * @param {number[]} sigmas     current singular values, length r
 * @param {number[]} sigmaStar  target singular values, length r (for cross, L=2)
 * @param {number}   n          output dim (for single_value count)
 * @param {number}   d          input dim (for single_value count)
 * @param {number}   L          number of weight matrices (depth). Default 2.
 * @returns {{aligned:Array, cross:Array, single_value:Array}}
 */
export function theory_GN(sigmas, sigmaStar, n, d, L = 2) {
  const r = sigmas.length;
  const aligned = [];
  const cross = [];

  if (L === 2) {
    // 2-layer closed-form path (unchanged — preserves parity + zero branches).
    for (let i = 0; i < r; i++) {
      aligned.push({ value: 2 * sigmas[i], indices: [i, i] });
    }
    for (let i = 0; i < r; i++) {
      for (let k = i + 1; k < r; k++) {
        for (const e of computeCrossEigenvalues(sigmas[i], sigmas[k], sigmaStar[i] || 0, sigmaStar[k] || 0)) {
          cross.push({ value: e.value_gn, indices: [i, k], branch: e.branch });
        }
      }
    }
  } else {
    // Depth-general path: enumerate ordered pairs via c_{ki}.
    for (let i = 0; i < r; i++) {
      aligned.push({ value: gnCrossEig(sigmas[i], sigmas[i], L), indices: [i, i] });
    }
    for (let i = 0; i < r; i++) {
      for (let k = 0; k < r; k++) {
        if (k === i) continue;                       // diagonal handled as aligned
        cross.push({ value: gnCrossEig(sigmas[k], sigmas[i], L), indices: [i, k] });
      }
    }
  }

  const single_value = buildSingleValue(sigmas, n, d, r);
  return { aligned, cross, single_value };
}

/**
 * Full-Hessian (GN + residual) eigenvalues, grouped. Stateless.
 * STRICTLY L = 2 — the residual closed form is 2-layer-specific. Callers must
 * not invoke this for L ≠ 2 (the plotting layer gates on depth).
 * @param {number[]} sigmas     current singular values, length r
 * @param {number[]} sigmaStar  target singular values, length r
 * @param {number}   n          output dim
 * @param {number}   d          input dim
 * @returns {{aligned:Array, aligned_null:Array, cross:Array, single_value:Array}}
 */
export function theory_Hessian_full(sigmas, sigmaStar, n, d) {
  const r = sigmas.length;

  const aligned = [];
  const aligned_null = [];
  for (let i = 0; i < r; i++) {
    const s = sigmas[i], s0 = sigmaStar[i] || 0;
    aligned.push({ value: 3 * s - s0, indices: [i, i] });
    aligned_null.push({ value: s0 - s, indices: [i, i] });
  }

  const cross = [];
  for (let i = 0; i < r; i++) {
    for (let k = i + 1; k < r; k++) {
      for (const e of computeCrossEigenvalues(sigmas[i], sigmas[k], sigmaStar[i] || 0, sigmaStar[k] || 0)) {
        cross.push({ value: e.value_full, indices: [i, k], branch: e.branch });
      }
    }
  }

  const single_value = buildSingleValue(sigmas, n, d, r);

  return { aligned, aligned_null, cross, single_value };
}
