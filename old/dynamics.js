// ============================================================================
// DYNAMICS - Saxe analytic dynamics for 2-layer deep linear networks.
// ============================================================================
// For a 2-layer linear network trained by gradient flow on whitened data, each
// singular value of the model evolves independently according to
//
//     dσ/dt = 2 σ (s − σ),
//
// where s is the corresponding target singular value. The closed-form solution
// (with initial condition σ(0) = σ₀ > 0) is
//
//                    s
//     σ(t) = ───────────────────────
//            1 + (s/σ₀ − 1) e^{−2 s t}
//
// We use this directly — no ODE integration required.
//
// The transition time t* — when σ(t) is roughly halfway to s — is
// t* = ln(s / σ₀) / (2 s). This sets the time scale at which each mode is
// "learned"; the slowest mode (smallest s > 0) determines when training is
// effectively complete.

/**
 * Evaluate σ(t) for a single mode.
 *
 * @param {number} s         Target singular value (≥ 0).
 * @param {number} sigma0    Initial singular value at t = 0 (> 0).
 * @param {number} t         Time (≥ 0).
 * @returns {number}
 */
export function sigmaAtTime(s, sigma0, t) {
  // s = 0: target is zero, σ(t) stays at σ₀ forever (gradient is zero).
  if (s <= 0) return sigma0;
  // sigma0 = 0: degenerate fixed point, can never escape — guard caller-side,
  // but be safe here too.
  if (sigma0 <= 0) return 0;
  // Avoid catastrophic cancellation in the denominator when s ≈ σ₀.
  const ratio = s / sigma0;
  const denom = 1 + (ratio - 1) * Math.exp(-2 * s * t);
  return s / denom;
}

/**
 * Approximate transition time for a mode: the time at which σ(t) is roughly
 * halfway to s. With t* = ln(s/σ₀) / (2s), one can check σ(t*) = s / (1 + 1) =
 * s/2 when σ₀ ≪ s — i.e. t* is the half-completion time. It tracks where the
 * staircase step for this mode happens.
 *
 * @param {number} s
 * @param {number} sigma0
 * @returns {number}  t* ≥ 0. Returns +∞ for s = 0 (mode never transitions).
 */
export function transitionTime(s, sigma0) {
  if (s <= 0) return Infinity;
  if (sigma0 <= 0 || s <= sigma0) return 0;
  return Math.log(s / sigma0) / (2 * s);
}

/**
 * Choose a sensible upper bound for the slider's time range. We base it on the
 * slowest mode's transition time, with a 2× margin so the user can scrub
 * past "training is done" without too much trailing dead space. Returns 0 if
 * all targets are zero (degenerate case — the caller should handle).
 *
 * @param {number[]} targetSigmas
 * @param {number}   sigma0
 * @returns {number}
 */
export function chooseTimeMax(targetSigmas, sigma0) {
  let tStarMax = 0;
  for (const s of targetSigmas) {
    if (s <= 0) continue;
    const t = transitionTime(s, sigma0);
    if (isFinite(t) && t > tStarMax) tStarMax = t;
  }
  return 2.0 * tStarMax;
}
