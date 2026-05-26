// ============================================================================
// THEORY - Saxe analytic prediction for deep linear networks
// ============================================================================
// Given the user's target singular values {σ_i⋆}, depth L, init scale ε, and
// learning rate η, integrate the Saxe gradient-flow ODE
//
//     dσ_i/dt = L · σ_i^(2 − 2/L) · (σ_i⋆ − σ_i)
//
// alongside training, with the standard correspondence t ↔ η · step.
//
// Architecture (σ-only): the simulation stores ONLY the singular-value
// trajectory σ(t), advanced by stepSaxeODE. It does not compute or store
// eigenvalues. At plot time the visualization layer calls the stateless
// grouped functions at the bottom of this file — theory_GN and
// theory_Hessian_full — on a stored σ vector to derive the eigenvalue curves
// it draws. predictedLoss is also computed from σ (the loss overlay). See the
// "GROUPED THEORY" section below for the two entry points and their formulas.
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
//   theory_Hessian_full(sigmas, sigmaStar, n, d, m)
//       → { aligned, aligned_null, cross, single_value, hidden_null, idle_null }
//     Full Hessian (H_GN + residual R) eigenvalues.
//       aligned       value = 3 s_i − s_i0
//       aligned_null  value = s_i0 − s_i      (null partner of aligned; gn=0)
//       cross         closed form, computeCrossEigenvalues (full branch values)
//       single_value  value = s_i             (residual trivial — identical to GN)
//       hidden_null   value = ±(s_i − s_i0)   (hidden-null NON-IDLE modes; gn=0)
//       idle_null     value = 0               (hidden-null IDLE modes; gn=0)
//
// single_value is shared and numerically identical between the two functions.
// aligned_null, hidden_null, and idle_null are residual/full constructs only
// (no GN counterpart). The two hidden-null families come from the surplus
// (m − r) hidden units once the network is overparameterized:
//   • hidden_null (non-idle): the residual lifts these off zero into a
//     +(s_i−s_i0) / −(s_i−s_i0) pair, each branch with multiplicity (m − r), so
//     the total is 2 r (m − r). At t = 0 they sit near ±s_i0 and dominate the
//     early sharpness — the GN term cannot see them.
//   • idle_null: the surplus hidden units paired with idle (zero-singular-value)
//     input/output directions. They stay at exactly 0 for all t, count
//     (m − r)(n + d − 2r). They carry no dynamics but ARE part of the spectrum,
//     so they are emitted (and plottable) for exact agreement with a full dense
//     diagonalization. Both families vanish when m = r (no overparameterization).
//
// The six full-Hessian groups together hold the COMPLETE spectrum: their record
// counts sum to P = m(n+d), the parameter count. theorySpectrumFull() returns
// that complete length-P vector, sorted, for element-wise numerical comparison.
//
// Record schema: { value, indices, [branch] }. `indices` is a 2-tuple; a real
// mode index where a singular value exists, null for a zero-mode partner slot.
// `branch` (1..4) appears on cross records for grouping only — see the note in
// computeCrossEigenvalues about per-branch vs set-level behavior.
//
// Counts (per call, full theory): aligned r, aligned_null r, cross 2·r(r−1)
// [one merged group = the 4 roots of each unordered pair's 4×4 block; this is
// the cross-NTK + cross-null pair from the derivation combined], single_value
// r·(n−r)+r·(d−r), hidden_null 2·r·(m−r), idle_null (m−r)·(n+d−2r). These sum to
// P = m(n+d). r = sigmas.length. (GN theory: aligned r, cross, single_value.)

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

  // λ_{1,2} = ½[ 2(s_i+s_k) − (s_i0+s_k0) ± √( 4 s_i s_k + (s_i0−s_k0)² ) ]
  const A   = 2 * sigma - (s_i0 + s_k0);
  const D12 = Math.sqrt(4 * s_i * s_k + (s_i0 - s_k0) ** 2);

  // λ_{3,4} = ½[ (s_i0+s_k0) ± √( (2(s_i−s_k) − (s_i0−s_k0))² + 4 s_i s_k ) ]
  const B   = s_i0 + s_k0;
  const D34 = Math.sqrt((2 * (s_i - s_k) - (s_i0 - s_k0)) ** 2 + 4 * s_i * s_k);

  // GN branch values (value_gn) are an independent fixed labeling and are
  // unchanged: the {1,2} block carries gn = σ, the {3,4} block carries gn = 0.
  // As before, the (full,gn) pairing is not invariant in the s→s0 limit — the
  // full set is {σ,0,σ,0} while the GN set is {σ,σ,0,0} (equal as SETS only).
  // Downstream plotting sorts within a group per timepoint, so the per-branch
  // pairing is for grouping bookkeeping, not a tracked correspondence.
  return [
    { value_full: 0.5 * (A + D12), value_gn: sigma, branch: 1 }, // λ_1
    { value_full: 0.5 * (A - D12), value_gn: sigma, branch: 2 }, // λ_2
    { value_full: 0.5 * (B + D34), value_gn: 0,     branch: 3 }, // λ_3
    { value_full: 0.5 * (B - D34), value_gn: 0,     branch: 4 }  // λ_4
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
 * This is the depth-general GN formula; multiplicities for L > 2 are not
 * modeled (they don't affect the GN eigenvalue *values*, only how many times
 * each would appear).
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
 * @param {number}   m          hidden width (for the hidden_null / idle_null counts)
 * @returns {{aligned:Array, aligned_null:Array, cross:Array,
 *            single_value:Array, hidden_null:Array, idle_null:Array}}
 *   The six groups together hold the COMPLETE spectrum: their record counts sum
 *   to P = m(n+d), the full parameter count (verify via theorySpectrumFull).
 *   idle_null carries the exactly-zero eigenvalues, count (m−r)(n+d−2r); it is a
 *   listed plotting group (see FULL_GROUPS) so the overlay can show the flat
 *   zero line as its own selectable class for direct comparison with numerics.
 */
export function theory_Hessian_full(sigmas, sigmaStar, n, d, m) {
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

  // Hidden-null (non-idle) modes. Per nonzero mode i the residual lifts the
  // GN-null subspace into a ±(s_i − s_i0) pair, each branch with multiplicity
  // (m − r) (one per surplus hidden unit). Total count 2 r (m − r); empty when
  // m ≤ r. We emit the FULL multiplicity (m − r copies per branch) so per-class
  // top-k and curve counts match the measured/dense spectrum directly. When m is
  // not supplied (legacy callers) we fall back to no surplus units → empty class.
  const hidden_null = [];
  const surplus = (typeof m === 'number' && m > r) ? (m - r) : 0;
  for (let i = 0; i < r; i++) {
    const shift = sigmas[i] - (sigmaStar[i] || 0);   // s_i − s_i0
    for (let c = 0; c < surplus; c++) {
      hidden_null.push({ value:  shift, indices: [i, null], branch: '+' });
      hidden_null.push({ value: -shift, indices: [i, null], branch: '-' });
    }
  }

  // Idle hidden-null modes: the surplus hidden units paired with idle (zero-
  // singular-value) input/output directions. These stay at exactly 0 for all t
  // (no residual lift, no GN value). Count (m − r)(n + d − 2r) — the modes that
  // make the grouped totals sum to P = m(n+d). Listed in FULL_GROUPS so the
  // overlay can plot the flat zero line as its own selectable class, and so a
  // flattened theory spectrum reconciles exactly with a dense diagonalization.
  const idle_null = [];
  const idleCount = surplus * Math.max(n - r, 0) + surplus * Math.max(d - r, 0);
  for (let c = 0; c < idleCount; c++) {
    idle_null.push({ value: 0, indices: [null, null] });
  }

  return { aligned, aligned_null, cross, single_value, hidden_null, idle_null };
}

/**
 * Complete flat L = 2 full-Hessian spectrum: all P = m(n+d) eigenvalues for one
 * σ-vector, sorted ASCENDING (the dense/Lanczos convention). This is the vector
 * to diff element-for-element against a dense numerical diagonalization.
 *
 * It flattens every group of theory_Hessian_full (all six, idle_null included),
 * so the length is exactly P regardless of which groups the plotting layer
 * happens to display.
 *
 * @returns {number[]} length m(n+d), ascending.
 */
export function theorySpectrumFull(sigmas, sigmaStar, n, d, m) {
  const g = theory_Hessian_full(sigmas, sigmaStar, n, d, m);
  const out = [];
  for (const key of Object.keys(g)) {
    for (const rec of g[key]) out.push(rec.value);
  }
  out.sort((a, b) => a - b);
  return out;
}
