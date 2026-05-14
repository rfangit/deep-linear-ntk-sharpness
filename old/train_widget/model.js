// ============================================================================
// MODEL - Deep linear MLP
// ============================================================================
// Stack of linear layers, no biases, no nonlinearities.
//   f(x) = W_L · W_{L-1} · … · W_1 · x
//
// Two initialization paths:
//
//   1. muP random init (default; alignedInitOpts === null)
//      W_ij ~ N(0, initScale² / fan_in)
//
//   2. Aligned init (alignedInitOpts !== null)
//      Each weight matrix W_ℓ has per-mode singular value ε, where ε is the
//      same `initScale` that the random path uses. The product
//      W_L · … · W_1 then equals U · Σ_{ε^L} · V^T at t=0, where
//      Σ_{ε^L} = ε^L · I_{r} (truncated/padded to the appropriate rectangle).
//
//      This convention is chosen so that ε has matching scale across both
//      init modes: muP-Gaussian's typical per-layer singular value is also
//      ≈ ε, so flipping the toggle at fixed ε keeps the product-matrix scale
//      roughly the same (ε^L either way). This makes the staircase-vs-
//      plateau comparison meaningful at the same ε.
//
//      Construction. With L weight matrices there are L-1 "junctions"
//      between adjacent hidden layers. Each junction has an associated
//      square orthogonal O_ℓ that is shared between the layers it joins:
//
//        W_1 = O_1 D V^T                (shape h_1 × k)
//        W_ℓ = O_ℓ D O_{ℓ-1}^T          (shape h_ℓ × h_{ℓ-1}, internal)
//        W_L = U   D O_{L-1}^T          (shape m × h_{L-1})
//
//      where D is the rectangular "ε on the diagonal" matrix sized to the
//      respective layer. Adjacent O's cancel pairwise in the product:
//          W_L · … · W_1 = U · D^L · V^T = U · (ε^L I) · V^T.
//
//      randomO=false ⇒ every O_ℓ = I, so each W_ℓ is just the diagonal
//      embedding times U or V at the boundaries.
//      randomO=true  ⇒ each O_ℓ is a fresh seeded random orthogonal of size
//      h_ℓ × h_ℓ.

import { mulberry32, seededRandn } from './rng.js';
import { randomOrthogonal } from './matrix.js';

export class MLP {
  /**
   * @param {number[]} layerSizes  e.g. [k, h₁, h₂, m]. First = input dim,
   *                                last = output dim, middle = hidden widths.
   * @param {number}   seed        Seed for weight init. null → Math.random.
   * @param {number}   initScale   In the random path: multiplier on the muP
   *                                std (W_ij ~ N(0, ε²/fan_in)). In the
   *                                aligned path: per-layer singular value ε,
   *                                so the product matrix has singular values
   *                                ε^L. The two conventions agree in typical
   *                                magnitude.
   * @param {object|null} alignedInitOpts
   *                  null → use muP random init (legacy behavior).
   *                  Otherwise, an object with:
   *                    U:        number[m][m]  — left  basis of target M
   *                    V:        number[k][k]  — right basis of target M
   *                    randomO:  boolean       — whether the hidden-layer
   *                              orthogonal rotations are random (seeded by
   *                              `seed`) or identity.
   */
  constructor(layerSizes, seed = null, initScale = 1.0, alignedInitOpts = null) {
    this.layerSizes = layerSizes;
    this.numLayers = layerSizes.length - 1;   // number of weight matrices
    this.initScale = initScale;
    this._rng = seed !== null ? mulberry32(seed) : null;

    if (alignedInitOpts) {
      this.W = this._alignedInit(alignedInitOpts);
    } else {
      this.W = [];
      for (let l = 0; l < this.numLayers; l++) {
        const fanIn = layerSizes[l];
        const fanOut = layerSizes[l + 1];
        this.W.push(this._mupNormal(fanOut, fanIn));
      }
    }
  }

  // -------------------------------------------------------------------------
  // muP random init
  // -------------------------------------------------------------------------

  /** muP normal init: W_ij ~ N(0, (initScale)² / fan_in) */
  _mupNormal(rows, cols) {
    const std = this.initScale * Math.sqrt(1.0 / cols);
    const M = new Array(rows);
    for (let i = 0; i < rows; i++) {
      const row = new Array(cols);
      for (let j = 0; j < cols; j++) row[j] = std * this._randn();
      M[i] = row;
    }
    return M;
  }

  /** Standard normal — uses seeded RNG if available, else Math.random. */
  _randn() {
    if (this._rng) return seededRandn(this._rng);
    const u1 = Math.random();
    const u2 = Math.random();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }

  // -------------------------------------------------------------------------
  // Aligned init
  // -------------------------------------------------------------------------

  /**
   * Build the L weight matrices so each layer's singular values are ε (=
   * this.initScale), making the product matrix have singular values ε^L.
   *
   * This convention is chosen to *match* the random-Gaussian path's typical
   * scale: under muP init, W_ij ~ N(0, ε²/n_{ℓ-1}) gives random matrices
   * whose typical singular values sit near ε. Using the same ε for the
   * per-layer aligned svs means switching modes at fixed ε yields
   * approximately the same product-matrix scale (ε^L) — useful for direct
   * comparisons of dynamics.
   *
   * @param {object} opts {U, V, randomO}
   * @returns {number[][][]} weight matrices [W_1, ..., W_L]
   */
  _alignedInit({ U, V, randomO }) {
    const L = this.numLayers;
    const eps = this.initScale;
    // Per-layer per-mode singular value — equal to ε, matching the muP path's
    // typical per-layer scale. The product matrix's singular values are
    // therefore ε^L (vs ε^L under muP-Gaussian, in expectation).
    const dPer = Math.max(eps, 0);

    // Junction orthogonals: there are L-1 of them. junctionO[j] is the square
    // orthogonal living between W_{j+1} (below) and W_{j+2} (above), of size
    // h_{j+1} × h_{j+1} where h_{j+1} = layerSizes[j+1] (a hidden width).
    //
    // Indexing: junctionO[0] sits at width layerSizes[1] = h_1, between W_1
    // and W_2; junctionO[1] sits at width layerSizes[2] = h_2, between W_2
    // and W_3 (only used when L >= 3); and so on.
    const junctionO = [];
    for (let j = 0; j < L - 1; j++) {
      const dim = this.layerSizes[j + 1];
      if (randomO) {
        // Reuse the matrix.js orthogonal sampler. Each call consumes RNG
        // state sequentially so distinct junctions get distinct matrices.
        junctionO.push(randomOrthogonal(dim, this._rng));
      } else {
        junctionO.push(identity(dim));
      }
    }

    // Assemble each layer. Per-layer formula:
    //   W_ℓ (1-indexed) has shape [layerSizes[ℓ], layerSizes[ℓ-1]].
    //   W_ℓ = A · D · B^T  where A, B are square orthogonals and D is the
    //   rectangular matrix with dPer on the leading diagonal.
    //   - For ℓ = 1: A = junctionO[0], B = V                 (B is k × k)
    //   - For ℓ = L: A = U,            B = junctionO[L-2]    (A is m × m)
    //   - For internal ℓ (2..L-1):
    //       A = junctionO[ℓ-1], B = junctionO[ℓ-2]
    //   When L = 1 (no hidden layers — not used in this widget but worth
    //   handling): A = U, B = V.
    const W = [];
    for (let ell = 0; ell < L; ell++) {   // 0-indexed: ell = (ℓ - 1)
      const rows = this.layerSizes[ell + 1];  // d_out
      const cols = this.layerSizes[ell];      // d_in

      let A, B;
      if (L === 1) {
        A = U;
        B = V;
      } else if (ell === 0) {
        // First layer
        A = junctionO[0];
        B = V;
      } else if (ell === L - 1) {
        // Last layer
        A = U;
        B = junctionO[L - 2];
      } else {
        // Internal layer
        A = junctionO[ell];
        B = junctionO[ell - 1];
      }

      // W = A · D · B^T, with D being the rows×cols matrix carrying dPer on
      // the leading diagonal. Folding D into a single nested loop:
      //   W[i][j] = Σ_p A[i][p] · D[p][p] · B[j][p]     (only p<r contributes)
      // where r = min(rows, cols). A is rows×rows, B is cols×cols.
      const r = Math.min(rows, cols);
      const Wl = new Array(rows);
      for (let i = 0; i < rows; i++) {
        const Wli = new Array(cols).fill(0);
        for (let p = 0; p < r; p++) {
          const aip = A[i][p];
          if (aip === 0) continue;
          const scaled = aip * dPer;
          for (let j = 0; j < cols; j++) {
            Wli[j] += scaled * B[j][p];
          }
        }
        Wl[i] = Wli;
      }
      W.push(Wl);
    }
    return W;
  }

  // -------------------------------------------------------------------------
  // Forward pass + parameter count (unchanged)
  // -------------------------------------------------------------------------

  /**
   * Forward pass: a₀ = x; aₗ₊₁ = Wₗ · aₗ.
   *
   * @param {number[]} x  Input vector of length layerSizes[0].
   * @returns {{ output, activations }}
   *   output: scalar if output dim is 1, else array.
   *   activations[l] is the value flowing into layer l. activations[0] = x;
   *   activations[numLayers] = output. The trainer uses these for backprop.
   */
  forward(x) {
    let a = x;
    const activations = [a];

    for (let l = 0; l < this.numLayers; l++) {
      const W = this.W[l];
      const rows = W.length;
      const cols = W[0].length;

      const z = new Array(rows);
      for (let i = 0; i < rows; i++) {
        let sum = 0;
        const Wi = W[i];
        for (let j = 0; j < cols; j++) sum += Wi[j] * a[j];
        z[i] = sum;
      }
      a = z;
      activations.push(a);
    }

    return {
      output: a.length === 1 ? a[0] : a,
      activations: activations
    };
  }

  numParameters() {
    let count = 0;
    for (let l = 0; l < this.numLayers; l++) {
      count += this.W[l].length * this.W[l][0].length;
    }
    return count;
  }
}

// Module-local identity helper. Kept here (and not exported) so model.js stays
// self-contained — matrix.js has its own private identity for component
// construction.
function identity(n) {
  const I = new Array(n);
  for (let i = 0; i < n; i++) {
    const row = new Array(n).fill(0);
    row[i] = 1;
    I[i] = row;
  }
  return I;
}
