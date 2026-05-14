// ============================================================================
// MODELS - Sources of (W₁, W₂, ...) for the NTK widgets
// ============================================================================
// All sources return objects shaped like MLP from model.js: they expose
// `W` (array of 2D weight matrices), `layerSizes`, `numLayers`, and
// `forward(x)` returning { output, activations }. ntk.js consumes this
// interface and doesn't care where the model came from.

import { MLP } from './model.js';
import { mulberry32 } from './rng.js';
import { randomOrthogonal } from './matrix.js';

/**
 * Random-initialized 2-layer linear MLP. Weights are drawn from the muP
 * normal init in model.js. By design we keep initScale = 1 here — no small-
 * init regime; the goal of widget 1 is to display a *generic* non-aligned
 * NTK for a freshly initialized network.
 *
 * @param {object} opts
 * @param {number} opts.inputDim
 * @param {number} opts.hiddenDim
 * @param {number} opts.outputDim
 * @param {number} opts.seed
 * @returns {MLP}
 */
export function randomModel({ inputDim, hiddenDim, outputDim, seed }) {
  return new MLP([inputDim, hiddenDim, outputDim], seed, 1.0);
}

// ---------------------------------------------------------------------------
// Analytic snapshot model
// ---------------------------------------------------------------------------
// Implements the Saxe-style decomposition with O = I:
//
//     W₁ = Σ^{1/2} V^T,   W₂ = U Σ^{1/2}
//
// where Σ is diagonal with entries σᵢ(t). Hidden dim equals output dim by
// convention. The factor of √σᵢ on each side gives W₂ W₁ = U Σ V^T = M(t).
//
// For the widget: pass in U, V (the SVD basis of the *target* M, fixed in
// time) and a list of singular values for the current snapshot. The shape
// of the returned object matches MLP, so it drops into ntk.js unchanged.
//
// @param {object} opts
// @param {number[][]} opts.U        outputDim × outputDim
// @param {number[][]} opts.V        inputDim × inputDim
// @param {number[]}   opts.sigmas   length = min(outputDim, inputDim)
// @returns plain object with the MLP interface
export function analyticModel({ U, V, sigmas }) {
  const outputDim = U.length;
  const inputDim = V.length;
  const r = sigmas.length;

  // Hidden dim = output dim (i.e., we represent Σ^{1/2} as an outputDim ×
  // inputDim matrix with √σᵢ on the diagonal, padded with zeros).
  const hiddenDim = outputDim;

  // W₁ has shape hiddenDim × inputDim. (Σ^{1/2} V^T)_{ij} = √σ_i · V[j][i].
  const W1 = new Array(hiddenDim);
  for (let i = 0; i < hiddenDim; i++) {
    const row = new Array(inputDim).fill(0);
    if (i < r) {
      const root = Math.sqrt(Math.max(sigmas[i], 0));
      for (let j = 0; j < inputDim; j++) row[j] = root * V[j][i];
    }
    W1[i] = row;
  }

  // W₂ has shape outputDim × hiddenDim. (U Σ^{1/2})_{ij} = U[i][j] · √σ_j.
  const W2 = new Array(outputDim);
  for (let i = 0; i < outputDim; i++) {
    const row = new Array(hiddenDim).fill(0);
    for (let j = 0; j < hiddenDim; j++) {
      const root = j < r ? Math.sqrt(Math.max(sigmas[j], 0)) : 0;
      row[j] = U[i][j] * root;
    }
    W2[i] = row;
  }

  const layerSizes = [inputDim, hiddenDim, outputDim];

  return {
    W: [W1, W2],
    layerSizes,
    numLayers: 2,

    forward(x) {
      let a = x;
      const activations = [a];
      for (let l = 0; l < 2; l++) {
        const W = this.W[l];
        const z = new Array(W.length);
        for (let i = 0; i < W.length; i++) {
          let s = 0;
          for (let k = 0; k < W[i].length; k++) s += W[i][k] * a[k];
          z[i] = s;
        }
        a = z;
        activations.push(a);
      }
      return { output: a.length === 1 ? a[0] : a, activations };
    }
  };
}

/**
 * Generate a random orthogonal pair (U, V) for a target's SVD basis. The
 * widgets pair these with caller-specified singular values via analyticModel.
 *
 * @param {object} opts
 * @param {number} opts.inputDim
 * @param {number} opts.outputDim
 * @param {number} opts.seed
 * @returns {{ U: number[][], V: number[][] }}
 */
export function randomBasis({ inputDim, outputDim, seed }) {
  const rng = mulberry32(seed);
  return {
    U: randomOrthogonal(outputDim, rng),
    V: randomOrthogonal(inputDim, rng)
  };
}
