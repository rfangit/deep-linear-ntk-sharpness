// ============================================================================
// INPUTS - Constructing the design matrix X
// ============================================================================
// Two modes:
//
//   'gaussian'  — iid samples x_i ~ N(0, I_k). The empirical covariance
//                  (1/N) X^T X is only approximately I; for finite N the
//                  off-diagonals are O(1/√N), the diagonals are 1 ± O(1/√N).
//
//   'whitened'  — exact whitening by construction:
//                  X = √N · Q,  where Q is an N × k matrix with Q^T Q = I_k.
//                  Then (1/N) X^T X = I_k as an exact algebraic identity for
//                  any finite N. Cost: one Gram-Schmidt over k Gaussian
//                  N-vectors. Requires N ≥ k.
//
// The deep-linear theory (Saxe et al. and friends) routinely simplifies
// expressions using Σ_x = I; whitened mode makes those simplifications
// hold *exactly* in the simulation, not just up to finite-sample noise.
//
// Returns X as number[][], shape [nTrain][inputDim], same convention used
// everywhere else in the codebase (each x_i is a row).

import { mulberry32, seededRandn } from './rng.js';

/**
 * Sample X iid from N(0, I_k). Each row x_i is k-dimensional.
 */
function sampleGaussian(inputDim, nTrain, rng) {
  const X = new Array(nTrain);
  for (let i = 0; i < nTrain; i++) {
    const xi = new Array(inputDim);
    for (let j = 0; j < inputDim; j++) xi[j] = seededRandn(rng);
    X[i] = xi;
  }
  return X;
}

/**
 * Construct X with (1/N) X^T X = I_k exactly. Strategy: build k orthonormal
 * N-vectors via Gram-Schmidt over Gaussian columns, stack them as columns of
 * an N × k matrix Q, then scale by √N.
 *
 * Throws if nTrain < inputDim — k orthonormal vectors don't exist in fewer
 * than k dimensions.
 */
function constructWhitened(inputDim, nTrain, rng) {
  if (nTrain < inputDim) {
    throw new Error(
      `whitened-input construction requires nTrain ≥ inputDim (got nTrain=${nTrain}, inputDim=${inputDim})`
    );
  }

  // Build k Gaussian N-vectors, then orthonormalize via modified Gram-Schmidt.
  // Each q[c] is the c-th column of Q, length nTrain.
  const q = new Array(inputDim);
  for (let c = 0; c < inputDim; c++) {
    const v = new Array(nTrain);
    for (let i = 0; i < nTrain; i++) v[i] = seededRandn(rng);

    // Subtract projections onto already-orthonormal columns
    for (let p = 0; p < c; p++) {
      const qp = q[p];
      let dot = 0;
      for (let i = 0; i < nTrain; i++) dot += qp[i] * v[i];
      for (let i = 0; i < nTrain; i++) v[i] -= dot * qp[i];
    }

    // Normalize
    let norm = 0;
    for (let i = 0; i < nTrain; i++) norm += v[i] * v[i];
    norm = Math.sqrt(norm);
    if (norm < 1e-12) {
      // Astronomically unlikely with random Gaussians, but be safe: fall back
      // to a canonical basis vector orthogonal to everything so far.
      for (let i = 0; i < nTrain; i++) v[i] = 0;
      v[c] = 1;
      norm = 1;
    }
    for (let i = 0; i < nTrain; i++) v[i] /= norm;
    q[c] = v;
  }

  // Assemble X: row i, column j is √N · q[j][i].
  const scale = Math.sqrt(nTrain);
  const X = new Array(nTrain);
  for (let i = 0; i < nTrain; i++) {
    const xi = new Array(inputDim);
    for (let j = 0; j < inputDim; j++) xi[j] = scale * q[j][i];
    X[i] = xi;
  }
  return X;
}

/**
 * Generate the input design matrix X.
 *
 * @param {object} opts
 * @param {number} opts.inputDim
 * @param {number} opts.nTrain
 * @param {number} opts.dataSeed
 * @param {'gaussian'|'whitened'} opts.mode
 * @returns {number[][]} X with shape [nTrain][inputDim]
 */
export function generateInputs({ inputDim, nTrain, dataSeed, mode }) {
  const rng = mulberry32(dataSeed);
  if (mode === 'whitened') return constructWhitened(inputDim, nTrain, rng);
  if (mode === 'gaussian') return sampleGaussian(inputDim, nTrain, rng);
  throw new Error(`generateInputs: unknown mode "${mode}"`);
}
