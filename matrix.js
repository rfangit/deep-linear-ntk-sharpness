// ============================================================================
// MATRIX - Helpers for constructing the target matrix M
// ============================================================================
// The data layer (data-generator.js) takes M as an explicit matrix. This
// module provides the constructors the UI uses to *build* M from user input.
//
// Currently supported modes:
//   'singular-values' — user types σ_1...σ_m, optionally with random U, V.
//                        randomBasis: true  → M = U Σ V^T   (full random rotation)
//                        randomBasis: false → M = Σ          (M is just diagonal)

import { mulberry32, seededRandn } from './rng.js';

/**
 * Random n×n orthogonal matrix via modified Gram-Schmidt on Gaussian columns.
 * Returns Q with Q^T Q = I (up to numerical precision).
 *
 * Exported so model.js can use the same routine when generating the optional
 * random orthogonal "O" matrices for aligned initialization.
 */
export function randomOrthogonal(n, rng) {
  const A = [];
  for (let i = 0; i < n; i++) {
    A.push([]);
    for (let j = 0; j < n; j++) A[i].push(seededRandn(rng));
  }
  const Q = [];
  for (let i = 0; i < n; i++) Q.push(new Array(n).fill(0));

  for (let j = 0; j < n; j++) {
    const v = new Array(n);
    for (let i = 0; i < n; i++) v[i] = A[i][j];
    for (let l = 0; l < j; l++) {
      let dot = 0;
      for (let i = 0; i < n; i++) dot += Q[i][l] * v[i];
      for (let i = 0; i < n; i++) v[i] -= dot * Q[i][l];
    }
    let norm = 0;
    for (let i = 0; i < n; i++) norm += v[i] * v[i];
    norm = Math.sqrt(norm);
    if (norm < 1e-12) {
      for (let i = 0; i < n; i++) v[i] = (i === j) ? 1 : 0;
      norm = 1;
    }
    for (let i = 0; i < n; i++) Q[i][j] = v[i] / norm;
  }
  return Q;
}

// ---------------------------------------------------------------------------
// COMPONENT-LEVEL CONSTRUCTOR
// ---------------------------------------------------------------------------
// buildMComponentsFromSpec returns M *together with* its SVD components
// {U, sigma, V}, dispatched from a matrixSpec. For aligned-init (see model.js)
// we need U, V and the singular value vector so the model's per-layer weights
// can be placed *in that basis*.
//
// Returns:
//   { M:    number[outputDim][inputDim],
//     U:    number[outputDim][outputDim],     // left orthogonal basis
//     V:    number[inputDim][inputDim],       // right orthogonal basis
//     sigma:number[min(outputDim, inputDim)]  // diagonal entries of Σ }
//
// When randomBasis is false, U and V are returned as identity matrices of the
// appropriate sizes (so M = Σ in the standard basis). Callers that need to
// align the model to "M = Σ" can use these identities directly.
//
// If singularValues has more entries than min(outputDim, inputDim) — which can
// happen when inputDim < outputDim — the extras are silently ignored. The UI
// is expected to surface a warning in that case.

/**
 * Identity matrix of size n×n.
 * @param {number} n
 * @returns {number[][]}
 */
function identityMatrix(n) {
  const I = [];
  for (let i = 0; i < n; i++) {
    const row = new Array(n).fill(0);
    row[i] = 1;
    I.push(row);
  }
  return I;
}

/**
 * Build M and expose its SVD components {U, sigma, V}. See top-of-section
 * comment for shapes and the randomBasis convention.
 */
export function buildMComponentsFromSpec(matrixSpec, inputDim, outputDim) {
  if (matrixSpec.mode !== 'singular-values') {
    throw new Error(`buildMComponentsFromSpec: unknown mode "${matrixSpec.mode}"`);
  }
  const m = outputDim, k = inputDim;
  const r = Math.min(m, k);
  const svsRaw = matrixSpec.singularValues || [];
  const sigma = new Array(r);
  for (let i = 0; i < r; i++) sigma[i] = svsRaw[i] !== undefined ? svsRaw[i] : 0;

  const randomBasis = matrixSpec.randomBasis !== false;
  let U, V;
  if (randomBasis) {
    const rng = mulberry32(matrixSpec.basisSeed || 0);
    U = randomOrthogonal(m, rng);
    V = randomOrthogonal(k, rng);
  } else {
    U = identityMatrix(m);
    V = identityMatrix(k);
  }

  // Assemble M = U Σ V^T. Computed inline rather than calling buildMFromSpec
  // again so we use the *same* U, V we're about to return (otherwise we'd
  // re-seed the RNG and get the same matrices, but doing it once is cheaper
  // and makes the data flow explicit).
  const M = [];
  for (let i = 0; i < m; i++) {
    M[i] = new Array(k);
    for (let j = 0; j < k; j++) {
      let sum = 0;
      for (let l = 0; l < r; l++) sum += U[i][l] * sigma[l] * V[j][l];
      M[i][j] = sum;
    }
  }

  return { M, U, sigma, V };
}
