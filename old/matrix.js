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

/**
 * Construct M = U Σ V^T (or M = Σ when randomBasis is false) where Σ is the
 * outputDim × inputDim "diagonal" matrix carrying the supplied singular values
 * along its diagonal.
 *
 * If singularValues has more entries than min(outputDim, inputDim) — which can
 * happen when inputDim < outputDim — the extras are silently ignored. The UI
 * is expected to surface a warning in that case.
 *
 * @param {object}   opts
 * @param {number}   opts.inputDim
 * @param {number}   opts.outputDim
 * @param {number[]} opts.singularValues
 * @param {number}   opts.basisSeed       Seed for U, V. Independent of dataSeed.
 * @param {boolean} [opts.randomBasis=true]
 *
 * @returns {number[][]} M with shape [outputDim][inputDim].
 */
export function buildMFromSingularValues({
  inputDim,
  outputDim,
  singularValues,
  basisSeed,
  randomBasis = true
}) {
  const m = outputDim, k = inputDim;
  const r = Math.min(m, k);
  const svs = singularValues.slice(0, r);

  if (!randomBasis) {
    // M = Σ, an m × k matrix with σ_i on the (i, i) diagonal.
    const M = [];
    for (let i = 0; i < m; i++) {
      M.push(new Array(k).fill(0));
      if (i < r) M[i][i] = svs[i] !== undefined ? svs[i] : 0;
    }
    return M;
  }

  const rng = mulberry32(basisSeed);
  const U = randomOrthogonal(m, rng);
  const V = randomOrthogonal(k, rng);

  // M[i][j] = Σ_{l=0..r-1} U[i][l] · σ_l · V[j][l]
  const M = [];
  for (let i = 0; i < m; i++) {
    M[i] = new Array(k);
    for (let j = 0; j < k; j++) {
      let sum = 0;
      for (let l = 0; l < r; l++) {
        const sigma = svs[l] !== undefined ? svs[l] : 0;
        sum += U[i][l] * sigma * V[j][l];
      }
      M[i][j] = sum;
    }
  }
  return M;
}

/**
 * Build M from a matrixSpec object. Single dispatch point so AppState (or any
 * other caller) can rebuild M without caring about the mode.
 *
 * @param {object} matrixSpec  { mode, ...mode-specific fields }
 * @param {number} inputDim
 * @param {number} outputDim
 * @returns {number[][]}
 */
export function buildMFromSpec(matrixSpec, inputDim, outputDim) {
  switch (matrixSpec.mode) {
    case 'singular-values':
      return buildMFromSingularValues({
        inputDim,
        outputDim,
        singularValues: matrixSpec.singularValues || [],
        basisSeed: matrixSpec.basisSeed || 0,
        randomBasis: matrixSpec.randomBasis !== false
      });
    default:
      throw new Error(`buildMFromSpec: unknown mode "${matrixSpec.mode}"`);
  }
}

// ---------------------------------------------------------------------------
// COMPONENT-LEVEL CONSTRUCTOR
// ---------------------------------------------------------------------------
// buildMFromSpec returns only the assembled matrix M. For aligned-init (see
// model.js) we additionally need the SVD basis U, V and the singular value
// vector that define M, so the model's per-layer weights can be placed *in
// that basis*. This sibling function exposes those pieces.
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
