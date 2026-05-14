// ============================================================================
// DATA GENERATOR - Compute Y = M·X given pre-built M and X
// ============================================================================
// This module's only job is the linear-regression map. It does not know:
//   - how M was constructed (σ values, explicit, …) — that's matrix.js
//   - how X was constructed (iid Gaussian, whitened, …) — that's inputs.js
//
// Splitting these three concerns lets each one have an isolated reason to
// change. M decides "what is the true function?" X decides "what statistics
// does the data have?" Y just plugs them together.

/**
 * Compute Y = M·X.
 *
 * @param {object}     opts
 * @param {number[][]} opts.M   Target matrix, shape [outputDim][inputDim].
 * @param {number[][]} opts.X   Design matrix, shape [nTrain][inputDim].
 *                              Each row x_i is one input sample.
 *
 * @returns {{ x: number[][], y: (number|number[])[] }}
 *   x is returned unchanged (just a reference to opts.X).
 *   y[i] is a scalar when outputDim === 1, an array otherwise — matching the
 *   convention the Trainer expects.
 */
export function generateLinearData({ M, X }) {
  if (!Array.isArray(M) || M.length === 0 || !Array.isArray(M[0])) {
    throw new Error('generateLinearData: M must be a non-empty 2D array');
  }
  if (!Array.isArray(X) || X.length === 0 || !Array.isArray(X[0])) {
    throw new Error('generateLinearData: X must be a non-empty 2D array');
  }
  const outputDim = M.length;
  const inputDim = M[0].length;
  if (X[0].length !== inputDim) {
    throw new Error(
      `generateLinearData: X has inputDim=${X[0].length} but M expects ${inputDim}`
    );
  }

  const N = X.length;
  const y = new Array(N);

  for (let n = 0; n < N; n++) {
    const xi = X[n];
    const yi = new Array(outputDim);
    for (let i = 0; i < outputDim; i++) {
      let sum = 0;
      const Mi = M[i];
      for (let j = 0; j < inputDim; j++) sum += Mi[j] * xi[j];
      yi[i] = sum;
    }
    y[n] = outputDim === 1 ? yi[0] : yi;
  }

  return { x: X, y };
}
