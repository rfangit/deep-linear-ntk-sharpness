// ============================================================================
// EIGEN-JACOBI - Symmetric eigendecomposition via the Jacobi method.
// ============================================================================
// For real symmetric matrices we apply repeated Givens rotations that zero out
// the largest off-diagonal entry. Each rotation preserves symmetry, so after
// enough sweeps the matrix is (numerically) diagonal and the accumulated
// rotation is an orthonormal eigenvector matrix.
//
// We use the classical Jacobi rule (zero the *largest* off-diagonal each
// step) rather than cyclic Jacobi — for the small (≤ ~16×16) matrices this
// module is intended for, the O(n²) search per rotation is negligible and
// classical Jacobi typically converges in fewer rotations.
//
// Returns eigenpairs sorted in descending order of eigenvalue. Eigenvectors
// are returned as a list of column vectors (i.e. eigenvectors[α] is e_α).

/**
 * Eigendecompose a real symmetric matrix.
 *
 * @param {number[][]} A          n×n symmetric matrix. Caller's array is not
 *                                 modified — we work on a copy.
 * @param {object}    [opts]
 * @param {number}    [opts.tol=1e-12]    Stop when off-diagonal Frobenius
 *                                         norm² is below this.
 * @param {number}    [opts.maxSweeps=200] Hard cap on rotations.
 *
 * @returns {{ eigenvalues: number[], eigenvectors: number[][] }}
 *           eigenvalues sorted descending; eigenvectors[α] is the unit
 *           eigenvector for eigenvalues[α].
 */
export function jacobiEigen(A, opts = {}) {
  const tol = opts.tol !== undefined ? opts.tol : 1e-12;
  const maxSweeps = opts.maxSweeps !== undefined ? opts.maxSweeps : 200;

  const n = A.length;

  // Working copies. M starts as A and is rotated toward diagonal; V starts as
  // I and accumulates the rotations as columns.
  const M = new Array(n);
  for (let i = 0; i < n; i++) M[i] = A[i].slice();
  const V = new Array(n);
  for (let i = 0; i < n; i++) {
    V[i] = new Array(n).fill(0);
    V[i][i] = 1;
  }

  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    // Find the largest off-diagonal entry by absolute value.
    let p = 0, q = 1, maxAbs = 0;
    for (let i = 0; i < n - 1; i++) {
      for (let j = i + 1; j < n; j++) {
        const v = Math.abs(M[i][j]);
        if (v > maxAbs) { maxAbs = v; p = i; q = j; }
      }
    }
    if (maxAbs < tol) break;

    // Compute the Jacobi rotation that zeros M[p][q]. The standard formula
    // diagonalizes the 2×2 block [[M[p][p], M[p][q]], [M[p][q], M[q][q]]].
    const app = M[p][p], aqq = M[q][q], apq = M[p][q];
    const theta = (aqq - app) / (2 * apq);
    let t;
    if (Math.abs(theta) > 1e150) {
      // Avoid overflow: when |theta| is enormous, t ≈ 1/(2θ) is tiny.
      t = 1 / (2 * theta);
    } else {
      const sgn = theta >= 0 ? 1 : -1;
      t = sgn / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
    }
    const c = 1 / Math.sqrt(t * t + 1);
    const s = t * c;

    // Update M: rows / columns p and q. The 2×2 block becomes diagonal; rows
    // and columns p, q in the rest of M get linearly combined.
    M[p][p] = app - t * apq;
    M[q][q] = aqq + t * apq;
    M[p][q] = 0;
    M[q][p] = 0;

    for (let i = 0; i < n; i++) {
      if (i !== p && i !== q) {
        const mip = M[i][p];
        const miq = M[i][q];
        M[i][p] = c * mip - s * miq;
        M[i][q] = s * mip + c * miq;
        M[p][i] = M[i][p];
        M[q][i] = M[i][q];
      }
    }

    // Accumulate the rotation into V (columns p and q).
    for (let i = 0; i < n; i++) {
      const vip = V[i][p];
      const viq = V[i][q];
      V[i][p] = c * vip - s * viq;
      V[i][q] = s * vip + c * viq;
    }
  }

  // Pull eigenvalues off the diagonal, package eigenvectors as columns of V.
  const eigenvalues = new Array(n);
  for (let i = 0; i < n; i++) eigenvalues[i] = M[i][i];

  const eigenvectors = new Array(n);
  for (let j = 0; j < n; j++) {
    const col = new Array(n);
    for (let i = 0; i < n; i++) col[i] = V[i][j];
    eigenvectors[j] = col;
  }

  // Sort by eigenvalue descending.
  const order = eigenvalues.map((_, i) => i)
    .sort((a, b) => eigenvalues[b] - eigenvalues[a]);
  const sortedVals = order.map(i => eigenvalues[i]);
  const sortedVecs = order.map(i => eigenvectors[i]);

  return { eigenvalues: sortedVals, eigenvectors: sortedVecs };
}
