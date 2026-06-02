// ============================================================================
// DENSE HESSIAN - exact full spectrum via finite-difference + Jacobi
// ============================================================================
// Builds the full P×P Hessian of the loss by finite-differencing the gradient
// component-by-component (reusing Trainer.computeGradientFlat), then returns
// its complete eigenvalue spectrum via the cyclic Jacobi method.
//
// This is O(P²) gradient evaluations to build H and O(P³) to diagonalize, so it
// is ONLY appropriate for very small models (P up to a few hundred). It exists
// as a ground-truth comparison against the Lanczos estimate — Lanczos returns
// only the top-k distinct eigenvalues (one per eigenspace) and can miss
// multiplicities and the negative branch of an indefinite Hessian, whereas this
// returns the entire spectrum exactly (up to FD error).
//
// Parameter order matches hessian.js / training.js: per layer, row-major.

function flattenParams(model) {
  const flat = [];
  for (let l = 0; l < model.numLayers; l++) {
    const W = model.W[l];
    for (let i = 0; i < W.length; i++) {
      for (let j = 0; j < W[i].length; j++) flat.push(W[i][j]);
    }
  }
  return flat;
}

function unflattenParams(model, flat) {
  let k = 0;
  for (let l = 0; l < model.numLayers; l++) {
    const W = model.W[l];
    for (let i = 0; i < W.length; i++) {
      for (let j = 0; j < W[i].length; j++) W[i][j] = flat[k++];
    }
  }
}

/**
 * Build the dense P×P Hessian via symmetric finite differences of the gradient.
 * Restores the model's original parameters before returning.
 *
 * @param {Trainer} trainer
 * @param {number[][]} dataX
 * @param {Array} dataYArrays
 * @param {number} epsilon  FD step (default 1e-5)
 * @returns {number[][]} symmetric P×P Hessian
 */
export function buildDenseHessian(trainer, dataX, dataYArrays, epsilon = 1e-5) {
  const model = trainer.model;
  const base = flattenParams(model);
  const P = base.length;
  const H = new Array(P);
  for (let i = 0; i < P; i++) H[i] = new Array(P).fill(0);

  for (let p = 0; p < P; p++) {
    const vp = base.slice(); vp[p] += epsilon;
    unflattenParams(model, vp);
    const gp = trainer.computeGradientFlat(dataX, dataYArrays);

    const vm = base.slice(); vm[p] -= epsilon;
    unflattenParams(model, vm);
    const gm = trainer.computeGradientFlat(dataX, dataYArrays);

    const inv = 1 / (2 * epsilon);
    for (let q = 0; q < P; q++) H[q][p] = (gp[q] - gm[q]) * inv;
  }

  unflattenParams(model, base);

  // Symmetrize (FD makes H[i][j] and H[j][i] agree only up to O(ε²)).
  for (let i = 0; i < P; i++) {
    for (let j = i + 1; j < P; j++) {
      const a = 0.5 * (H[i][j] + H[j][i]);
      H[i][j] = a; H[j][i] = a;
    }
  }
  return H;
}

/**
 * All eigenvalues of a symmetric matrix via cyclic Jacobi rotations, returned
 * sorted ASCENDING (to match hessian.js's lanczosTopEigenvalues convention, so
 * the visualization layer can treat both the same way).
 *
 * @param {number[][]} A symmetric matrix (modified internally on a copy)
 * @param {object} [opts] { maxSweeps=100, tol=1e-12 }
 * @returns {number[]} eigenvalues ascending, length A.length
 */
export function symmetricEigenvalues(A, opts = {}) {
  const { maxSweeps = 100, tol = 1e-12 } = opts;
  const n = A.length;
  if (n === 0) return [];
  const a = A.map(r => r.slice());

  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    let off = 0;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += a[i][j] * a[i][j];
    if (off < tol) break;

    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(a[p][q]) < 1e-300) continue;
        const app = a[p][p], aqq = a[q][q], apq = a[p][q];
        const phi = 0.5 * Math.atan2(2 * apq, aqq - app);
        const c = Math.cos(phi), s = Math.sin(phi);
        // Apply rotation on columns p,q then rows p,q.
        for (let k = 0; k < n; k++) {
          const akp = a[k][p], akq = a[k][q];
          a[k][p] = c * akp - s * akq;
          a[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = a[p][k], aqk = a[q][k];
          a[p][k] = c * apk - s * aqk;
          a[q][k] = s * apk + c * aqk;
        }
      }
    }
  }

  const ev = new Array(n);
  for (let i = 0; i < n; i++) ev[i] = a[i][i];
  ev.sort((x, y) => x - y);   // ascending
  return ev;
}

/**
 * Convenience: exact top-k eigenvalues of the loss Hessian, ascending — the
 * same shape lanczosTopEigenvalues returns, so the two are drop-in comparable.
 * Returns ALL eigenvalues if k >= P.
 *
 * Also returns the COMPLETE ascending spectrum (`allEigenvalues`, length P) at
 * no extra cost — symmetricEigenvalues computes the whole spectrum regardless,
 * so callers that want every exact eigenvalue (e.g. to plot all P) can take this
 * instead of the truncated top-k. `eigenvalues` stays the top-k slice for the
 * existing Lanczos-aligned overlay.
 *
 * @returns {{ eigenvalues: number[], allEigenvalues: number[], P: number }}
 */
export function denseTopEigenvalues(trainer, dataX, dataYArrays, options = {}) {
  const { kEigs = 10, epsilon = 1e-5 } = options;
  const H = buildDenseHessian(trainer, dataX, dataYArrays, epsilon);
  const all = symmetricEigenvalues(H);          // ascending, length P
  const P = all.length;
  const k = Math.min(kEigs, P);
  // top-k by magnitude? No — match lanczos: it returns the algebraically
  // largest top-k (ascending). Take the last k (largest) ascending.
  const top = all.slice(P - k);
  return { eigenvalues: top, allEigenvalues: all, P };
}
