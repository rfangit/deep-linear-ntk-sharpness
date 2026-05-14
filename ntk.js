// ============================================================================
// NTK - Neural Tangent Kernel for deep linear MLPs
// ============================================================================
// Computes Θ(x, x') = Σ_θ ∂f(x)/∂θ · (∂f(x')/∂θ)^T, an n × n matrix where n is
// the model's output dimension.
//
// Implementation: per-output Jacobians via backprop. For each output index j we
// run a backward sweep with the output-layer delta set to e_j (j-th standard
// basis vector), which gives ∂f_j/∂θ as a flat vector of length P (parameter
// count). Stacking these as rows yields J(x) ∈ R^{n × P}. Then
//
//     Θ(x, x') = J(x) · J(x')^T.
//
// This works for *any* depth and any architecture you can backprop through —
// it is the autodiff-equivalent of computing the NTK, but riding on the same
// linear backward sweep that training.js uses. No autodiff library required.
//
// For 2-layer linear nets there is also a closed-form expression
//
//     Θ(x, x') = ⟨x, x'⟩ W₂ W₂^T  +  ⟨W₁ x, W₁ x'⟩ I_n.
//
// We expose this as `ntkBlockAnalytic2Layer` and use it as a development-time
// correctness check.

// ---------------------------------------------------------------------------
// Per-output Jacobian via backprop. Returns ∂f_j(x)/∂θ as a flat array of
// length numParameters(), in the same parameter order as model.W (per-layer,
// row-major). Does NOT touch any external state — uses fresh buffers — so it
// is safe to call from anywhere, including alongside an active Trainer.
// ---------------------------------------------------------------------------
function jacobianRow(model, x, j) {
  const numLayers = model.numLayers;
  const fwd = model.forward(x);
  const activations = fwd.activations;

  // Output-layer delta: e_j (one-hot at output index j).
  const outputDim = model.layerSizes[numLayers];
  let delta = new Array(outputDim).fill(0);
  delta[j] = 1;

  // We'll collect per-layer parameter gradients then flatten.
  const gradPerLayer = new Array(numLayers);

  for (let l = numLayers - 1; l >= 0; l--) {
    const W = model.W[l];
    const rows = W.length;
    const cols = W[0].length;
    const aIn = activations[l];

    // ∂f_j/∂W[l] = delta ⊗ aIn   (rows × cols)
    const g = new Array(rows);
    for (let i = 0; i < rows; i++) {
      const gi = new Array(cols);
      const di = delta[i];
      for (let k = 0; k < cols; k++) gi[k] = di * aIn[k];
      g[i] = gi;
    }
    gradPerLayer[l] = g;

    // Propagate delta to previous layer: delta_{l-1} = W^T · delta_l
    if (l > 0) {
      const dPrev = new Array(cols).fill(0);
      for (let k = 0; k < cols; k++) {
        let sum = 0;
        for (let i = 0; i < rows; i++) sum += W[i][k] * delta[i];
        dPrev[k] = sum;
      }
      delta = dPrev;
    }
  }

  // Flatten in (layer, row, col) order.
  const flat = [];
  for (let l = 0; l < numLayers; l++) {
    const g = gradPerLayer[l];
    const rows = g.length;
    const cols = g[0].length;
    for (let i = 0; i < rows; i++) {
      for (let k = 0; k < cols; k++) flat.push(g[i][k]);
    }
  }
  return flat;
}

// ---------------------------------------------------------------------------
// Full Jacobian J(x) ∈ R^{n × P}: stack jacobianRow for j = 0..n-1.
// ---------------------------------------------------------------------------
function jacobian(model, x) {
  const n = model.layerSizes[model.numLayers];
  const J = new Array(n);
  for (let j = 0; j < n; j++) J[j] = jacobianRow(model, x, j);
  return J;
}

// ---------------------------------------------------------------------------
// NTK block for a pair of inputs (x, x'): n × n matrix.
//
// Θ(x, x')_{ij} = ∂f_i(x)/∂θ · ∂f_j(x')/∂θ
//
// Internal: the public API is ntkMatrix below, which tiles many such blocks.
// ---------------------------------------------------------------------------
function ntkBlock(model, x, xPrime) {
  const Jx = jacobian(model, x);
  const Jxp = (x === xPrime) ? Jx : jacobian(model, xPrime);

  const n = Jx.length;
  const P = Jx[0].length;
  const block = new Array(n);
  for (let i = 0; i < n; i++) {
    const row = new Array(n);
    const Ji = Jx[i];
    for (let j = 0; j < n; j++) {
      const Jj = Jxp[j];
      let s = 0;
      for (let p = 0; p < P; p++) s += Ji[p] * Jj[p];
      row[j] = s;
    }
    block[i] = row;
  }
  return block;
}

// ---------------------------------------------------------------------------
// Full NTK matrix for a list of data points: (N·n) × (N·n).
//
// Indexing: row/column index is (a · n + i) where a is the data-point index
// and i is the output-dim index. So the (1,0)-output1, (1,0)-output2,
// (0,1)-output1, (0,1)-output2 ordering the widget displays falls out
// naturally when dataPoints = [[1,0], [0,1]].
// ---------------------------------------------------------------------------
export function ntkMatrix(model, dataPoints) {
  const N = dataPoints.length;
  const n = model.layerSizes[model.numLayers];

  // Cache jacobians once per data point.
  const Js = dataPoints.map(x => jacobian(model, x));
  const P = Js[0][0].length;

  const dim = N * n;
  const M = new Array(dim);
  for (let r = 0; r < dim; r++) M[r] = new Array(dim).fill(0);

  for (let a = 0; a < N; a++) {
    for (let b = 0; b < N; b++) {
      const Ja = Js[a];
      const Jb = Js[b];
      for (let i = 0; i < n; i++) {
        const Ji = Ja[i];
        for (let j = 0; j < n; j++) {
          const Jj = Jb[j];
          let s = 0;
          for (let p = 0; p < P; p++) s += Ji[p] * Jj[p];
          M[a * n + i][b * n + j] = s;
        }
      }
    }
  }
  return M;
}

// ---------------------------------------------------------------------------
// Analytic 2-layer version of ntkMatrix: tiles the closed-form blocks.
// Identical interface to ntkMatrix; only valid for 2-layer models.
// ---------------------------------------------------------------------------
export function ntkMatrixAnalytic2Layer(model, dataPoints) {
  if (model.numLayers !== 2) {
    throw new Error(`ntkMatrixAnalytic2Layer: requires 2 layers, got ${model.numLayers}`);
  }
  const N = dataPoints.length;
  const n = model.layerSizes[model.numLayers];
  const dim = N * n;
  const M = new Array(dim);
  for (let r = 0; r < dim; r++) M[r] = new Array(dim).fill(0);

  for (let a = 0; a < N; a++) {
    for (let b = 0; b < N; b++) {
      const block = ntkBlockAnalytic2Layer(model, dataPoints[a], dataPoints[b]);
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          M[a * n + i][b * n + j] = block[i][j];
        }
      }
    }
  }
  return M;
}

// ---------------------------------------------------------------------------
// Closed-form 2-layer NTK block:
//   Θ(x, x') = ⟨x, x'⟩ W₂ W₂^T + ⟨W₁ x, W₁ x'⟩ I_n
//
// Used as a correctness check against the Jacobian implementation. Only valid
// when model has exactly two weight matrices.
// ---------------------------------------------------------------------------
export function ntkBlockAnalytic2Layer(model, x, xPrime) {
  if (model.numLayers !== 2) {
    throw new Error(`ntkBlockAnalytic2Layer: requires 2 layers, got ${model.numLayers}`);
  }
  const W1 = model.W[0];
  const W2 = model.W[1];
  const n = W2.length;       // output dim
  const m = W2[0].length;    // hidden dim
  const d = W1[0].length;    // input dim

  // ⟨x, x'⟩
  let xx = 0;
  for (let k = 0; k < d; k++) xx += x[k] * xPrime[k];

  // W₁ x and W₁ x'
  const W1x = new Array(m).fill(0);
  const W1xp = new Array(m).fill(0);
  for (let i = 0; i < m; i++) {
    let s1 = 0, s2 = 0;
    const W1i = W1[i];
    for (let k = 0; k < d; k++) {
      s1 += W1i[k] * x[k];
      s2 += W1i[k] * xPrime[k];
    }
    W1x[i] = s1;
    W1xp[i] = s2;
  }
  let w1xw1xp = 0;
  for (let i = 0; i < m; i++) w1xw1xp += W1x[i] * W1xp[i];

  // (W₂ W₂^T)_{ij} = Σ_k W₂[i][k] · W₂[j][k]
  const block = new Array(n);
  for (let i = 0; i < n; i++) {
    const row = new Array(n);
    const W2i = W2[i];
    for (let j = 0; j < n; j++) {
      const W2j = W2[j];
      let s = 0;
      for (let k = 0; k < m; k++) s += W2i[k] * W2j[k];
      row[j] = xx * s + (i === j ? w1xw1xp : 0);
    }
    block[i] = row;
  }
  return block;
}

// ---------------------------------------------------------------------------
// Dev-time check: Jacobian and analytic agree for a 2-layer model.
// Returns the max absolute difference between the two implementations on a
// pair of inputs. Should be ~1e-12 or smaller.
// ---------------------------------------------------------------------------
export function _checkAnalyticVsJacobian(model, x, xPrime) {
  const a = ntkBlock(model, x, xPrime);
  const b = ntkBlockAnalytic2Layer(model, x, xPrime);
  const n = a.length;
  let maxDiff = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const d = Math.abs(a[i][j] - b[i][j]);
      if (d > maxDiff) maxDiff = d;
    }
  }
  return maxDiff;
}
