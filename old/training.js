// ============================================================================
// TRAINING - Full-batch GD for a deep linear network
// ============================================================================
// Operates on a fixed dataset, full-batch every step. Loss is sum-over-outputs
// of squared error, averaged over the batch:
//   L = (1/|B|) Σ_i (1/2) Σ_j (y_ij − ŷ_ij)²
//
// No biases. No activation derivatives — every layer is linear, so the local
// derivative is the identity and backprop is just repeated W^T multiplications.

export class Trainer {
  /**
   * @param {MLP}    model
   * @param {number} learningRate
   * @param {object} dataset  { x: number[][], y: (number|number[])[] }
   */
  constructor(model, learningRate, dataset) {
    this.model = model;
    this.eta = learningRate;

    this.dataX = dataset.x;
    this.dataSize = dataset.x.length;
    this.outputDim = model.layerSizes[model.layerSizes.length - 1];

    // Normalize targets to arrays (scalar y wrapped to [y]) for uniform handling.
    this.dataYArrays = dataset.y.map(y => Array.isArray(y) ? y : [y]);

    // Pre-allocated gradient and delta buffers.
    this.gradW = [];
    this.delta = [];
    for (let l = 0; l < model.numLayers; l++) {
      const rows = model.W[l].length;
      const cols = model.W[l][0].length;
      this.gradW.push(zeros2D(rows, cols));
      this.delta.push(new Array(rows).fill(0));
    }

    // The flat gradient from the most recent step(), in the same parameter
    // order as hessian.js (per-layer, row-major). Used by Simulation to
    // project gradient updates onto a stored eigenvector.
    this.lastGradFlat = null;
  }

  /**
   * Compute the full-dataset gradient as a flat vector. Used by hessian.js
   * for finite-difference Hessian-vector products.
   *
   * Parameter order: for each layer l, all W[l] entries row-major.
   */
  computeGradientFlat(dataX, dataYArrays) {
    const model = this.model;
    const numLayers = model.numLayers;
    const outputDim = this.outputDim;
    const N = dataX.length;

    // Zero accumulators
    for (let l = 0; l < numLayers; l++) {
      const rows = model.W[l].length;
      const cols = model.W[l][0].length;
      const g = this.gradW[l];
      for (let i = 0; i < rows; i++) {
        const gi = g[i];
        for (let j = 0; j < cols; j++) gi[j] = 0;
      }
    }

    for (let idx = 0; idx < N; idx++) {
      const x = dataX[idx];
      const yArr = dataYArrays[idx];
      const fwd = model.forward(x);
      const outArr = fwd.activations[fwd.activations.length - 1];

      // Output-layer delta: dL/dz_L = -(y - ŷ)
      const lastLayer = numLayers - 1;
      const deltaLast = this.delta[lastLayer];
      for (let j = 0; j < outputDim; j++) {
        deltaLast[j] = -(yArr[j] - outArr[j]);
      }

      // Backward sweep. For linear nets, dL/dz_{l-1} = W_l^T · dL/dz_l.
      for (let l = lastLayer; l >= 0; l--) {
        const W = model.W[l];
        const rows = W.length;
        const cols = W[0].length;
        const aIn = fwd.activations[l];
        const dl = this.delta[l];
        const gl = this.gradW[l];

        // gradW[l] += dl ⊗ aIn
        for (let i = 0; i < rows; i++) {
          const gi = gl[i];
          const dli = dl[i];
          for (let j = 0; j < cols; j++) gi[j] += dli * aIn[j];
        }

        // Propagate delta to previous layer
        if (l > 0) {
          const dPrev = this.delta[l - 1];
          for (let j = 0; j < cols; j++) {
            let sum = 0;
            for (let i = 0; i < rows; i++) sum += W[i][j] * dl[i];
            dPrev[j] = sum;
          }
        }
      }
    }

    // Flatten + average
    const flat = [];
    for (let l = 0; l < numLayers; l++) {
      const rows = model.W[l].length;
      const cols = model.W[l][0].length;
      const g = this.gradW[l];
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) flat.push(g[i][j] / N);
      }
    }
    return flat;
  }

  /**
   * One full-batch GD step. Returns the average loss on the batch.
   */
  step() {
    const model = this.model;
    const numLayers = model.numLayers;
    const outputDim = this.outputDim;
    const N = this.dataSize;

    // Zero gradient accumulators
    for (let l = 0; l < numLayers; l++) {
      const rows = model.W[l].length;
      const cols = model.W[l][0].length;
      const g = this.gradW[l];
      for (let i = 0; i < rows; i++) {
        const gi = g[i];
        for (let j = 0; j < cols; j++) gi[j] = 0;
      }
    }

    let totalLoss = 0;

    for (let idx = 0; idx < N; idx++) {
      const x = this.dataX[idx];
      const yArr = this.dataYArrays[idx];
      const fwd = model.forward(x);
      const outArr = fwd.activations[fwd.activations.length - 1];

      // Per-sample loss: (1/2) Σ_j (y_j - ŷ_j)²
      // Initial delta: dL/dz_L = -(y - ŷ)
      const lastLayer = numLayers - 1;
      const deltaLast = this.delta[lastLayer];
      for (let j = 0; j < outputDim; j++) {
        const err = yArr[j] - outArr[j];
        totalLoss += 0.5 * err * err;
        deltaLast[j] = -err;
      }

      // Backward sweep
      for (let l = lastLayer; l >= 0; l--) {
        const W = model.W[l];
        const rows = W.length;
        const cols = W[0].length;
        const aIn = fwd.activations[l];
        const dl = this.delta[l];
        const gl = this.gradW[l];

        // gradW[l] += dl ⊗ aIn
        for (let i = 0; i < rows; i++) {
          const gi = gl[i];
          const dli = dl[i];
          for (let j = 0; j < cols; j++) gi[j] += dli * aIn[j];
        }

        // Linear: dL/dz_{l-1} = W^T · dL/dz_l
        if (l > 0) {
          const dPrev = this.delta[l - 1];
          for (let j = 0; j < cols; j++) {
            let sum = 0;
            for (let i = 0; i < rows; i++) sum += W[i][j] * dl[i];
            dPrev[j] = sum;
          }
        }
      }
    }

    // Average gradients, store the flat copy, apply update.
    const flatGrad = [];
    for (let l = 0; l < numLayers; l++) {
      const rows = model.W[l].length;
      const cols = model.W[l][0].length;
      const g = this.gradW[l];
      for (let i = 0; i < rows; i++) {
        const gi = g[i];
        for (let j = 0; j < cols; j++) {
          gi[j] /= N;
          flatGrad.push(gi[j]);
        }
      }
    }
    this.lastGradFlat = flatGrad;

    for (let l = 0; l < numLayers; l++) {
      const rows = model.W[l].length;
      const cols = model.W[l][0].length;
      const W = model.W[l];
      const g = this.gradW[l];
      for (let i = 0; i < rows; i++) {
        const Wi = W[i];
        const gi = g[i];
        for (let j = 0; j < cols; j++) Wi[j] -= this.eta * gi[j];
      }
    }

    return totalLoss / N;
  }
}

function zeros2D(rows, cols) {
  const M = new Array(rows);
  for (let i = 0; i < rows; i++) M[i] = new Array(cols).fill(0);
  return M;
}
