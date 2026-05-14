// Smoke test for the weight-display formatting helpers in train_widget.js.
// We re-implement them here standalone (they're not exported), and verify the
// shapes and the product-matrix math.

const NUMBER_COL_WIDTH = 9;

function formatNumber(x) {
  if (!isFinite(x)) return '   NaN  ';
  const sign = x < 0 ? '-' : '+';
  const abs = Math.abs(x);
  const s = sign + abs.toFixed(4);
  return s.length >= NUMBER_COL_WIDTH ? s : s.padStart(NUMBER_COL_WIDTH, ' ');
}

function formatMatrix(M) {
  const rows = M.length;
  if (rows === 0) return '  (empty)';
  const cols = M[0].length;
  const lines = [];
  for (let i = 0; i < rows; i++) {
    let row = ' ';
    for (let j = 0; j < cols; j++) row += formatNumber(M[i][j]) + ' ';
    lines.push(row);
  }
  return lines.join('\n');
}

function productMatrix(weightsSnapshot) {
  if (weightsSnapshot.length === 0) return [];
  let P = weightsSnapshot[0].map(row => row.slice());
  for (let l = 1; l < weightsSnapshot.length; l++) {
    const W = weightsSnapshot[l];
    const rows = W.length;
    const inner = W[0].length;
    const outerCols = P[0].length;
    const newP = new Array(rows);
    for (let i = 0; i < rows; i++) {
      const Wi = W[i];
      const newRow = new Array(outerCols).fill(0);
      for (let k = 0; k < inner; k++) {
        const wik = Wi[k];
        const Pk = P[k];
        for (let j = 0; j < outerCols; j++) newRow[j] += wik * Pk[j];
      }
      newP[i] = newRow;
    }
    P = newP;
  }
  return P;
}

let passed = 0, failed = 0;
function check(name, cond, msg = '') {
  if (cond) { console.log(`  ✓ ${name}`); passed++; }
  else { console.log(`  ✗ ${name} ${msg}`); failed++; }
}
function approx(a, b, tol = 1e-9) { return Math.abs(a - b) < tol; }

// ---------------------------------------------------------------------------
console.log('\n[A] formatNumber: column width and sign');
{
  check('positive', formatNumber(0.5) === '  +0.5000');
  check('negative', formatNumber(-0.5) === '  -0.5000');
  check('zero', formatNumber(0) === '  +0.0000');
  check('wide number truncates gracefully', formatNumber(-1234.5).length === '+1234.5000'.length);
  check('NaN renders without crash', formatNumber(NaN).includes('NaN'));
}

// ---------------------------------------------------------------------------
console.log('\n[B] productMatrix: identity chain');
{
  const I2 = [[1,0],[0,1]];
  const P = productMatrix([I2, I2, I2]);
  check('I·I·I = I', approx(P[0][0], 1) && approx(P[1][1], 1) && approx(P[0][1], 0));
}

// ---------------------------------------------------------------------------
console.log('\n[C] productMatrix: deep aligned init produces ε^L · diag');
// Build a chain that mimics aligned init with randomO=false and U=V=I.
// All inner orthogonals are identity, so W_ℓ = ε·D (diag matrix with ε on
// the diagonal, possibly rectangular). Product W_L···W_1 must be diagonal
// with entries ε^L.
{
  const eps = 0.1;
  const L = 3;
  const layers = [];
  // Shape: [k=3, h1=3, h2=3, m=3] all square for simplicity
  for (let l = 0; l < L; l++) {
    const W = [
      [eps, 0,   0],
      [0,   eps, 0],
      [0,   0,   eps]
    ];
    layers.push(W);
  }
  const P = productMatrix(layers);
  const expected = Math.pow(eps, L);  // 0.001
  check('product diagonal entries = ε^L',
        approx(P[0][0], expected) && approx(P[1][1], expected) && approx(P[2][2], expected),
        `got [${P[0][0]}, ${P[1][1]}, ${P[2][2]}], expected ${expected}`);
  check('off-diagonals are zero',
        approx(P[0][1], 0) && approx(P[1][0], 0) && approx(P[0][2], 0));
}

// ---------------------------------------------------------------------------
console.log('\n[D] productMatrix: rectangular layers');
// W_1: 3×2, W_2: 2×3 → product 2×2
{
  const W1 = [[1, 2], [3, 4], [5, 6]];        // 3×2
  const W2 = [[1, 0, 0], [0, 1, 0]];          // 2×3
  // (W2 · W1)[i][j] = sum_k W2[i][k] · W1[k][j]
  //   row 0: [1,0,0]·col0 = 1, [1,0,0]·col1 = 2  → [1, 2]
  //   row 1: [0,1,0]·col0 = 3, [0,1,0]·col1 = 4  → [3, 4]
  const P = productMatrix([W1, W2]);
  check('rectangular shape', P.length === 2 && P[0].length === 2);
  check('rectangular product values',
        approx(P[0][0], 1) && approx(P[0][1], 2) && approx(P[1][0], 3) && approx(P[1][1], 4));
}

// ---------------------------------------------------------------------------
console.log('\n[E] formatMatrix: shape and basic rendering');
{
  const M = [[0.1, -0.2], [0.3, 0.4]];
  const s = formatMatrix(M);
  const lines = s.split('\n');
  check('two rows', lines.length === 2);
  check('mentions +0.1000 (positive)', lines[0].includes('+0.1000'));
  check('mentions -0.2000 (negative)', lines[0].includes('-0.2000'));
}

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
