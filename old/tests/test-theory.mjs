// Quick sanity checks for theory.js. Run with: node test-theory.mjs

import { stepSaxeODE, initialSingularValues, predictedEigenvalues, predictedLoss, SaxePredictor } from './theory.js';

let passed = 0, failed = 0;
function check(name, cond, msg = '') {
  if (cond) { console.log(`  ✓ ${name}`); passed++; }
  else { console.log(`  ✗ ${name} ${msg}`); failed++; }
}
function approx(a, b, tol = 1e-6) { return Math.abs(a - b) < tol; }

// ---------------------------------------------------------------------------
console.log('\n[1] Initial conditions');
// σ_i(0) = ε^L for every mode.
{
  const s0 = initialSingularValues(0.01, 3, 4);
  check('all entries equal ε^L', s0.every(s => approx(s, 1e-6)), `got ${s0[0]}`);
  check('length matches r', s0.length === 4);
}

// ---------------------------------------------------------------------------
console.log('\n[2] Saxe ODE fixed points');
// σ⋆ is a fixed point; σ = 0 is also a fixed point.
{
  const stayAtTarget = stepSaxeODE([1.0, 0.5], [1.0, 0.5], 2, 0.01);
  check('σ = σ⋆ is stationary', approx(stayAtTarget[0], 1.0) && approx(stayAtTarget[1], 0.5));

  const stayAtZero = stepSaxeODE([0, 0], [1.0, 0.5], 2, 0.01);
  check('σ = 0 is stationary', stayAtZero[0] === 0 && stayAtZero[1] === 0);
}

// ---------------------------------------------------------------------------
console.log('\n[3] Saxe ODE convergence');
// Integrate from σ(0) = ε^L for many steps; should approach σ⋆.
{
  const L = 2;
  const eta = 0.01;
  let sigmas = initialSingularValues(0.01, L, 2);
  const sStar = [1.0, 0.5];
  for (let step = 0; step < 100000; step++) {
    sigmas = stepSaxeODE(sigmas, sStar, L, eta);
  }
  check('converges to σ⋆[0]=1', approx(sigmas[0], 1.0, 1e-3), `got ${sigmas[0]}`);
  check('converges to σ⋆[1]=0.5', approx(sigmas[1], 0.5, 1e-3), `got ${sigmas[1]}`);
}

// ---------------------------------------------------------------------------
console.log('\n[4] Saxe ODE staircase ordering');
// Mode with larger σ⋆ should reach its target before the smaller one — the
// transition time scales as 1/σ⋆ (Saxe et al).
{
  const L = 2;
  const eta = 0.005;
  let sigmas = initialSingularValues(0.001, L, 2);
  const sStar = [2.0, 0.2];   // 10x ratio in targets
  let t1Hit = -1, t2Hit = -1;
  for (let step = 0; step < 50000; step++) {
    sigmas = stepSaxeODE(sigmas, sStar, L, eta);
    if (t1Hit < 0 && sigmas[0] > 0.9 * sStar[0]) t1Hit = step;
    if (t2Hit < 0 && sigmas[1] > 0.9 * sStar[1]) t2Hit = step;
    if (t1Hit >= 0 && t2Hit >= 0) break;
  }
  check('large σ⋆ hits target first', t1Hit > 0 && t1Hit < t2Hit, `t1=${t1Hit} t2=${t2Hit}`);
}

// ---------------------------------------------------------------------------
console.log('\n[5] Eigenvalue formula — aligned modes');
// At σ = σ⋆, aligned eigenvalue = L · σ⋆^(2(L−1)/L).
//   L=2: 2σ⋆     L=3: 3σ⋆^(4/3)     L=1: 1·σ⋆^0 = 1
{
  // L=2, σ⋆ = 1.0
  // Aligned (k=i=0): 2·1 = 2
  // No other modes since r=1.
  const eigs2 = predictedEigenvalues([1.0], [1.0], 2);
  check('L=2 r=1 aligned eig = 2σ', eigs2.length === 1 && approx(eigs2[0], 2.0));

  // L=3, σ⋆ = 1.0
  const eigs3 = predictedEigenvalues([1.0], [1.0], 3);
  check('L=3 r=1 aligned eig = 3·1^(4/3) = 3', eigs3.length === 1 && approx(eigs3[0], 3.0));

  // L=3, σ⋆ = 2.0: 3·2^(4/3) ≈ 7.5595
  const eigs3b = predictedEigenvalues([2.0], [2.0], 3);
  const expected = 3 * Math.pow(2, 4/3);
  check(`L=3 σ=2 aligned eig = 3·2^(4/3) ≈ ${expected.toFixed(4)}`,
        approx(eigs3b[0], expected, 1e-9), `got ${eigs3b[0]}`);
}

// ---------------------------------------------------------------------------
console.log('\n[6] Eigenvalue formula — cross modes (L=2)');
// L=2 cross: c_{ki} = s_k^0·s_i^1 + s_k^1·s_i^0 = s_k + s_i (per writeup).
{
  const L = 2;
  const sigmas = [3.0, 1.0];
  const eigs = predictedEigenvalues(sigmas, [3.0, 1.0], L);
  // All 4 modes:
  //   (i=0,k=0): aligned, 2·3 = 6
  //   (i=0,k=1): cross,   1+3 = 4
  //   (i=1,k=0): cross,   3+1 = 4
  //   (i=1,k=1): aligned, 2·1 = 2
  // Sorted desc: [6, 4, 4, 2]
  check('count = r² = 4', eigs.length === 4);
  check('top = 2·max(σ) = 6', approx(eigs[0], 6));
  check('two cross modes = σ_k + σ_i = 4', approx(eigs[1], 4) && approx(eigs[2], 4));
  check('bottom = 2·min(σ) = 2', approx(eigs[3], 2));
}

// ---------------------------------------------------------------------------
console.log('\n[7] Eigenvalue formula — low-rank-M (one σ⋆ = 0)');
// L=2, r=2 but only σ⋆_0 ≠ 0. σ_1 → 0 over time. Test at σ = (1.0, 0):
//   (0,0) aligned: 2·1 = 2
//   (0,1) cross:   0+1 = 1   (s_k=0, s_i=1)
//   (1,0) cross:   1+0 = 1   (s_k=1, s_i=0)
//   (1,1) aligned: 2·0 = 0    — skipped (both σ zero)
// Per writeup: low-rank-M eig should be half the aligned for L=2 — yes, 1 = 2/2 ✓
{
  const eigs = predictedEigenvalues([1.0, 0], [1.0, 0], 2);
  check('low-rank eig count = 3 (one mode dropped)', eigs.length === 3);
  check('low-rank top = 2', approx(eigs[0], 2));
  check('low-rank eig = aligned/2 for L=2', approx(eigs[1], 1) && approx(eigs[2], 1));
}

// Same for L=3: aligned = 3·1^(4/3) = 3, low-rank = aligned/L = 1.
{
  const eigs = predictedEigenvalues([1.0, 0], [1.0, 0], 3);
  check('L=3 low-rank aligned = 3', approx(eigs[0], 3));
  check('L=3 low-rank eig = aligned/3 = 1', approx(eigs[1], 1) && approx(eigs[2], 1));
}

// ---------------------------------------------------------------------------
console.log('\n[8] SaxePredictor end-to-end (L=2 convergence + eigenvalues)');
{
  const pred = new SaxePredictor({ sigmaStar: [1.0, 0.5], L: 2, epsilon: 0.01 });
  // At init, σ = ε^L = 0.0001 for both. Top eig = 2·0.0001 ≈ small.
  const initEigs = pred.currentEigenvalues();
  check('init top eig is small', initEigs[0] < 0.001, `got ${initEigs[0]}`);

  // Run many steps to convergence.
  const eta = 0.005;
  for (let i = 0; i < 200000; i++) pred.step(eta);
  const finalEigs = pred.currentEigenvalues();
  // Top eig should approach 2·σ⋆_max = 2·1.0 = 2.0
  check('converged top eig ≈ 2σ⋆_max = 2', approx(finalEigs[0], 2.0, 1e-2), `got ${finalEigs[0]}`);
}

// ---------------------------------------------------------------------------
console.log('\n[9] predictedLoss formula');
// L = (1/2) Σ (σ - σ⋆)². At convergence (σ = σ⋆), L = 0. At init (σ = ε^L),
// L = (1/2) Σ (ε^L - σ_i⋆)².
{
  check('L = 0 at σ = σ⋆', predictedLoss([1.0, 0.5, 0.25], [1.0, 0.5, 0.25]) === 0);

  // Hand: σ=(0, 0), σ⋆=(1, 0.5) → L = 0.5·(1 + 0.25) = 0.625
  check('L = 0.5·Σ(σ⋆)² when σ = 0',
        approx(predictedLoss([0, 0], [1.0, 0.5]), 0.625));

  // Hand: σ=(1, 0), σ⋆=(1, 0.5) → L = 0.5·(0 + 0.25) = 0.125 (one mode learned)
  check('one mode learned, one not',
        approx(predictedLoss([1.0, 0], [1.0, 0.5]), 0.125));

  // Low-rank-M: σ⋆ has a zero, σ at init has ε^L > 0 there → small positive contribution.
  // σ=(0.5, 0.001), σ⋆=(0.5, 0) → L = 0.5·(0 + 0.001²) = 5e-7
  check('low-rank-M residual contribution',
        approx(predictedLoss([0.5, 0.001], [0.5, 0]), 0.5e-6, 1e-9));

  // SaxePredictor.currentLoss agrees with predictedLoss directly
  const pred = new SaxePredictor({ sigmaStar: [1.0, 0.5], L: 2, epsilon: 0.1 });
  // Init: σ = ε^L = 0.01 for both modes. L = 0.5·((0.01-1)² + (0.01-0.5)²)
  //     = 0.5·(0.9801 + 0.2401) = 0.6101
  check('SaxePredictor.currentLoss matches at init',
        approx(pred.currentLoss(), 0.6101, 1e-6), `got ${pred.currentLoss()}`);
}

// ---------------------------------------------------------------------------
console.log('\n[10] Substepped RK4 stability at η that singlestep can\'t handle');
// Single-RK4-step at η=1.4 with L=3, σ⋆_max=1 has |dt·λ| = 4.2 — outside the
// real-axis RK4 stability bound (~2.78). With dt/2 substeps, |dt·λ| = 2.1 per
// substep, inside the region. Verify σ stays bounded near σ⋆.
{
  const sigmaStar = [1.0, 0.5];
  const L = 3;
  const epsilon = 0.01;
  const eta = 1.4;
  const pred = new SaxePredictor({ sigmaStar, L, epsilon });
  let maxObserved = 0;
  for (let i = 0; i < 5000; i++) {
    pred.step(eta);
    const s = pred.currentSigmas();
    for (const v of s) {
      if (!isFinite(v)) { maxObserved = Infinity; break; }
      if (Math.abs(v) > maxObserved) maxObserved = Math.abs(v);
    }
    if (!isFinite(maxObserved)) break;
  }
  // Should converge to (1.0, 0.5); peak observed should be only slightly above
  // σ⋆_max. The original single-step RK4 produced σ → 0.155 from 0.79 in one
  // step at η=1.4 (a discontinuous drop) — substepping eliminates that.
  check('σ stays bounded at η = 1.4 (no RK4 blow-up)',
        isFinite(maxObserved) && maxObserved < 2.0, `peak |σ| = ${maxObserved}`);

  const final = pred.currentSigmas();
  check('converges to σ⋆ at η = 1.4', approx(final[0], 1.0, 1e-3) && approx(final[1], 0.5, 1e-3),
        `final = [${final.map(s => s.toFixed(4)).join(', ')}]`);
}

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
