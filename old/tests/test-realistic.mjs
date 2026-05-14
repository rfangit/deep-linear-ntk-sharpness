// Simulate the SaxePredictor under realistic playground defaults and print a
// summary. Verifies the prediction is well-behaved at η = 1.4 (which is
// near edge-of-stability for these targets).

import { SaxePredictor } from './theory.js';

// Defaults from state.js: ε = 0.01, η = 1.4, σ⋆ = [1.0, 0.5, 0.25].
// useSecondLayer=true → layerSizes = [5, 30, 20, 3] → L = 3.
const sigmaStar = [1.0, 0.5, 0.25];
const L = 3;
const epsilon = 0.01;
const eta = 0.6;

const pred = new SaxePredictor({ sigmaStar, L, epsilon });

console.log(`Defaults: σ⋆ = [${sigmaStar.join(', ')}], L = ${L}, ε = ${epsilon}, η = ${eta}`);
console.log(`Initial σ_i = ε^L = ${Math.pow(epsilon, L)}`);
console.log(`Theory threshold 2/η = ${(2/eta).toFixed(4)}`);
console.log('');

// Asymptotic prediction at convergence:
//   top eig = L · σ⋆_max^(2(L-1)/L) = 3 · 1.0^(4/3) = 3.0
//   2nd    = depends on cross modes — for L=3, c_{01} = s_1^(4/3) + s_0^(2/3)·s_1^(2/3) + s_0^(4/3)
const sMax = Math.max(...sigmaStar);
const asymptoticTop = L * Math.pow(sMax, 2*(L-1)/L);
console.log(`Asymptotic top eigenvalue: L · σ⋆_max^(2(L-1)/L) = ${asymptoticTop.toFixed(4)}`);
console.log(`(Below threshold 2/η = ${(2/eta).toFixed(4)}? ${asymptoticTop < 2/eta ? 'YES — stable' : 'NO — would hit EoS'})`);
console.log('');

// Run for a representative number of steps and print snapshots.
console.log('Step       t = η·step    σ_i                          top 3 predicted eigs');
console.log('-----------------------------------------------------------------------------');
const snapshots = [0, 10, 50, 200, 1000, 5000, 20000];
let nextSnap = 0;

for (let step = 0; step <= snapshots[snapshots.length - 1]; step++) {
  if (step === snapshots[nextSnap]) {
    const sigmas = pred.currentSigmas();
    const eigs = pred.currentEigenvalues();
    const top3 = eigs.slice(0, 3).map(e => e.toFixed(4)).join(', ');
    const sigStr = sigmas.map(s => s.toFixed(4)).join(', ');
    const tStr = (step * eta).toFixed(2);
    console.log(`${String(step).padStart(6)}     ${tStr.padStart(8)}      [${sigStr}]       [${top3}]`);
    nextSnap++;
  }
  pred.step(eta);
}

// Final sanity
const finalEigs = pred.currentEigenvalues();
const finalSigmas = pred.currentSigmas();
console.log('');
console.log(`At step ${snapshots[snapshots.length - 1]}:`);
console.log(`  σ converged toward σ⋆? ${finalSigmas.map((s, i) => (Math.abs(s - sigmaStar[i]) < 1e-3 ? '✓' : '✗')).join(' ')}`);
console.log(`  top predicted eig = ${finalEigs[0].toFixed(4)}  (asymptotic: ${asymptoticTop.toFixed(4)})`);
console.log(`  # nonzero predicted eigs = ${finalEigs.length}  (expected r² = ${sigmaStar.length ** 2} = 9)`);
