// ============================================================================
// THEORY AGGREGATION - σ history → plottable eigenvalue curves
// ============================================================================
// The simulation stores only σ(t) (sigmaHistory). This module derives grouped
// eigenvalue curves at plot time by calling the stateless theory functions
// (theory_GN / theory_Hessian_full) on each stored σ, then aggregating the
// per-timepoint grouped eigenvalues into continuous time series for Chart.js.
//
// Two independent user-knob axes:
//
//   WHICH THEORY ('gn' | 'full' | 'both')  → which theory function(s) to call.
//     GN and full are different theoretical objects: when 'both', they are
//     aggregated and drawn SEPARATELY (GN dotted, full dash-dot), never merged
//     into one ranked list.
//
//   HOW TO PLOT ('pooled' | 'perClass')    → aggregation within each theory.
//     pooled:   flatten that theory's groups, sort desc, take top-k overall.
//     perClass: independent top-k per group (class), each its own curve set.
//
// Curve continuity: at each timepoint we read values, sort DESCENDING, and
// slice top-k. Sorting per-timepoint (not tracking a fixed record across time)
// keeps curves continuous through mode crossings.
//
// Group names per theory:
//   GN:   aligned, cross, single_value
//   full: aligned, aligned_null, cross, single_value, hidden_null, idle_null
//
// idle_null is the flat zero class (count (m−r)(n+d−2r)). It's a genuine part of
// the spectrum, listed here so it can be drawn as its own per-class curve and so
// pooled totals reconcile with a dense diagonalization. In the pooled top-k its
// zeros sort to the bottom, so they only ever appear once k exceeds the number
// of nonzero eigenvalues.

import { theory_GN, theory_Hessian_full } from './theory.js';

export const GN_GROUPS   = ['aligned', 'cross', 'single_value'];
export const FULL_GROUPS = ['aligned', 'aligned_null', 'cross', 'single_value', 'hidden_null', 'idle_null'];

export const GROUP_LABELS = {
  aligned:      'aligned',
  aligned_null: 'aligned (null)',
  cross:        'cross',
  single_value: 'single-value',
  hidden_null:  'hidden null (±)',
  idle_null:    'hidden null (0)'
};

/**
 * Evaluate one theory function over the whole σ history.
 * @param {'gn'|'full'} theory
 * @param {Array<{iteration:number, sigmas:number[]}>} sigmaHistory
 * @param {number[]} sigmaStar
 * @param {number} n
 * @param {number} d
 * @param {number} m   hidden width — only used by the full theory's hidden_null
 *                     class. Ignored by GN.
 * @returns {Array<{iteration:number, groups:object}>}
 *   groups: { groupName: number[] of values } at that timepoint.
 */
function evalTheory(theory, sigmaHistory, sigmaStar, n, d, m, L = 2) {
  const useFull = theory === 'full';
  return sigmaHistory.map(entry => {
    const grouped = useFull
      ? theory_Hessian_full(entry.sigmas, sigmaStar, n, d, m)  // L = 2 only
      : theory_GN(entry.sigmas, sigmaStar, n, d, L);           // any depth
    const groups = {};
    for (const g of Object.keys(grouped)) {
      groups[g] = grouped[g].map(rec => rec.value);
    }
    return { iteration: entry.iteration, groups };
  });
}

/**
 * Largest member count a given group reaches across history — the cap for that
 * group's top-k input in the UI.
 */
export function groupModeCount(theory, sigmaHistory, sigmaStar, n, d, m, group, L = 2) {
  if (sigmaHistory.length === 0) return 0;
  const evaluated = evalTheory(theory, sigmaHistory, sigmaStar, n, d, m, L);
  let max = 0;
  for (const e of evaluated) {
    const c = e.groups[group] ? e.groups[group].length : 0;
    if (c > max) max = c;
  }
  return max;
}

/** Total pooled member count for a theory — cap for pooled top-k. Counts the
 *  plotting groups (GN_GROUPS / FULL_GROUPS); for the full theory this now
 *  includes idle_null, so the cap equals the full P = m(n+d) when 'full'. */
export function pooledModeCount(theory, sigmaHistory, sigmaStar, n, d, m, L = 2) {
  if (sigmaHistory.length === 0) return 0;
  const groupNames = theory === 'full' ? FULL_GROUPS : GN_GROUPS;
  const evaluated = evalTheory(theory, sigmaHistory, sigmaStar, n, d, m, L);
  let max = 0;
  for (const e of evaluated) {
    let total = 0;
    for (const g of groupNames) total += e.groups[g] ? e.groups[g].length : 0;
    if (total > max) max = total;
  }
  return max;
}

/**
 * POOLED top-k for one theory. Flatten all groups at each timepoint, sort
 * descending, take top-k. Returns k curves, each an array of {x, y}.
 *
 * @returns {Array<Array<{x:number, y:number}>>} length k (trailing curves may
 *          be empty if fewer than k eigenvalues exist)
 */
export function aggregatePooled(theory, sigmaHistory, sigmaStar, n, d, m, k, L = 2) {
  const groupNames = theory === 'full' ? FULL_GROUPS : GN_GROUPS;
  const evaluated = evalTheory(theory, sigmaHistory, sigmaStar, n, d, m, L);
  const curves = Array.from({ length: k }, () => []);
  for (const e of evaluated) {
    const all = [];
    // Iterate the theory's plotting groups (FULL_GROUPS includes idle_null, so
    // its zeros enter the ranking but sort to the bottom — they surface only
    // once k exceeds the count of nonzero eigenvalues).
    for (const g of groupNames) {
      const vals = e.groups[g];
      if (vals) for (const v of vals) all.push(v);
    }
    all.sort((a, b) => b - a);
    for (let rank = 0; rank < k; rank++) {
      if (rank < all.length) curves[rank].push({ x: e.iteration, y: all[rank] });
    }
  }
  return curves;
}

/**
 * PER-CLASS top-k for one theory. For each group with k>0, sort that group's
 * values descending per timepoint and take its own top-k.
 *
 * @param {Object<string,number>} perClassK  group -> k
 * @returns {Object<string, Array<Array<{x:number,y:number}>>>}
 *          group -> array of curves (each {x,y}[])
 */
export function aggregatePerClass(theory, sigmaHistory, sigmaStar, n, d, m, perClassK, L = 2) {
  const groupNames = theory === 'full' ? FULL_GROUPS : GN_GROUPS;
  const evaluated = evalTheory(theory, sigmaHistory, sigmaStar, n, d, m, L);
  const out = {};
  for (const group of groupNames) {
    const k = perClassK[group] || 0;
    if (k <= 0) continue;
    const curves = Array.from({ length: k }, () => []);
    for (const e of evaluated) {
      const vals = (e.groups[group] || []).slice().sort((a, b) => b - a);
      for (let rank = 0; rank < k; rank++) {
        if (rank < vals.length) curves[rank].push({ x: e.iteration, y: vals[rank] });
      }
    }
    out[group] = curves;
  }
  return out;
}
