// ============================================================================
// RNG - Shared seeded random utilities
// ============================================================================
// Single source of truth for the deterministic PRNG used everywhere in this
// playground. Same algorithm as before (mulberry32 + Box-Muller); previously
// duplicated verbatim across inputs.js, data-generator.js, and matrix.js.

/**
 * Mulberry32 seeded PRNG. Returns a function that yields uniform [0, 1).
 */
export function mulberry32(seed) {
  let a = seed | 0;
  return function() {
    a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

/**
 * Standard normal sample via Box-Muller, using a uniform RNG.
 */
export function seededRandn(rng) {
  const u1 = rng();
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}
